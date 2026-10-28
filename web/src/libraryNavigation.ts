import { paperCollections } from './paperCollections';
import type { Paper } from './types';
import {readInterviewRoute,interviewHref} from './interviewNavigation';

export type ExerciseSort='difficulty'|'difficulty-desc'|'duration'|'duration-desc';
export const isExerciseSort=(value:unknown):value is ExerciseSort=>typeof value==='string'&&['difficulty','difficulty-desc','duration','duration-desc'].includes(value);
export type LibrarySort = 'default' | 'newest' | 'oldest' | 'title' | ExerciseSort;
export type LibraryPageSize = 20 | 50 | 100;
export type PaperRegistryView={query:string;category:string;sort:LibrarySort;page:number};
export type LibraryFilters = { query: string; category: string; tag: string; difficulty: string; runtime: string; state: string; sort?: LibrarySort; view?:'modules'; company?:string; company_days?:string; page?:number; page_size?:LibraryPageSize };
export const emptyLibraryFilters = (): LibraryFilters => ({ query: '', category: 'all', tag: 'all', difficulty: 'all', runtime: 'all', state: 'all' });
export const routePath = (route: string) => route.split('?')[0];
const practicePath = /^(?:[1-9]\d*|(?:foundations|rl|architecture|diffusion|finetuning|optimization|representation|world_models|retrieval|self_evolving|multimodal|robotics|ai_infra|diffusion_rl)\/[a-z0-9_-]+|papers\/[a-z0-9_-]+)$/;

function parts(route: string) {
  if (route.length > 8192) return;
  const hash = route.startsWith('#');
  const raw = hash ? route.replace(/^#\/?/, '') : route;
  const query = raw.indexOf('?');
  try {
    return { path: hash ? decodeURIComponent(query < 0 ? raw : raw.slice(0, query)) : query < 0 ? raw : raw.slice(0, query), params: new URLSearchParams(query < 0 ? '' : raw.slice(query + 1)) };
  } catch { return; }
}
function only(params: URLSearchParams, names: string[]) {
  return [...params.keys()].every(key => names.includes(key) && params.getAll(key).length === 1);
}
const bounded = (value: string, length: number) => Array.from(value).length <= length && !/[\u0000-\u001f\u007f]/.test(value);
const topicOrder = new Intl.Collator('en', { numeric: true, sensitivity: 'base' });
export const sameLibraryTag = (left: string, right: string) => left.toLowerCase() === right.toLowerCase();

/** Present one named topic for case aliases without changing catalog metadata or URLs. */
export function libraryTags(papers: Paper[], category = 'all'): string[] {
  const tags = new Map<string, string>();
  for (const paper of papers) if (category === 'all' || paper.category === category) {
    for (const tag of paper.tags) if (!tags.has(tag.toLowerCase())) tags.set(tag.toLowerCase(), tag);
  }
  return [...tags.values()].sort(topicOrder.compare);
}

export function readLibraryRoute(route: string): { collectionId?: string; filters: LibraryFilters } | undefined {
  const value = parts(route);
  if (!value || !only(value.params, ['q', 'category', 'tag', 'difficulty', 'runtime', 'state', 'sort','view','company','company_days','page','page_size'])) return;
  const collectionId = /^(?:collections|lists)\/([a-z0-9_-]+)$/.exec(value.path)?.[1];
  if (value.path && (!collectionId || !paperCollections.some(collection => collection.id === collectionId))) return;
  const filters = { query: value.params.get('q') || '', category: value.params.get('category') || 'all', tag: value.params.get('tag') || 'all',
    difficulty: value.params.get('difficulty') || 'all', runtime: value.params.get('runtime') || 'all', state: value.params.get('state') || 'all' };
  if (!bounded(filters.query, 200) || !bounded(filters.tag, 80) || !/^[a-z0-9-]{1,80}$/.test(filters.category)
    || !['all', 'E', 'M', 'H'].includes(filters.difficulty) || !['all', 'browser', 'server'].includes(filters.runtime)
    || !['all', 'todo', 'started', 'done'].includes(filters.state)) return;
  const company=value.params.get('company'),days=value.params.get('company_days');
  if(company!==null&&!/^[a-z][a-z0-9-]{0,39}$/.test(company)||days!==null&&(!company||!['30','90','365','all'].includes(days)))return;
  const companyScope=company?{company,...(days&&days!=='365'?{company_days:days}:{})}:{};
  const view=value.params.get('view');
  if(view!==null&&!['papers','modules'].includes(view))return;
  const sort = value.params.get('sort');
  if (sort && !['default', 'newest', 'oldest', 'title'].includes(sort)&&!isExerciseSort(sort)) return;
  if(isExerciseSort(sort)&&(view!=='modules'||company||paperCollections.find(item=>item.id===collectionId)?.kind==='course'))return;
  const page=value.params.get('page')??'1',size=value.params.get('page_size')??'20';
  if(!/^[1-9]\d{0,4}$/.test(page)||+page>10000||!['20','50','100'].includes(size))return;
  return { collectionId, filters:{...filters,...companyScope,...(view==='modules'?{view:'modules' as const}:{}),...(sort&&sort!=='default'?{sort:sort as LibrarySort}:{}),...(+page>1?{page:+page}:{}),...(size!=='20'?{page_size:+size as LibraryPageSize}:{})} };
}

export function libraryHref(collectionId: string | undefined, filters: LibraryFilters): string {
  const page=filters.page??1,size=filters.page_size??20;
  if(!Number.isSafeInteger(page)||page<1||page>10000||![20,50,100].includes(size))throw Error('Invalid library pagination');
  const params = new URLSearchParams();
  if (filters.query) params.set('q', filters.query);
  for (const key of ['category', 'tag', 'difficulty', 'runtime', 'state'] as const) if (filters[key] !== 'all') params.set(key, filters[key]);
  if (filters.sort && filters.sort !== 'default') params.set('sort', filters.sort);
  if(filters.view)params.set('view',filters.view);
  if(filters.company)params.set('company',filters.company);
  if(filters.company_days&&filters.company_days!=='365')params.set('company_days',filters.company_days);
  if(page!==1)params.set('page',String(page));
  if(size!==20)params.set('page_size',String(size));
  const base = collectionId ? `#/collections/${collectionId}` : '#';
  const href = base + (params.size ? `${collectionId ? '?' : '/?'}${params}` : '');
  if (!readLibraryRoute(href)) throw Error('Invalid library filters');
  return href;
}

export function readPaperRegistryRoute(route:string):PaperRegistryView|undefined{
  const value=parts(route),index=route.indexOf('?'),query=index<0?'':route.slice(index+1);
  if(!value||value.path!=='paper-registry'||!only(value.params,['q','category','sort','page'])||!/^[a-z0-9%_=&+.,~-]*$/i.test(query))return;
  try{decodeURIComponent(query);}catch{return;}
  const page=value.params.get('page')??'1';
  if(!/^[1-9]\d{0,2}$/.test(page)||+page>501)return;
  value.params.delete('page');
  const view=readLibraryRoute('?'+value.params);
  return view?{query:view.filters.query,category:view.filters.category,sort:view.filters.sort||'default',page:+page}:undefined;
}
export function paperRegistryHref(view:PaperRegistryView):string{
  const params=new URLSearchParams(libraryHref(undefined,{...emptyLibraryFilters(),query:view.query,category:view.category,sort:view.sort}).split('?')[1]);
  if(view.page!==1)params.set('page',String(view.page));
  const href='#/paper-registry'+(params.size?'?'+params:'');
  if(!readPaperRegistryRoute(href))throw Error('Invalid paper registry view');
  return href;
}

/** Use stable paper numbers by default, with optional bibliography sorting. */
export function sortLibraryPapers(papers: Paper[], sort: LibrarySort = 'default'): Paper[] {
  if (sort === 'default'||isExerciseSort(sort)) return [...papers].sort((a, b) => a.number - b.number);
  return [...papers].sort((a, b) => {
    if (sort === 'title') return a.short_title.localeCompare(b.short_title, 'en', { numeric: true }) || a.number - b.number;
    if (!a.year || !b.year) return a.year ? -1 : b.year ? 1 : a.number - b.number;
    return (sort === 'newest' ? b.year - a.year : a.year - b.year) || a.number - b.number;
  });
}

/** Retire old runtime filters now that browser execution is the catalog baseline. */
export function normalizeLibraryFilters(filters: LibraryFilters, papers: Paper[]): LibraryFilters {
  const category = filters.category === 'all' || papers.some(paper => paper.category === filters.category) ? filters.category : 'all';
  const tag = filters.tag === 'all' || papers.some(paper => (category === 'all' || paper.category === category) && paper.tags.some(tag => sameLibraryTag(tag, filters.tag))) ? filters.tag : 'all';
  const {company,company_days,...rest}=filters;
  if(isExerciseSort(rest.sort)&&(rest.view!=='modules'||company))delete rest.sort;
  return { ...rest, category, tag, runtime: 'all',...(company?{company,...(company_days&&company_days!=='365'?{company_days}:{})}:{}) };
}

export type FavoriteView={query:string;state:string;list?:string;page?:number;page_size?:LibraryPageSize};
export function readFavoriteRoute(route:string):FavoriteView|undefined {
  const value=parts(route);
  if(!value||value.path!=='favorites'||!only(value.params,['q','state','list','page','page_size']))return;
  const list=value.params.get('list');
  if(list!==null&&(!/^[a-f0-9]{32}$/.test(list)||value.params.has('page')||value.params.has('page_size')))return;
  value.params.delete('list');
  const filters=readLibraryRoute('?'+value.params)?.filters;
  return filters?{query:filters.query,state:filters.state,...(list?{list}:{}),...(filters.page?{page:filters.page}:{}),...(filters.page_size?{page_size:filters.page_size}:{})}:undefined;
}
export function favoriteHref(view:FavoriteView):string {
  const params=new URLSearchParams(libraryHref(undefined,{...emptyLibraryFilters(),...view}).split('?')[1]);
  if(view.list!==undefined)params.set('list',view.list);
  const href='#/favorites'+(params.size?'?'+params:'');
  if(!readFavoriteRoute(href))throw Error('Invalid saved-problem view');
  return href;
}

/** Practice may return to one public post ID, without private or nested list context. */
export function discussionPracticeReturn(route:string):string|undefined{
  const path=route.replace(/^#\/?/,'');
  return /^community\/[a-f0-9]{32}$/.test(path)?`#/${path}`:undefined;
}

export function libraryReturnHref(route: string): string | undefined {
  const value = parts(route);
  if (!value || !practicePath.test(value.path) || !only(value.params, ['from']) || !value.params.has('from')) return;
  const from=value.params.get('from')||'',interview=readInterviewRoute(from);
  const registry=readPaperRegistryRoute(from);if(registry)return paperRegistryHref(registry);
  const discussion=discussionPracticeReturn(from);if(discussion)return discussion;
  const favorites=readFavoriteRoute(from);if(favorites)return favoriteHref(favorites);
  if(interview)return interviewHref(interview);
  const view = readLibraryRoute(from);
  return view ? libraryHref(view.collectionId, view.filters) : undefined;
}

export function withLibraryContext(path: string | number, returnTo?: string): string {
  if (!practicePath.test(String(path))) throw Error('Invalid practice destination');
  const registry=returnTo?readPaperRegistryRoute(returnTo):undefined,favorites=returnTo?readFavoriteRoute(returnTo):undefined,local=returnTo?discussionPracticeReturn(returnTo)||(favorites?favoriteHref(favorites):undefined)||(registry?paperRegistryHref(registry):undefined):undefined;
  if(local)return `#/${path}?${new URLSearchParams({from:local.slice(2)})}`;
  const base = `#/${path}`, view = returnTo ? readLibraryRoute(returnTo) : undefined,interview=returnTo?readInterviewRoute(returnTo):undefined;
  if (!view&&!interview) return base;
  const href = interview?interviewHref(interview):libraryHref(view!.collectionId, view!.filters);
  return href === '#' ? base : `${base}?${new URLSearchParams({ from: href.slice(2) })}`;
}

/** Account return targets may carry only a library view, never another continuation. */
export function libraryContinuation(route: string): string | undefined {
  const registry=readPaperRegistryRoute(route);if(registry)return paperRegistryHref(registry).slice(2);
  const view = readLibraryRoute(route);
  if (view) return libraryHref(view.collectionId, view.filters).replace(/^#\/?/, '');
  const back = libraryReturnHref(route), value = parts(route);
  return back && value ? withLibraryContext(value.path, back).slice(2) : undefined;
}
