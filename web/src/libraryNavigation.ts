import { paperCollections } from './paperCollections';
import type { Paper } from './types';
import {readInterviewRoute,interviewHref} from './interviewNavigation';

export type LibrarySort = 'default' | 'newest' | 'oldest' | 'title';
export type LibraryFilters = { query: string; category: string; tag: string; difficulty: string; runtime: string; state: string; sort?: LibrarySort };
export const emptyLibraryFilters = (): LibraryFilters => ({ query: '', category: 'all', tag: 'all', difficulty: 'all', runtime: 'all', state: 'all' });
export const routePath = (route: string) => route.split('?')[0];
const practicePath = /^(?:[1-9]\d*|(?:foundations|rl|architecture|diffusion|finetuning|optimization|representation|world_models|retrieval)\/[a-z0-9_-]+|papers\/[a-z0-9_-]+)$/;

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
  if (!value || !only(value.params, ['q', 'category', 'tag', 'difficulty', 'runtime', 'state', 'sort'])) return;
  const collectionId = /^(?:collections|lists)\/([a-z0-9_-]+)$/.exec(value.path)?.[1];
  if (value.path && (!collectionId || !paperCollections.some(collection => collection.id === collectionId))) return;
  const filters = { query: value.params.get('q') || '', category: value.params.get('category') || 'all', tag: value.params.get('tag') || 'all',
    difficulty: value.params.get('difficulty') || 'all', runtime: value.params.get('runtime') || 'all', state: value.params.get('state') || 'all' };
  if (!bounded(filters.query, 200) || !bounded(filters.tag, 80) || !/^[a-z0-9-]{1,80}$/.test(filters.category)
    || !['all', 'E', 'M', 'H'].includes(filters.difficulty) || !['all', 'browser', 'server'].includes(filters.runtime)
    || !['all', 'todo', 'started', 'done'].includes(filters.state)) return;
  const sort = value.params.get('sort');
  if (sort && !['default', 'newest', 'oldest', 'title'].includes(sort)) return;
  return { collectionId, filters: sort && sort !== 'default' ? { ...filters, sort: sort as LibrarySort } : filters };
}

export function libraryHref(collectionId: string | undefined, filters: LibraryFilters): string {
  const params = new URLSearchParams();
  if (filters.query) params.set('q', filters.query);
  for (const key of ['category', 'tag', 'difficulty', 'runtime', 'state'] as const) if (filters[key] !== 'all') params.set(key, filters[key]);
  if (filters.sort && filters.sort !== 'default') params.set('sort', filters.sort);
  const base = collectionId ? `#/collections/${collectionId}` : '#';
  const href = base + (params.size ? `${collectionId ? '?' : '/?'}${params}` : '');
  if (!readLibraryRoute(href)) throw Error('Invalid library filters');
  return href;
}

/** Preserve the catalog's discovery order until the learner chooses a sort. */
export function sortLibraryPapers(papers: Paper[], sort: LibrarySort = 'default'): Paper[] {
  if (sort === 'default') return papers;
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
  return { ...filters, category, tag, runtime: 'all' };
}

export function readFavoriteRoute(route:string):{query:string;state:string}|undefined {
  const value=parts(route);
  if(!value||value.path!=='favorites'||!only(value.params,['q','state']))return;
  const query=value.params.get('q')||'',state=value.params.get('state')||'all';
  return bounded(query,200)&&['all','todo','started','done'].includes(state)?{query,state}:undefined;
}
export function favoriteHref(view:{query:string;state:string}):string {
  const params=new URLSearchParams();if(view.query)params.set('q',view.query);if(view.state!=='all')params.set('state',view.state);
  return '#/favorites'+(params.size?'?'+params:'');
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
  const discussion=discussionPracticeReturn(from);if(discussion)return discussion;
  const favorites=readFavoriteRoute(from);if(favorites)return favoriteHref(favorites);
  if(interview)return interviewHref(interview);
  const view = readLibraryRoute(from);
  return view ? libraryHref(view.collectionId, view.filters) : undefined;
}

export function withLibraryContext(path: string | number, returnTo?: string): string {
  if (!practicePath.test(String(path))) throw Error('Invalid practice destination');
  const favorites=returnTo?readFavoriteRoute(returnTo):undefined,local=returnTo?discussionPracticeReturn(returnTo)||(favorites?favoriteHref(favorites):undefined):undefined;
  if(local)return `#/${path}?${new URLSearchParams({from:local.slice(2)})}`;
  const base = `#/${path}`, view = returnTo ? readLibraryRoute(returnTo) : undefined,interview=returnTo?readInterviewRoute(returnTo):undefined;
  if (!view&&!interview) return base;
  const href = interview?interviewHref(interview):libraryHref(view!.collectionId, view!.filters);
  return href === '#' ? base : `${base}?${new URLSearchParams({ from: href.slice(2) })}`;
}

/** Account return targets may carry only a library view, never another continuation. */
export function libraryContinuation(route: string): string | undefined {
  const view = readLibraryRoute(route);
  if (view) return libraryHref(view.collectionId, view.filters).replace(/^#\/?/, '');
  const back = libraryReturnHref(route), value = parts(route);
  return back && value ? withLibraryContext(value.path, back).slice(2) : undefined;
}
