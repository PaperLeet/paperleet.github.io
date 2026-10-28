import type {Paper, Problem} from './types';
import {paperCollections} from './paperCollections';

export type SearchCatalog = {papers: Paper[]; problems: Problem[]};
export type SearchMetadata = {title: string; description: string; canonical: string; heading: string; kind: 'home'|'paper'|'problem'|'collection'; id?: string};
export const siteOrigin = 'https://paperleet.com';
export const paperPath = (paper: Pick<Paper,'id'>) => `/papers/${paper.id}/`;
export const problemPath = (problem: Pick<Problem,'id'>) => `/problems/${problem.id}/`;
const concise = (value: string) => {
  const text = value.replace(/\s+/g,' ').trim();
  return text.length <= 180 ? text : text.slice(0,177).replace(/\s+\S*$/,'') + '…';
};

/** Shared by static HTML generation and client navigation; no invented paper dates or venues. */
export function searchMetadata(route: string, catalog: SearchCatalog): SearchMetadata | undefined {
  const path = route.replace(/^#\/?/,'').split('?')[0];
  if (!path) return {kind:'home', title:'PaperLeet - LeetCode for AI Researchers', heading:'Practice AI research algorithms',
    description:`PaperLeet - LeetCode for AI Researchers. Implement ${catalog.problems.length} Python exercises from research papers and prepare for machine learning interviews.`, canonical:siteOrigin+'/'};
  const problem = catalog.problems.find(row=>row.id===path||String(row.number)===path);
  if (problem) {
    const paper = catalog.papers.find(row=>row.id===problem.paper_id);
    const heading = problem.translations?.en?.title || problem.title;
    return {kind:'problem',id:problem.id,heading,title:`${paper?.short_title || 'AI Research'}: ${heading} in Python | PaperLeet`,
      description:concise(`${problem.module.goal_en} ${problem.module.scope_en}`),canonical:siteOrigin+problemPath(problem)};
  }
  const paper = catalog.papers.find(row=>path===`papers/${row.id}`);
  if (paper) return {kind:'paper',id:paper.id,heading:paper.short_title,
    title:`${paper.short_title}${paper.venue || paper.year ? ` (${paper.venue || paper.year})` : ''}: ${paper.kind==='course'?'Python Course':'Python Implementation'} | PaperLeet`,
    description:concise(`${paper.summary_en} ${paper.scope_en}`),canonical:siteOrigin+paperPath(paper)};
  const collection = paperCollections.find(row=>path===`collections/${row.id}`);
  if (collection) return {kind:'collection',id:collection.id,heading:collection.name_en,title:`${collection.name_en}: ${collection.readingCount?'Reading List & Python Exercises':'Python Exercises for AI Researchers'} | PaperLeet`,
    description:collection.readingCount?`Explore ${collection.readingCount} ${collection.name_en} papers by topic, read original sources, and practice available implementations in Python on PaperLeet.`:`Practice ${collection.name_en} in Python with research-based exercises, implementation guides, and tests on PaperLeet.`,canonical:`${siteOrigin}/collections/${collection.id}/`};
}

export function applySearchMetadata(metadata: SearchMetadata | undefined) {
  document.title = metadata?.title || 'PaperLeet';
  const values: [string,string,string][] = [
    ['name','description',metadata?.description||'Practice research algorithms in Python on PaperLeet.'],
    ['property','og:title',document.title],['name','twitter:title',document.title],
    ['property','og:description',metadata?.description||'Practice research algorithms in Python on PaperLeet.'],
    ['name','twitter:description',metadata?.description||'Practice research algorithms in Python on PaperLeet.'],
    ['property','og:url',metadata?.canonical||siteOrigin+'/'],
  ];
  for (const [key,name,content] of values) {
    let element = document.head.querySelector<HTMLMetaElement>(`meta[${key}="${name}"]`);
    if (!element) {element=document.createElement('meta');element.setAttribute(key,name);document.head.append(element);}
    element.content=content;
  }
  let canonical=document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  if (!canonical) {canonical=document.createElement('link');canonical.rel='canonical';document.head.append(canonical);}
  canonical.href=metadata?.canonical||siteOrigin+'/';
}
