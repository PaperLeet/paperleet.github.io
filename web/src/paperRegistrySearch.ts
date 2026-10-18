import type {Paper,Problem} from './types';

/** arXiv versions, abstract pages and PDFs identify the same source. */
export function paperIdentity(value:string):string {
  const raw=value.trim().replace(/^arxiv:\s*/i,'');
  const id=/^(?:https?:\/\/(?:www\.|export\.)?arxiv\.org\/(?:abs|pdf|html)\/)?((?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*\/\d{7}))(?:v\d+)?(?:\.pdf)?(?:[?#].*)?$/i.exec(raw);
  return id?`arxiv:${id[1].toLowerCase()}`:raw.toLowerCase();
}
export function searchPaperRegistry(papers:Paper[],problems:Problem[],query:string):Paper[]{
  const needle=paperIdentity(query),words=query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  const byId=new Map(problems.map(problem=>[problem.id,problem]));
  return papers.filter(paper=>paper.kind!=='course').filter(paper=>{
    if(!words.length)return true;
    if(needle===paperIdentity(paper.arxiv||paper.url))return true;
    const text=[paper.title,paper.short_title,paper.arxiv,paper.url,paper.venue,paper.year,...[paper.authors].flat(),...paper.tags,
      ...paper.module_ids.flatMap(id=>{const p=byId.get(id);return p?[p.title,p.translations?.en?.title]:[];})].join(' ').toLocaleLowerCase();
    return words.every(word=>text.includes(word));
  }).sort((a,b)=>a.number-b.number);
}
