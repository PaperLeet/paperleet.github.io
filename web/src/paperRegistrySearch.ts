import type {Paper,Problem} from './types';

/** arXiv versions, abstract pages and PDFs identify the same source. */
export function paperIdentity(value:string):string {
  const raw=value.normalize('NFKC').trim().replace(/^arxiv:\s*/i,'');
  const id=/^(?:https?:\/\/(?:www\.|export\.)?arxiv\.org\/(?:abs|pdf|html)\/)?((?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*\/\d{7}))(?:v\d+)?(?:\.pdf)?(?:[?#].*)?$/i.exec(raw);
  return id?`arxiv:${id[1].toLowerCase()}`:raw.toLowerCase();
}
export function searchPaperRegistry(papers:Paper[],problems:Problem[],query:string):Paper[]{
  const needle=paperIdentity(query),words=query.normalize('NFKC').trim().toLowerCase().split(/\s+/).filter(Boolean);
  const exercise=/^#([1-9]\d*)$/.exec(query.normalize('NFKC').trim());
  const byId=new Map(problems.map(problem=>[problem.id,problem]));
  return papers.filter(paper=>paper.kind!=='course').filter(paper=>{
    if(exercise)return paper.module_ids.some(id=>byId.get(id)?.number===Number(exercise[1]));
    if(!words.length)return true;
    if(needle===paperIdentity(paper.arxiv||paper.url))return true;
    const text=[paper.id,paper.number,`#${paper.number}`,paper.title,paper.short_title,paper.arxiv,paper.url,paper.venue,paper.year,...[paper.authors].flat(),...paper.tags,
      ...paper.module_ids.flatMap(id=>{const p=byId.get(id);return p?[p.id,p.number,`#${p.number}`,p.title,p.translations?.en?.title]:[];})].join(' ').normalize('NFKC').toLowerCase();
    return words.every(word=>text.includes(word));
  }).sort((a,b)=>a.number-b.number);
}
