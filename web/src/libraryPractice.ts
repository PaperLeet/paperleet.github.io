import { status } from './types';
import type { Paper, Problem, Saved } from './types';
import { paperCollections, papersInCollection } from './paperCollections';
import { normalizeLibraryFilters, sortLibraryPapers, sameLibraryTag, type LibraryFilters } from './libraryNavigation';
import { parseArxivUrl } from './arxivLinks';

type Practice = { problems: Problem[]; saved: Record<string, Saved> };
export function paperModules(paper: Paper, problems: Problem[]) {
  return paper.module_ids.flatMap((id) => {
    const problem = problems.find((p) => p.id === id);
    return problem ? [problem] : [];
  });
}
export function matchingPaperModules(paper: Paper, problems: Problem[], query = "", difficulty = "all", runtime = "all", moduleNumbers = false) {
  const normalized = query.trim().toLowerCase();
  const terms = normalized.split(/\s+/);
  const reference = normalized ? (parseArxivUrl(normalized.replace(/[?#].*$/, '')) ||
    parseArxivUrl(`https://arxiv.org/abs/${normalized.replace(/^arxiv:\s*/, '')}`)) : undefined;
  const moduleNumber=moduleNumbers&&/^[1-9]\d*$/.test(normalized)?Number(normalized):undefined;
  const referenceId = reference?.arxivId.replace(/v\d+$/, '');
  const paperReference = paper.arxiv || parseArxivUrl(paper.url)?.arxivId;
  const paperText = [paper.number, paper.id, paper.title, paper.short_title, paper.summary, paper.summary_en,
    Array.isArray(paper.authors) ? paper.authors.join(' ') : paper.authors, paper.venue, paper.year, paper.arxiv, paper.url, ...paper.tags]
    .join(" ").toLowerCase();
  const paperMatches = terms.every(term => paperText.includes(term)) ||
    !!referenceId && paperReference?.toLowerCase().replace(/v\d+$/, '') === referenceId;
  return paperModules(paper, problems).filter((problem) => {
    const moduleText = [problem.title, problem.translations?.en?.title, problem.module.goal, problem.module.goal_en]
      .join(" ").toLowerCase();
    const moduleMatches = terms.every(term => paperText.includes(term) || moduleText.includes(term));
    return (difficulty === "all" || problem.difficulty === difficulty) && (moduleNumber===undefined?paperMatches||moduleMatches:problem.number===moduleNumber) &&
      (runtime === "all" || (runtime === "browser") === !!problem.browser_judge);
  });
}
export function paperProgress(paper: Paper, { problems, saved }: Practice) {
  const modules = paperModules(paper, problems).filter((p) => !p.module.optional);
  const completed = modules.filter((p) => status(p, saved[p.id]) === "done").length;
  const started = modules.some((p) => status(p, saved[p.id]) !== "todo");
  return {
    completed,
    total: modules.length,
    state: completed > 0 && completed === modules.length ? "done" : started ? "started" : "todo",
  };
}


/** Use the same collection and filters for library cards and every practice entry point. */
export function selectLibraryPractice(papers: Paper[], problems: Problem[], saved: Record<string, Saved>, view?: { collectionId?: string; filters: LibraryFilters }, companyModules?:ReadonlySet<string>,locale='zh') {
  if (!view) return { papers, problems };
  const collection = paperCollections.find(collection => collection.id === view.collectionId);
  const listed = papersInCollection(papers, collection);
  const filters = normalizeLibraryFilters(view.filters, listed);
  if(filters.company&&!companyModules)return {papers:[],problems:[]};
  const matching=(paper:Paper)=>matchingPaperModules(paper,problems,filters.query,filters.difficulty,filters.runtime,filters.view==='modules').filter(problem=>!filters.company||companyModules!.has(problem.id));
  const moduleStatus=collection?.kind==='course'||!!filters.company||filters.view==='modules';
  const visible = listed.filter(paper =>
    (filters.category === 'all' || paper.category === filters.category) &&
    matching(paper).length > 0 &&
    (moduleStatus || filters.state === 'all' || paperProgress(paper, { problems, saved }).state === filters.state) &&
    (filters.tag === 'all' || paper.tags.some(tag => sameLibraryTag(tag, filters.tag))));
  const modules = visible.flatMap(matching)
    .filter(problem => !moduleStatus || filters.state === 'all' || status(problem, saved[problem.id]) === filters.state);
  if(filters.company){const rank=new Map([...companyModules!].map((id,index)=>[id,index]));modules.sort((a,b)=>(rank.get(a.id)??Infinity)-(rank.get(b.id)??Infinity));}
  else if(filters.view==='modules'&&collection?.kind!=='course'){
    if(filters.sort==='title'){const title=(p:Problem)=>locale==='en'?p.translations?.en?.title||p.title:p.title;modules.sort((a,b)=>title(a).localeCompare(title(b),locale==='en'?'en':'zh-CN',{numeric:true})||a.number-b.number);}
    else if(filters.sort==='newest'||filters.sort==='oldest'){const rank=new Map(sortLibraryPapers(visible,filters.sort).flatMap(p=>p.module_ids).map((id,index)=>[id,index]));modules.sort((a,b)=>rank.get(a.id)!-rank.get(b.id)!);}
    else modules.sort((a,b)=>a.number-b.number);
  }

  return { papers: visible.filter(paper => modules.some(problem => paper.module_ids.includes(problem.id))), problems: modules };
}
