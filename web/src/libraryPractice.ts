import { status } from './types';
import type { Paper, Problem, Saved } from './types';
import { paperCollections, papersInCollection } from './paperCollections';
import { normalizeLibraryFilters, type LibraryFilters } from './libraryNavigation';
import { parseArxivUrl } from './communityPapers';

type Practice = { problems: Problem[]; saved: Record<string, Saved> };
export function paperModules(paper: Paper, problems: Problem[]) {
  return paper.module_ids.flatMap((id) => {
    const problem = problems.find((p) => p.id === id);
    return problem ? [problem] : [];
  });
}
export function matchingPaperModules(paper: Paper, problems: Problem[], query = "", difficulty = "all", runtime = "all") {
  const normalized = query.trim().toLowerCase();
  const terms = normalized.split(/\s+/);
  const reference = normalized ? (parseArxivUrl(normalized.replace(/[?#].*$/, '')) ||
    parseArxivUrl(`https://arxiv.org/abs/${normalized.replace(/^arxiv:\s*/, '')}`)) : undefined;
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
    return (difficulty === "all" || problem.difficulty === difficulty) && (paperMatches || moduleMatches) &&
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
export function selectLibraryPractice(papers: Paper[], problems: Problem[], saved: Record<string, Saved>, view?: { collectionId?: string; filters: LibraryFilters }) {
  if (!view) return { papers, problems };
  const listed = papersInCollection(papers, paperCollections.find(collection => collection.id === view.collectionId));
  const filters = normalizeLibraryFilters(view.filters, listed);
  const visible = listed.filter(paper =>
    (filters.category === 'all' || paper.category === filters.category) &&
    matchingPaperModules(paper, problems, filters.query, filters.difficulty, filters.runtime).length > 0 &&
    (filters.state === 'all' || paperProgress(paper, { problems, saved }).state === filters.state) &&
    (filters.tag === 'all' || paper.tags.includes(filters.tag)));
  return { papers: visible, problems: visible.flatMap(paper => matchingPaperModules(paper, problems, filters.query, filters.difficulty, filters.runtime)) };
}
