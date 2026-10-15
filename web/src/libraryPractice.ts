import { status } from './types';
import type { Paper, Problem, Saved } from './types';
import { paperCollections, papersInCollection } from './paperCollections';
import { normalizeLibraryFilters, type LibraryFilters } from './libraryNavigation';

type Practice = { problems: Problem[]; saved: Record<string, Saved> };
export function paperModules(paper: Paper, problems: Problem[]) {
  return paper.module_ids.flatMap((id) => {
    const problem = problems.find((p) => p.id === id);
    return problem ? [problem] : [];
  });
}
export function matchingPaperModules(paper: Paper, problems: Problem[], query = "", difficulty = "all", runtime = "all") {
  const normalized = query.trim().toLowerCase();
  const paperMatches = [paper.number, paper.title, paper.short_title, paper.summary, paper.summary_en, ...paper.tags]
    .join(" ").toLowerCase().includes(normalized);
  return paperModules(paper, problems).filter((problem) => {
    const moduleMatches = [problem.title, problem.translations?.en?.title, problem.module.goal, problem.module.goal_en]
      .join(" ").toLowerCase().includes(normalized);
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
