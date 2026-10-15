import { status, type Paper, type Problem, type Saved } from './types';
import { emptyLibraryFilters, libraryHref } from './libraryNavigation';

export type AchievementIcon = 'spark' | 'layers' | 'code' | 'gem' | 'crown' | 'book' | 'library' | 'compass' | 'mountain' | 'cpu' | 'target' | 'orbit' | 'scan' | 'wand' | 'stars';
type Metric = 'solved' | 'papers' | 'directions' | 'hard' | 'core';
export type AchievementDefinition = {
  id: string; title: [string,string]; description: [string,string]; icon: AchievementIcon;
  tier: 'bronze' | 'silver' | 'gold' | 'prism'; target: number; metric: Metric; category?: string;
};
export type Achievement = AchievementDefinition & {current: number; unlocked: boolean; href: string; earnedAt?:string};
export function newlyUnlocked(previous: ReadonlySet<string> | undefined, badges: Achievement[]) {
  return previous ? badges.filter(badge => badge.unlocked && !previous.has(badge.id)) : [];
}
export const achievementDefinitions: readonly AchievementDefinition[] = [
  {id:'first-light',title:['初次点亮','First light'],description:['完整通过第一个模块的测试。','Pass the complete test suite for your first module.'],icon:'spark',tier:'bronze',target:1,metric:'solved'},
  {id:'finding-rhythm',title:['渐入佳境','Finding your rhythm'],description:['累计完整通过 5 个不同模块。','Pass the complete suites for 5 distinct modules.'],icon:'layers',tier:'bronze',target:5,metric:'solved'},
  {id:'ten-proofs',title:['十次求证','Ten proofs'],description:['累计完整通过 10 个不同模块。','Pass the complete suites for 10 distinct modules.'],icon:'code',tier:'silver',target:10,metric:'solved'},
  {id:'deep-practice',title:['深度实践','Deep practice'],description:['累计完整通过 25 个不同模块。','Pass the complete suites for 25 distinct modules.'],icon:'gem',tier:'gold',target:25,metric:'solved'},
  {id:'algorithm-collector',title:['算法收藏家','Algorithm collector'],description:['累计完整通过 50 个不同模块。','Pass the complete suites for 50 distinct modules.'],icon:'crown',tier:'prism',target:50,metric:'solved'},
  {id:'paper-complete',title:['读懂一篇','Paper into practice'],description:['完成一篇论文的全部核心模块。','Complete every core module in one paper.'],icon:'book',tier:'bronze',target:1,metric:'papers'},
  {id:'paper-explorer',title:['论文漫游者','Paper explorer'],description:['完成 3 篇论文的全部核心模块。','Complete every core module in 3 papers.'],icon:'library',tier:'silver',target:3,metric:'papers'},
  {id:'across-disciplines',title:['跨界探索','Across disciplines'],description:['在 3 个研究方向中各完整通过至少一个模块。','Pass at least one complete module in each of 3 research directions.'],icon:'compass',tier:'silver',target:3,metric:'directions'},
  {id:'hard-problem',title:['迎难而上','Rise to the challenge'],description:['完整通过一个困难模块。','Pass the complete suite for a Hard module.'],icon:'mountain',tier:'gold',target:1,metric:'hard'},
  {id:'architecture',title:['架构解码者','Architecture decoder'],description:['完整通过 3 个架构与推理模块。','Pass 3 complete Architecture modules.'],icon:'cpu',tier:'silver',target:3,metric:'solved',category:'architecture'},
  {id:'reinforcement',title:['奖励塑造者','Reward shaper'],description:['完整通过 3 个强化学习模块。','Pass 3 complete Reinforcement Learning modules.'],icon:'target',tier:'silver',target:3,metric:'solved',category:'reinforcement-learning'},
  {id:'generative',title:['生成艺术家','Generative thinker'],description:['完整通过 3 个生成模型模块。','Pass 3 complete Generative Models modules.'],icon:'orbit',tier:'silver',target:3,metric:'solved',category:'generative-models'},
  {id:'representation',title:['表征探索者','Representation explorer'],description:['完整通过 3 个表征学习模块。','Pass 3 complete Representation Learning modules.'],icon:'scan',tier:'silver',target:3,metric:'solved',category:'representation'},
  {id:'post-training',title:['对齐实践者','Alignment practitioner'],description:['完整通过 3 个后训练与微调模块。','Pass 3 complete Post-training modules.'],icon:'wand',tier:'silver',target:3,metric:'solved',category:'post-training'},
  {id:'all-core',title:['群星点亮','A sky full of ideas'],description:['完整通过当前题库中的全部核心模块。','Pass every core module in the current catalog.'],icon:'stars',tier:'prism',target:1,metric:'core'},
];

/** Derive rewards from complete-suite passes; never store or sync grades here. */
export function achievementProgress(problems: Problem[], papers: Paper[], saved: Record<string,Saved>) {
  const modules = new Map(problems.map(problem => [problem.id,problem]));
  const byPaper = new Map(papers.map(paper => [paper.id,paper]));
  const solved = [...modules.values()].filter(problem => status(problem,saved[problem.id]) === 'done');
  const passed = new Set(solved.map(problem => problem.id));
  const core = [...modules.values()].filter(problem => !problem.module.optional);
  const categories = new Map<string,number>();
  for (const problem of solved) {
    const category = byPaper.get(problem.paper_id)?.category;
    if (category) categories.set(category,(categories.get(category) || 0) + 1);
  }
  const completedPapers = [...byPaper.values()].filter(paper => {
    const ids = [...new Set(paper.module_ids)];
    if (!ids.length || ids.some(id => !modules.has(id))) return false;
    const required = ids.filter(id => !modules.get(id)!.module.optional);
    return required.length > 0 && required.every(id => passed.has(id));
  }).length;
  const metrics: Record<Metric,number> = {solved:solved.length,papers:completedPapers,directions:categories.size,hard:solved.filter(problem => problem.difficulty === 'H').length,core:core.filter(problem => passed.has(problem.id)).length};
  const badges: Achievement[] = achievementDefinitions.map(definition => {
    const target = definition.metric === 'core' ? Math.max(1,core.length) : definition.target;
    const current = definition.category ? categories.get(definition.category) || 0 : metrics[definition.metric];
    return {...definition,target,current,unlocked:current >= target,href:libraryHref(undefined,{...emptyLibraryFilters(),...(definition.category ? {category:definition.category} : {}),...(definition.metric === 'hard' ? {difficulty:'H'} : {})})};
  });
  const earned = badges.filter(badge => badge.unlocked);
  const next = badges.filter(badge => !badge.unlocked).sort((a,b) => Math.min(b.current / b.target,1) - Math.min(a.current / a.target,1))[0];
  return {badges,earned,next,solved:solved.length,completedPapers};
}
