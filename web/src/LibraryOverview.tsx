import { CheckCircle2, Code2, ArrowRight } from 'lucide-react';
import { useI18n } from './i18n';
import { status, type Problem, type Saved } from './types';

export default function LibraryOverview({ problems, saved, ready, difficulty, onDifficulty, onGuide }: {
  problems: Problem[];
  saved: Record<string, Saved>;
  ready: boolean;
  difficulty: string;
  onDifficulty: (value: string) => void;
  onGuide: () => void;
}) {
  const { t } = useI18n();
  const core = problems.filter(problem => !problem.module.optional);
  const solved = core.filter(problem => status(problem, saved[problem.id]) === 'done').length;
  const started = core.filter(problem => status(problem, saved[problem.id]) === 'started').length;
  const percent = ready && core.length ? Math.round(solved / core.length * 100) : 0;
  return <aside className="library-overview" aria-label={t('练习概览', 'Practice overview')}>
    <section className="library-progress-card">
      <header><h2>{t('我的进度', 'Your progress')}</h2><CheckCircle2 size={16} aria-hidden="true" /></header>
      <p>{t('当前合集 · 核心模块', 'This collection · core modules')}</p>
      <div className="library-progress-ring" role="img" aria-label={ready ? t(`${solved} / ${core.length} 核心模块已通过，${percent}%`, `${solved} of ${core.length} core modules solved, ${percent}%`) : t('正在加载进度', 'Loading progress')} style={{ background: `conic-gradient(var(--green) ${percent}%, var(--surface3) 0)` }}>
        <div><b>{ready ? solved : '—'}<span> / {core.length}</span></b><small>{t('已通过', 'Solved')}</small></div>
      </div>
      <div className="library-progress-summary"><span><i />{t('进行中', 'In progress')}<b>{ready ? started : '—'}</b></span><span><i />{t('未开始', 'Not started')}<b>{ready ? core.length - solved - started : '—'}</b></span></div>
      <div className="library-difficulty-list" aria-label={t('按难度练习', 'Practice by difficulty')}>
        {(['E', 'M', 'H'] as const).map(value => {
          const group = core.filter(problem => problem.difficulty === value);
          const count = group.filter(problem => status(problem, saved[problem.id]) === 'done').length;
          return <button key={value} className={`difficulty-summary ${value}`} aria-pressed={difficulty === value} onClick={() => onDifficulty(difficulty === value ? 'all' : value)}>
            <span>{({ E: t('简单', 'Easy'), M: t('中等', 'Medium'), H: t('困难', 'Hard') })[value]}</span><b>{ready ? count : '—'}<small> / {group.length}</small></b>
            <div aria-hidden="true"><i style={{ width: `${ready && group.length ? count / group.length * 100 : 0}%` }} /></div>
          </button>;
        })}
      </div>
    </section>
    <section className="library-help-card"><Code2 size={19} aria-hidden="true" /><h3>{t('从一个核心函数开始', 'One core function at a time')}</h3><p>{t('读题、实现、测试，再用实验理解算法。草稿自动保存在当前工作区。', 'Read, implement, test, then explore the algorithm. Drafts save automatically in your workspace.')}</p><button onClick={onGuide}>{t('查看练习指南', 'Practice guide')}<ArrowRight size={14} aria-hidden="true" /></button></section>
  </aside>;
}
