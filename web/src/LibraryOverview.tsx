import { CheckCircle2 } from 'lucide-react';
import { useI18n } from './i18n';
import { status, type Problem, type Saved } from './types';

export default function LibraryOverview({ problems, saved, ready, difficulty, onDifficulty }: {
  problems: Problem[];
  saved: Record<string, Saved>;
  ready: boolean;
  difficulty: string;
  onDifficulty: (value: string) => void;
}) {
  const { t } = useI18n();
  const core = problems.filter(problem => !problem.module.optional);
  const solved = core.filter(problem => status(problem, saved[problem.id]) === 'done').length;
  const started = core.filter(problem => status(problem, saved[problem.id]) === 'started').length;
  const percent = ready && core.length ? Math.round(solved / core.length * 100) : 0;
  return <aside className="library-overview" aria-label={t('练习概览', 'Practice overview')}>
    <section className="library-progress-card">
      <header><h2>{t('我的进度', 'Your progress')}</h2><CheckCircle2 size={16} aria-hidden="true" /></header>
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
  </aside>;
}
