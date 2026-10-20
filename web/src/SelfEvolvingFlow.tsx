import {useEffect, useId, useState} from 'react';
import {ArrowLeft, ArrowRight, ExternalLink, LockKeyhole, Pause, Play, RotateCcw, Sparkles} from 'lucide-react';
import {useI18n} from './i18n';
import {consensusExample, evolutionFlows, hintExample, renderExample} from './selfEvolving';
import './self-evolving.css';

function Range({label, value, min = 0, max = 1, step = .05, change}: {label: string; value: number; min?: number; max?: number; step?: number; change: (n: number) => void}) {
  return <label className="evolution-range"><span>{label}<output>{Number(value.toFixed(2))}</output></span><input type="range" min={min} max={max} step={step} value={value} onChange={e => change(Number(e.target.value))}/></label>;
}

function Feedback({paper}: {paper: string}) {
  const {t} = useI18n();
  const [p, setP] = useState(.6), [duplicates, setDuplicates] = useState(2), [valid, setValid] = useState(true);
  const [delta, setDelta] = useState(.1), [length, setLength] = useState(200), [successes, setSuccesses] = useState(3), [easy, setEasy] = useState(.8);
  const consensus = consensusExample(p, duplicates, valid), hint = hintExample(delta, length, valid), render = renderExample(successes, easy, p);
  const keep = paper === 'g-zero' ? hint.keep : consensus.keep;
  return <div className="evolution-feedback">
    <div className="evolution-example-heading"><strong>{t('试着改变反馈', 'Try changing the feedback')}</strong><span>{t('教学示例', 'Teaching example')}</span></div>
    {paper === 'g-zero' ? <>
      <Range label="Hint-δ" value={delta} min={-.5} max={1} change={setDelta}/>
      <Range label={t('提示字符数', 'Hint characters')} value={length} max={500} step={10} change={setLength}/>
      <label className="evolution-checkbox"><input type="checkbox" checked={valid} onChange={e => setValid(e.target.checked)}/>{t('格式与结构检查通过', 'Format and structure checks pass')}</label>
      <div className="evolution-equation"><span>{t('出题奖励', 'Proposer reward')}</span><code data-score="reward">{hint.reward.toFixed(3)}</code><small>{valid ? `${delta.toFixed(2)} − ${hint.lengthPenalty.toFixed(3)} − 0.1` : '−1 − 0.1'}</small></div>
      <p>{t('固定另三条 δ 为 −0.2、0.1、0.6，全池中位数为', 'With three other δ values fixed at −0.2, 0.1, 0.6, the full-pool median is')} <b data-score="median">{hint.median.toFixed(3)}</b>。</p>
      <p>{t('出题奖励偏向较大的 δ；DPO 训练池默认保留下半部分，并包含中位数处的所有并列项。这里重复惩罚固定为 0.1。', 'The proposer reward favors larger δ; the default DPO pool keeps the lower half, including all median ties. The repetition penalty here is fixed at 0.1.')}</p>
    </> : paper === 'mm-zero' ? <>
      <Range label={t('成功渲染 / 4 次尝试', 'Successful renders / 4 attempts')} value={successes} max={4} step={1} change={setSuccesses}/>
      <Range label={t('简单题正确比例', 'Easy-answer match rate')} value={easy} change={setEasy}/>
      <Range label={t('困难题一致率 c', 'Hard-answer consistency c')} value={p} min={.1} change={setP}/>
      <div className="evolution-renders" aria-label={t('渲染结果', 'Render outcomes')}>{[0,1,2,3].map(i => <span key={i} data-success={i < successes}>{i < successes ? '✓' : '×'} {i + 1}</span>)}</div>
      <div className="evolution-equation"><span>{t('Proposer 平均奖励', 'Mean Proposer reward')}</span><code data-score="reward">{render.proposer.toFixed(3)}</code><small>{successes}/4 × (min({easy.toFixed(2)}, 0.5) + {render.difficulty.toFixed(2)})</small></div>
      <p>{t('Coder 单次奖励：', 'Per-attempt Coder reward: ')}<b data-score="coder">{render.coder.toFixed(2)}</b> {t('（成功），−0.10（失败）。', '(success), −0.10 (failure).')}</p>
      <p>{t('此例固定格式有效、多样性与其他调整为零、无语法错误。失败仍占分母；Coder 的简单题得分不截断到 0.5。', 'This example fixes valid format, zero diversity/other adjustments and no syntax errors. Failed renders remain in the denominator; the Coder easy score is not capped at 0.5.')}</p>
    </> : <>
      <Range label={t('多数答案一致率 p', 'Majority-answer consistency p')} value={p} min={.1} change={setP}/>
      <Range label={paper === 'visplay' ? t('同图 10 道问题中的重复数', 'Duplicates among 10 same-image questions') : t('10 道问题中的重复数', 'Duplicates among 10 questions')} value={duplicates} min={1} max={10} step={1} change={setDuplicates}/>
      <label className="evolution-checkbox"><input type="checkbox" checked={valid} onChange={e => setValid(e.target.checked)}/>{t('问题格式有效', 'Question format is valid')}</label>
      <svg className="evolution-curve" viewBox="0 0 300 90" role="img" aria-label={t('不确定性在一致率 0.5 时达到最大；阴影为课程筛选区间。', 'Uncertainty peaks at consistency 0.5; the shaded region is the curriculum interval.')}>
        <rect x="80" y="8" width="140" height="60" fill="var(--accent)" opacity=".12"/><path d="M10 68 L150 8 L290 68" fill="none" stroke="var(--accent)" strokeWidth="2"/>
        <circle cx={10 + p * 280} cy={68 - consensus.uncertainty * 60} r="5" fill="var(--accent)"/>
        <text x="10" y="86">0</text><text x="150" y="86" textAnchor="middle">0.5</text><text x="290" y="86" textAnchor="end">1</text>
      </svg>
      <div className="evolution-equation"><span>{t('出题奖励', 'Question reward')}</span><code data-score="reward">{consensus.reward.toFixed(3)}</code><small>{valid ? `max(0, ${consensus.uncertainty.toFixed(2)} − ${duplicates}/10)` : t('格式无效 → 0', 'Invalid format → 0')}</small></div>
      <p>{paper === 'visplay' ? t('重复只在同图问题中计数。改变其他图的问题数量，不应改变这里的奖励。', 'Count repetition within the same image. Questions for other images must not change this reward.') : t('一致率描述答案是否相同，不保证答案正确。奖励与课程筛选分别计算：奖励为零，也可能落在筛选区间。', 'Consistency measures agreement, not correctness. Reward and curriculum selection are separate: a zero reward can still fall inside the selection interval.')}</p>
    </>}
    {paper !== 'mm-zero' && <div className="evolution-selection" data-kept={keep} role="status">{keep ? '✓ ' : '× '}{paper === 'g-zero' ? keep ? t('进入 DPO 偏好池', 'Keep in DPO preference pool') : t('不进入 DPO 偏好池', 'Exclude from DPO preference pool') : keep ? t('保留：0.25 ≤ p ≤ 0.75', 'Keep: 0.25 ≤ p ≤ 0.75') : t('丢弃：格式无效或 p 超出区间', 'Drop: invalid format or p outside interval')}</div>}
  </div>;
}

export default function SelfEvolvingFlow({paper}: {paper: string}) {
  const flow = evolutionFlows[paper];
  const {t} = useI18n();
  const [step, setStep] = useState(0), [playing, setPlaying] = useState(false), [reduced, setReduced] = useState(false);
  const arrow = useId().replace(/:/g, ''), title = useId();
  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const sync = () => {setReduced(media.matches); if (media.matches) setPlaying(false);};
    sync(); media.addEventListener('change', sync);
    return () => media.removeEventListener('change', sync);
  }, []);
  useEffect(() => {
    if (!playing || !flow) return;
    if (step >= flow.steps.length - 1) {setPlaying(false); return;}
    const timer = window.setTimeout(() => setStep(n => n + 1), 3500);
    return () => window.clearTimeout(timer);
  }, [playing, step, flow]);
  useEffect(() => {
    const pause = () => {if (document.hidden) setPlaying(false);};
    document.addEventListener('visibilitychange', pause);
    return () => document.removeEventListener('visibilitychange', pause);
  }, []);
  useEffect(() => {setStep(0); setPlaying(false);}, [paper]);
  if (!flow) return null;
  const current = flow.steps[step] ?? flow.steps[0];
  const select = (n: number) => {setPlaying(false); setStep(n);};
  return <section className="evolution" aria-labelledby={title} data-paper={paper} data-playing={playing && !reduced}>
    <header className="evolution-heading"><div><span>{flow.name} · {t('动态讲解', 'Interactive walkthrough')}</span><h3 id={title}>{t('一轮自演化，如何发生？', 'How does one round of self-evolution work?')}</h3></div><a href={flow.source} target="_blank" rel="noreferrer">{t('论文原图', 'Paper figure')}<ExternalLink size={14}/></a></header>
    <div className="evolution-roles">{flow.roles.map((role, i) => <span key={role} data-updating={current.update === i}>{current.update === i ? <Sparkles size={14}/> : <LockKeyhole size={14}/>}<b>{role}</b>{current.update === i ? t('正在更新', 'Updating') : t('参数冻结', 'Frozen')}</span>)}</div>
    <div className="evolution-body"><div className="evolution-story">
      <div className="evolution-map">
        <svg viewBox="0 0 400 240" preserveAspectRatio="none" aria-hidden="true"><defs><marker id={arrow} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--accent)"/></marker></defs>{['M176 49H224','M313 100V140','M224 191H176','M87 140V100'].map((d,i) => <path key={d} d={d} markerEnd={`url(#${arrow})`} className={current.node === (i+1)%4 ? 'is-active' : ''}/>)}</svg>
        {flow.nodes.map((node,i) => <button key={i} className={`evolution-node node-${i}`} data-active={current.node === i} onClick={() => select(flow.steps.findIndex(s => s.node === i))}><span>0{i + 1}</span><strong>{t(...node)}</strong></button>)}
      </div>
      <ol className="evolution-steps">{flow.steps.map((s,i) => <li key={i}><button aria-current={i === step ? 'step' : undefined} onClick={() => select(i)}><span>{i + 1}</span>{t(...s.title)}</button></li>)}</ol>
      <div className="evolution-detail" aria-live={playing ? 'off' : 'polite'}><span>{step + 1} / {flow.steps.length}</span><h4>{t(...current.title)}</h4><p>{t(...current.detail)}</p></div>
      <div className="evolution-controls">
        <button onClick={() => select(Math.max(0,step-1))} disabled={step === 0} aria-label={t('上一步', 'Previous step')}><ArrowLeft size={16}/></button>
        <button className="evolution-play" aria-pressed={playing} onClick={() => {if (step === flow.steps.length-1) setStep(0); setPlaying(p => !p);}}>{playing ? <Pause size={16}/> : <Play size={16}/>} {playing ? t('暂停', 'Pause') : step === flow.steps.length-1 ? t('重播这一轮', 'Replay round') : t('播放这一轮', 'Play round')}</button>
        <button onClick={() => select(Math.min(flow.steps.length-1,step+1))} disabled={step === flow.steps.length-1} aria-label={t('下一步', 'Next step')}><ArrowRight size={16}/></button>
        <button onClick={() => select(0)} aria-label={t('重置流程', 'Reset walkthrough')}><RotateCcw size={16}/></button>
      </div>
      <p className="evolution-caption">{t('按论文方法重绘的教学流程。箭头表示数据与反馈的循环；步骤说明交替更新的次序。', 'An original teaching diagram based on the paper. Arrows show the data/feedback loop; steps explain the alternating update order.')}</p>
    </div><Feedback key={paper} paper={paper}/></div>
  </section>;
}
