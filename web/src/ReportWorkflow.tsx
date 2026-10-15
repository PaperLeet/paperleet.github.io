import { useEffect, useState } from 'react';
import { ArrowRight, ChevronLeft, ChevronRight, Code2, Play, Search, X } from 'lucide-react';
import { useI18n } from './i18n';
import { SourceLocationButton } from './SourceLocationButton';
import { reportFailureTargets, failureCodeContext } from './reportFailureModel';
import type { Report } from './types';
import type { BrowserSyntaxFailure } from './browserSyntax';
import type { ReportSourceState } from './reportSource';
import type { SourceLocation } from './sourceLocation';

export type ReportWorkflowProps = {report?:Report;syntax?:BrowserSyntaxFailure;source:string;sourceState:ReportSourceState;canRun:boolean;running:boolean;onEdit:()=>void;onRun:()=>void;onLocate?:(location:SourceLocation)=>void;onStop?:()=>void;compact?:boolean;resultTitle?:string};
export default function ReportWorkflow({report,syntax,source,sourceState,canRun,running,onEdit,onRun,onLocate,onStop,compact=false,resultTitle}:ReportWorkflowProps) {
  const {t} = useI18n();
  const [index,setIndex] = useState(0);
  useEffect(() => setIndex(0),[report,syntax]);
  const targets = reportFailureTargets(report,syntax,source,sourceState);
  const current = Math.min(index,Math.max(0,targets.length - 1));
  const target = targets[current];
  const changed = sourceState === 'changed' || (!!syntax && syntax.source !== source && sourceState !== 'current');
  const context = compact?[]:failureCodeContext(source,target?.location);
  const stage = running || changed ? 2 : target ? 0 : report ? 2 : 1;
  return <section className="report-workflow" aria-label={t('修正与验证','Fix and verify')}>
    {!compact&&<ol className="feedback-steps">{[[Search,t('查看原因','Inspect')],[Code2,t('修改代码','Edit')],[Play,t('重新验证','Verify')]].map(([Icon,label],step) => {const StepIcon=Icon as typeof Search;return <li key={step} aria-current={stage === step ? 'step' : undefined}><StepIcon size={13} aria-hidden="true"/><span>{label as string}</span>{step < 2 && <ArrowRight size={11} aria-hidden="true"/>}</li>;})}</ol>}
    <div className="feedback-heading"><div>{!compact&&<span className="feedback-eyebrow">{running ? t('正在验证','VERIFICATION IN PROGRESS') : changed ? t('当前修改尚未验证','YOUR CHANGES NEED A NEW RUN') : target ? t('从这里开始修正','START HERE') : t('每次验证，靠近一步','READY FOR THE NEXT STEP')}</span>}<h3>{running ? t('可以继续编辑，结果将对应提交时的代码','Keep editing; this run uses the submitted source') : changed ? t('重新运行，检查这次修改','Run again to check your changes') : target ? target.label || t('判题中断','Run interrupted') : report ? resultTitle||t('回到代码，继续验证你的想法','Return to your code and keep exploring') : t('写好实现后，在这里验证','Implement your solution, then verify it here')}</h3></div>
      {targets.length > 1 && <div className="feedback-error-nav" aria-label={t('切换错误摘要','Browse failure summaries')}><button type="button" className="icon-button" disabled={current === 0} aria-label={t('上一个错误','Previous failure')} onClick={() => setIndex(current - 1)}><ChevronLeft size={15}/></button><span aria-live="polite">{current + 1} / {targets.length}</span><button type="button" className="icon-button" disabled={current === targets.length - 1} aria-label={t('下一个错误','Next failure')} onClick={() => setIndex(current + 1)}><ChevronRight size={15}/></button></div>}
    </div>
    {target?.message && !running && <p className="feedback-diagnostic">{changed && t('上次运行：','Previous run: ')}{target.message}</p>}
    {!changed && !running && !!context.length && <pre className="feedback-code-context" aria-label={t('当前实现的错误上下文','Failure context in the current source')}><code>{context.map(row => <span key={row.line} data-active={row.active}><i aria-hidden="true">{row.line}</i><span>{row.text || ' '}</span></span>)}</code></pre>}
    {target && !target.location && !changed && !running && <p className="feedback-location-note">{t('此错误没有可确认的实现行号，可先查看下方数值对比与完整诊断。','No verified implementation line is available. Review the numerical comparison and full diagnostic below.')}</p>}
    <div className="feedback-actions">{!running && <SourceLocationButton location={target?.location} onLocate={onLocate}/>}<button className="button subtle" type="button" onClick={onEdit}><Code2 size={14} aria-hidden="true"/>{t('返回代码修改','Edit code')}</button>{running && onStop && <button type="button" className="button subtle" onClick={onStop}><X size={13} aria-hidden="true"/>{t('停止当前任务','Stop active run')}</button>}<button className="button feedback-rerun" type="button" disabled={!canRun || running} onClick={onRun} aria-keyshortcuts="Meta+Enter Control+Enter"><Play size={13} aria-hidden="true"/>{running ? t('运行中…','Running…') : changed ? t('验证当前修改','Verify changes') : report || syntax ? t('重新运行全部测试','Run all tests again') : t('运行全部测试','Run all tests')}</button></div>
    {!canRun && !running && <p className="feedback-location-note" role="status">{t('工作区和运行服务就绪后，即可重新验证。','Run again once the workspace and execution service are ready.')}</p>}
  </section>;
}
