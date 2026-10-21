import { useEffect, useId, useState, type KeyboardEvent } from 'react';
import { ArrowRight, ChevronLeft, ChevronRight, Code2, Play, Search, X } from 'lucide-react';
import { useI18n } from './i18n';
import { SourceLocationButton } from './SourceLocationButton';
import TestComparison from './TestComparison';
import { reportFailureTargets, failureCodeContext } from './reportFailureModel';
import type { Report } from './types';
import type { BrowserSyntaxFailure } from './browserSyntax';
import type { ReportSourceState } from './reportSource';
import type { SourceLocation } from './sourceLocation';

export type ReportWorkflowProps = {report?:Report;syntax?:BrowserSyntaxFailure;source:string;sourceState:ReportSourceState;canRun:boolean;running:boolean;onEdit:()=>void;onRun:()=>void;onLocate?:(location:SourceLocation)=>void;onStop?:()=>void;compact?:boolean;resultTitle?:string;onTestDetails?:(index:number,trigger:HTMLButtonElement)=>void};
export default function ReportWorkflow({report,syntax,source,sourceState,canRun,running,onEdit,onRun,onLocate,onStop,compact=false,resultTitle,onTestDetails}:ReportWorkflowProps) {
  const {t} = useI18n();
  const [index,setIndex] = useState(0);
  const panel = useId();
  useEffect(() => setIndex(0),[report,syntax]);
  const targets = reportFailureTargets(report,syntax,source,sourceState);
  const current = Math.min(index,Math.max(0,targets.length - 1));
  const target = targets[current];
  const changed = sourceState === 'changed' || (!!syntax && syntax.source !== source && sourceState !== 'current');
  const context = compact?[]:failureCodeContext(source,target?.location);
  const stage = running || changed ? 2 : target ? 0 : report ? 2 : 1;
  // A few failures read best as LeetCode-style case tabs; long lists keep the native picker.
  const tabs = targets.length > 1 && targets.length <= 8;
  function moveCase(event:KeyboardEvent<HTMLDivElement>) {
    const step = ({ArrowRight:1,ArrowLeft:-1,Home:-targets.length,End:targets.length} as Record<string,number>)[event.key];
    if (step === undefined || running) return;
    event.preventDefault();
    const next = Math.max(0,Math.min(targets.length - 1,current + step));
    setIndex(next);
    event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
  }
  return <section className="report-workflow" aria-label={t('修正与验证','Fix and verify')}>
    {!compact&&<ol className="feedback-steps">{[[Search,t('查看原因','Inspect')],[Code2,t('修改代码','Edit')],[Play,t('重新验证','Verify')]].map(([Icon,label],step) => {const StepIcon=Icon as typeof Search;return <li key={step} aria-current={stage === step ? 'step' : undefined}><StepIcon size={13} aria-hidden="true"/><span>{label as string}</span>{step < 2 && <ArrowRight size={11} aria-hidden="true"/>}</li>;})}</ol>}
    {tabs && <div className="feedback-case-tabs" role="tablist" aria-label={t('切换错误摘要','Browse failure summaries')} onKeyDown={moveCase}>
      {targets.map((item,position) => {const cases=targets[0].id === 'run-error' ? position : position + 1;return <button key={item.id} id={`${panel}-tab-${position}`} type="button" role="tab" aria-controls={panel} data-failure={item.id} aria-selected={position === current} tabIndex={position === current ? 0 : -1} title={item.label || t('判题中断','Run interrupted')} disabled={running} onClick={() => setIndex(position)}>{item.id === 'run-error' ? t('判题中断','Run interrupted') : t(`用例 ${cases}`,`Case ${cases}`)}</button>;})}
    </div>}
    <div id={panel} role={tabs ? 'tabpanel' : undefined} aria-labelledby={tabs ? `${panel}-tab-${current}` : undefined}>
    <div className="feedback-heading"><div>{!compact&&<span className="feedback-eyebrow">{running ? t('正在验证','VERIFICATION IN PROGRESS') : changed ? t('当前修改尚未验证','YOUR CHANGES NEED A NEW RUN') : target ? t('从这里开始修正','START HERE') : t('每次验证，靠近一步','READY FOR THE NEXT STEP')}</span>}<h3>{running ? t('可以继续编辑，结果将对应提交时的代码','Keep editing; this run uses the submitted source') : changed ? t('重新运行，检查这次修改','Run again to check your changes') : target ? target.label || t('判题中断','Run interrupted') : report ? resultTitle||t('回到代码，继续验证你的想法','Return to your code and keep exploring') : t('写好实现后，在这里验证','Implement your solution, then verify it here')}</h3></div>
      {targets.length > 8 && <div className="feedback-error-nav" aria-label={t('切换错误摘要','Browse failure summaries')}>
        <select className="feedback-failure-picker" aria-label={t('选择失败项','Choose a failure')} value={target.id} disabled={running} onChange={event=>setIndex(Math.max(0,targets.findIndex(item=>item.id===event.target.value)))}>
          {targets.map((item,position)=><option key={item.id} value={item.id}>{position+1}. {item.label || t('判题中断','Run interrupted')}</option>)}
        </select>
        <button type="button" className="icon-button" disabled={running || current === 0} aria-label={t('上一个错误','Previous failure')} onClick={() => setIndex(current - 1)}><ChevronLeft size={15}/></button><span aria-live="polite">{current + 1} / {targets.length}</span><button type="button" className="icon-button" disabled={running || current === targets.length - 1} aria-label={t('下一个错误','Next failure')} onClick={() => setIndex(current + 1)}><ChevronRight size={15}/></button></div>}
    </div>
    {target?.message && !running && <p className="feedback-diagnostic">{changed && t('上次运行：','Previous run: ')}{target.message}</p>}
    {!running&&onTestDetails&&target?.testIndex!==undefined&&<button type="button" className="text-button feedback-test-details" onClick={event=>onTestDetails(target.testIndex!,event.currentTarget)}><Search size={14} aria-hidden="true"/>{t('查看这项失败详情','View failure details')}</button>}
    {target?.comparison && !running && <TestComparison comparison={target.comparison}/>}
    {!changed && !running && !!context.length && <pre className="feedback-code-context" aria-label={t('当前实现的错误上下文','Failure context in the current source')}><code>{context.map(row => <span key={row.line} data-active={row.active}><i aria-hidden="true">{row.line}</i><span>{row.text || ' '}</span></span>)}</code></pre>}
    {target && !target.location && !changed && !running && <p className="feedback-location-note">{t('此错误没有可确认的实现行号，可先查看下方数值对比与完整诊断。','No verified implementation line is available. Review the numerical comparison and full diagnostic below.')}</p>}
    </div>
    <div className="feedback-actions">{!running && <SourceLocationButton location={target?.location} onLocate={onLocate}/>}<button className="button subtle" type="button" onClick={onEdit}><Code2 size={14} aria-hidden="true"/>{t('返回代码修改','Edit code')}</button>{running && onStop && <button type="button" className="button subtle" onClick={onStop}><X size={13} aria-hidden="true"/>{t('停止当前任务','Stop active run')}</button>}<button className="button feedback-rerun" type="button" disabled={!canRun || running} onClick={onRun} aria-keyshortcuts="Meta+Enter Control+Enter"><Play size={13} aria-hidden="true"/>{running ? t('运行中…','Running…') : changed ? t('验证当前修改','Verify changes') : report || syntax ? t('重新运行全部测试','Run all tests again') : t('运行全部测试','Run all tests')}</button></div>
    {!canRun && !running && <p className="feedback-location-note" role="status">{t('工作区和运行服务就绪后，即可重新验证。','Run again once the workspace and execution service are ready.')}</p>}
  </section>;
}
