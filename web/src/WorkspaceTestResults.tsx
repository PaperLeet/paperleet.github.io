import {useLayoutEffect,useRef,useState} from 'react';
import type {ReactNode} from 'react';
import {CheckCircle2,XCircle} from 'lucide-react';
import {useI18n} from './i18n';
import {deferredFeature} from './deferredFeature';
import {BrowserSyntaxNotice} from './BrowserSyntaxNotice';
import ReportWorkflow,{type ReportWorkflowProps} from './ReportWorkflow';
import {isPassed,type Problem} from './types';
import type {TestDetailReveal} from './ReportTests';
import {controlIsActive} from './activeControls';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import {keepReadingTriggerVisible} from './readingViewport';
import {formatRuntime,runtimeFailure,runtimeMs} from './feedbackVerdict';
import './workspace-test-results.css';
import './lc-feedback.css';

function DetailFailure({retry}:{retry:()=>void}){
  const {t}=useI18n();
  return <div className="workspace-test-detail-error"><p role="alert">{t('测试详情暂时无法加载，代码与运行结果仍保留。','Test details could not load. Your code and run result are preserved.')}</p><button type="button" onClick={event=>{event.currentTarget.closest('details')?.querySelector('summary')?.focus({preventScroll:true});retry();}}>{t('重试详情','Retry details')}</button></div>;
}
const DetailedReport=deferredFeature(()=>import('./ReportView'),['测试详情','test details'],{failure:(_props,retry)=><DetailFailure retry={retry}/>});

export default function WorkspaceTestResults({problem,children,...workflow}:ReportWorkflowProps&{problem:Problem;children?:ReactNode}){
  const {t}=useI18n(),[opened,setOpened]=useState(false),[expanded,setExpanded]=useState(false),[revealTest,setRevealTest]=useState<TestDetailReveal>();
  const root=useRef<HTMLElement>(null),details=useRef<HTMLDetailsElement>(null);
  const {report,syntax,source,sourceState,onLocate}=workflow;
  const latest=useRef({report,syntax,source,running:workflow.running,problemId:problem.id,revision:0});
  if(latest.current.report!==report||latest.current.syntax!==syntax||latest.current.source!==source||latest.current.running!==workflow.running||latest.current.problemId!==problem.id)latest.current={report,syntax,source,running:workflow.running,problemId:problem.id,revision:latest.current.revision+1};
  const intent=useWorkspaceIntent(`${problem.id}:${latest.current.revision}`),href=typeof window==='undefined'?'':window.location.href;
  const pending=useRef<{request:TestDetailReveal;claim:()=>boolean}>(undefined);
  useLayoutEffect(()=>{pending.current=undefined;setRevealTest(undefined);},[latest.current.revision]);
  const current=()=>!latest.current.running&&document.visibilityState!=='hidden'&&controlIsActive(root.current)&&!root.current?.closest('details:not([open]),dialog:not([open])')&&location.href===href&&intent.isCurrent();
  function reveal(index:number,trigger:HTMLButtonElement){
    if(!current()||!root.current?.contains(trigger)||!controlIsActive(trigger)||!report||report.problem!==problem.id||!Number.isSafeInteger(index)||index<0||index>=report.tests.length||report.tests[index].passed)return;
    const revision=latest.current.revision,selected=report,claim=intent.create();
    const request:TestDetailReveal={tests:report.tests,index,active:()=>claim.active()&&current()&&latest.current.report===selected&&latest.current.revision===revision&&details.current?.open===true&&pending.current?.request===request,complete:summary=>{
      if(!request.active()||!controlIsActive(summary)||!root.current?.contains(summary))return;
      const allowed=claim();pending.current=undefined;setRevealTest(undefined);if(!allowed)return;summary.focus({preventScroll:true});keepReadingTriggerVisible(summary,100);
    }};
    pending.current={request,claim};setRevealTest(request);setOpened(true);setExpanded(true);
  }
  const syntaxCurrent = !!syntax && syntax.source === source;
  const passed = !syntaxCurrent && !!report && isPassed(problem,report);
  const selectedPassed = !syntaxCurrent && !!report && !report.error && report.tests.length > 0 && report.tests.every(test => test.passed);
  const runtime=!syntaxCurrent&&report?runtimeMs(report):undefined,browserRun=report?.runtime==='browser-numpy-wasm';
  const verdict=report&&isPassed(problem,report)
    ?sourceState==='current'?t('本次测试已通过','Tests passed'):t('上次测试已通过','Previous tests passed')
    :t('本次运行已结束','Run finished');
  return <section ref={root} className="workspace-test-results" aria-label={t('测试反馈','Test feedback')}>
    {(report || syntax) && <div className={`result-head ${selectedPassed ? 'success' : 'failure'}`} role="status">
      <div>{selectedPassed ? <CheckCircle2 size={22} aria-hidden="true"/> : <XCircle size={22} aria-hidden="true"/>}<h3>{syntaxCurrent
        ? t('语法错误','Syntax error') : report?.error ? t('运行未完成','Run incomplete')
        : passed ? browserRun ? t('测试通过','Tests passed') : 'Accepted'
        : selectedPassed ? t('所选测试通过','Selected tests passed') : report && runtimeFailure(report) ? t('执行出错','Runtime Error') : t('解答错误','Wrong Answer')}</h3></div>
      <span>{!syntaxCurrent && report ? `${report.tests.filter(test => test.passed).length} / ${report.tests.length} ${t('项通过','tests passed')}` : ''}
        {!syntaxCurrent && report && sourceState !== 'current' && <small>{t('上次运行结果','Previous run result')}</small>}</span>
      {!syntaxCurrent && report && (runtime !== undefined || browserRun) && <dl className="result-head-metrics">
        {runtime !== undefined && <div><dt>{t('运行时间','Runtime')}</dt><dd>{formatRuntime(runtime)}</dd></div>}
        {browserRun && <div><dt>{t('环境','Environment')}</dt><dd>{t('浏览器 · WASM','Browser · WASM')}</dd></div>}
      </dl>}
    </div>}
    <ReportWorkflow {...workflow} onTestDetails={reveal} compact resultTitle={passed ? t('可以继续修改代码或查看测试详情','Edit your code or review test details') : verdict}/>
    {children}
    {(report||syntax)&&<details ref={details} className="workspace-test-details" open={expanded} onToggle={event=>{setExpanded(event.currentTarget.open);if(event.currentTarget.open)setOpened(true);else{pending.current=undefined;intent.cancel();}}}>
      <summary>{t('测试详情与日志','Test details and logs')}</summary>
      {opened&&<div className="workspace-test-detail-body">
        {syntax&&<BrowserSyntaxNotice failure={syntax} source={source} onLocate={onLocate}/>}
        {report&&<DetailedReport revealTest={revealTest} report={report} problem={problem} sourceState={sourceState} source={source} onLocate={onLocate}/>}
      </div>}
    </details>}
  </section>;
}
