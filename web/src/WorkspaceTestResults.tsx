import {useState} from 'react';
import type {ReactNode} from 'react';
import {CheckCircle2,ChevronDown,FileText,XCircle} from 'lucide-react';
import {useI18n} from './i18n';
import {deferredFeature} from './deferredFeature';
import {BrowserSyntaxNotice} from './BrowserSyntaxNotice';
import ReportWorkflow,{type ReportWorkflowProps} from './ReportWorkflow';
import {isPassed,type Problem} from './types';
import './workspace-test-results.css';

function DetailFailure({retry}:{retry:()=>void}){
  const {t}=useI18n();
  return <div className="workspace-test-detail-error"><p role="alert">{t('测试详情暂时无法加载，代码与运行结果仍保留。','Test details could not load. Your code and run result are preserved.')}</p><button type="button" onClick={event=>{event.currentTarget.closest('details')?.querySelector('summary')?.focus({preventScroll:true});retry();}}>{t('重试详情','Retry details')}</button></div>;
}
const DetailedReport=deferredFeature(()=>import('./ReportView'),['测试详情','test details'],{failure:(_props,retry)=><DetailFailure retry={retry}/>});

export default function WorkspaceTestResults({problem,children,...workflow}:ReportWorkflowProps&{problem:Problem;children?:ReactNode}){
  const {t}=useI18n(),[opened,setOpened]=useState(false);
  const {report,syntax,source,sourceState,onLocate}=workflow;
  const syntaxCurrent = !!syntax && syntax.source === source;
  const passed = !syntaxCurrent && !!report && isPassed(problem,report);
  const selectedPassed = !syntaxCurrent && !!report && !report.error && report.tests.length > 0 && report.tests.every(test => test.passed);
  const verdict=report&&isPassed(problem,report)
    ?sourceState==='current'?t('本次测试已通过','Tests passed'):t('上次测试已通过','Previous tests passed')
    :t('本次运行已结束','Run finished');
  return <section className="workspace-test-results" aria-label={t('测试反馈','Test feedback')}>
    {(report || syntax) && <div className={`result-head ${selectedPassed ? 'success' : 'failure'}`} role="status">
      <div>{selectedPassed ? <CheckCircle2 size={22} aria-hidden="true"/> : <XCircle size={22} aria-hidden="true"/>}<h3>{syntaxCurrent
        ? t('语法错误','Syntax error') : report?.error ? t('运行未完成','Run incomplete')
        : passed ? report?.runtime === 'browser-numpy-wasm' ? t('测试通过','Tests passed') : 'Accepted'
        : selectedPassed ? t('所选测试通过','Selected tests passed') : t('测试未通过','Tests failed')}</h3></div>
      <span>{!syntaxCurrent && report ? `${report.tests.filter(test => test.passed).length} / ${report.tests.length} ${t('项通过','tests passed')}` : ''}
        {!syntaxCurrent && report && sourceState !== 'current' && <small>{t('上次运行结果','Previous run result')}</small>}</span>
    </div>}
    <ReportWorkflow {...workflow} compact resultTitle={passed ? t('可以继续修改代码或查看测试详情','Edit your code or review test details') : verdict}/>
    {(report||syntax)&&<details className="workspace-test-details" onToggle={event=>{if(event.currentTarget.open)setOpened(true);}}>
      <summary><FileText size={14} aria-hidden="true"/>{t('测试详情与日志','Test details and logs')}<ChevronDown size={14} aria-hidden="true"/></summary>
      {opened&&<div className="workspace-test-detail-body">
        {syntax&&<BrowserSyntaxNotice failure={syntax} source={source} onLocate={onLocate}/>}
        {report&&<DetailedReport report={report} problem={problem} sourceState={sourceState} source={source} onLocate={onLocate}/>}
      </div>}
    </details>}
    {children}
  </section>;
}
