import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, ChevronDown, ChevronUp, ListFilter, XCircle } from "lucide-react";
import { useI18n } from "./i18n";
import { layers, type TestResult } from "./types";
import { reportTestLabel } from "./experimentResults";
import { visibleReportTests, formatTestDuration, testKey, type TestOutcome } from "./testReportView";
import {controlIsActive} from './activeControls';
import TestComparison from "./TestComparison";
import FailureDiagnostic from './FailureDiagnostic';
import { learnerTraceFrames, type SourceLocation } from "./sourceLocation";
import type { ReportSourceState } from "./reportSource";
import { ArtifactCharts } from "./deferredPages";
import CommunitySearch from "./CommunitySearch";
import "./report-test-search.css";

export type TestDetailReveal={tests:TestResult[];index:number;active:()=>boolean;complete:(summary:HTMLElement)=>void};
type Props={revealTest?:TestDetailReveal;tests:TestResult[];source:string;sourceState:ReportSourceState;impl?:string;onLocate?:(location:SourceLocation)=>void};
export function ReportTests(props:Props) {
  const [lifetime,setLifetime]=useState({tests:props.tests,revision:0});
  // Reset before children commit, including identical-content A → B → A runs.
  if(lifetime.tests!==props.tests){setLifetime({tests:props.tests,revision:lifetime.revision+1});return null;}
  return <ReportTestList key={lifetime.revision} {...props}/>;
}
function ReportTestList({tests,source,sourceState,impl,onLocate,revealTest}:Props) {
  const { t } = useI18n();
  const [outcome, setOutcome] = useState<TestOutcome>(() => tests.some(test => !test.passed) ? 'failed' : 'all');
  const [layer, setLayer] = useState("all");
  const [query,setQuery]=useState("");
  const [located,setLocated]=useState<string>(),root=useRef<HTMLElement>(null),pending=useRef<TestDetailReveal>(undefined),handled=useRef<TestDetailReveal>(undefined);
  const searchInput=useRef<HTMLInputElement>(null),filters=useRef<HTMLDivElement>(null);
  const href=typeof window==='undefined'?'':window.location.href;
  const here=()=>controlIsActive(root.current)&&!root.current?.closest('details:not([open]),dialog:not([open])')&&location.href===href;
  const [expanded, setExpanded] = useState<Set<string>>(() => {
    const first = tests.findIndex((test) => !test.passed);
    return new Set(first < 0 ? [] : [testKey(tests[first], first)]);
  });
  const passed = tests.filter((test) => test.passed).length;
  const visible = useMemo(()=>visibleReportTests(tests,outcome,layer,query),[tests,outcome,layer,query]);
  const pinned=located?tests.map((test,index)=>({test,key:testKey(test,index)})).find(row=>row.key===located):undefined;
  const outside=!!pinned&&!visible.some(row=>row.key===pinned.key),shown=outside?[pinned!,...visible]:visible;
  useLayoutEffect(()=>{
    if(!revealTest||handled.current===revealTest||revealTest.tests!==tests||!revealTest.active()||!tests[revealTest.index]||tests[revealTest.index].passed)return;
    handled.current=revealTest;pending.current=revealTest;
    const key=testKey(tests[revealTest.index],revealTest.index);setLocated(key);setExpanded(previous=>new Set([...previous,key]));
  },[revealTest,tests]);
  useLayoutEffect(()=>{
    const request=pending.current;if(!request)return;
    if(!request.active()){pending.current=undefined;return;}
    const key=testKey(tests[request.index],request.index),row=Array.from(root.current?.querySelectorAll<HTMLDetailsElement>(':scope > details.test-detail')||[]).find(node=>node.dataset.testKey===key);
    if(located!==key||!row?.open)return;
    const summary=row.querySelector<HTMLElement>(':scope > summary');if(summary){pending.current=undefined;request.complete(summary);}
  },[located,shown,expanded]);
  function search(value:string){
    setLocated(undefined);setQuery(value);
    if(value){const first=visibleReportTests(tests,outcome,layer,value)[0];if(first)setExpanded(previous=>new Set([...previous,first.key]));}
  }
  const allExpanded = shown.length > 0 && shown.every(({ key }) => expanded.has(key));
  function toggle(key: string, open: boolean) {
    setExpanded((previous) => {
      if (previous.has(key) === open) return previous;
      const next = new Set(previous);
      if (open) next.add(key); else next.delete(key);
      return next;
    });
  }
  return <section ref={root} className="report-tests" aria-label={t("测试详情", "Test details")}>
    <div ref={filters} className="report-test-filters" role="group" tabIndex={-1} aria-label={t("测试筛选","Test filters")}>
      <div className="test-outcome-switch" role="group" aria-label={t("按测试结果筛选", "Filter by test outcome")}>
        {([
          ["all", t("全部", "All"), tests.length],
          ["failed", t("未通过", "Failed"), tests.length - passed],
          ["passed", t("已通过", "Passed"), passed],
        ] as const).map(([value, label, count]) => <button type="button" key={value} aria-pressed={outcome === value} onClick={() => {setLocated(undefined);setOutcome(value);}}>{label}<span>{count}</span></button>)}
      </div>
      <label className="test-layer-filter"><ListFilter size={14} aria-hidden="true" /><select aria-label={t("按测试层筛选", "Filter by test layer")} value={layer} onChange={(event) => {setLocated(undefined);setLayer(event.target.value);}}><option value="all">{t("全部测试层", "All test layers")}</option>{layers.filter((value) => tests.some((test) => test.layer === value)).map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
      {(tests.length>8||query)&&<CommunitySearch className="community-search report-test-search" inputRef={searchInput} applied={query} onApply={search} label={t('搜索测试详情','Search test details')} placeholder={t('测试名、参数、报错或性质…','Test name, parameters, error or property…')}/>}
    </div>
    <div className="report-test-actions"><span role="status">{visible.length} / {tests.length} {t("项测试", "tests")}</span><button type="button" className="text-button" disabled={!shown.length} onClick={() => setExpanded((previous) => {
      const next = new Set(previous);
      shown.forEach(({ key }) => { if (allExpanded) next.delete(key); else next.add(key); });
      return next;
    })}>{allExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}{allExpanded ? t("收起当前结果", "Collapse visible") : t("展开当前结果", "Expand visible")}</button></div>
    {located&&<div className="report-test-located" role="status"><span>{outside?t('已定位的失败测试不符合当前筛选，暂时单独显示。你的筛选和搜索输入已保留。','This failure is outside your filters and is shown separately. Your filters and search input are kept.'):t('已定位所选失败测试。','Selected failure located.')}</span><button type="button" className="text-button" onClick={()=>{if(!here())return;setLocated(undefined);filters.current?.focus({preventScroll:true});}}>{t('返回筛选结果','Return to filtered results')}</button></div>}
    {shown.map(({ test, key }) => {const frames=test.passed?[]:learnerTraceFrames(test.message,impl,source,sourceState);return <details key={key} data-test-key={key} data-test-revealed={located===key||undefined} className={`test-detail ${test.passed ? "passed" : "failed"}`} open={expanded.has(key)} onToggle={(event) => toggle(key, event.currentTarget.open)}>
      <summary aria-label={`${test.passed ? t("已通过", "Passed") : t("未通过", "Failed")} · ${reportTestLabel(test)} · ${test.layer} · ${formatTestDuration(test.duration_s || 0)}`}>{test.passed ? <CheckCircle2 size={15} aria-hidden="true" /> : <XCircle size={15} aria-hidden="true" />}<span title={reportTestLabel(test)}>{reportTestLabel(test)}</span><b>{test.layer}</b><small>{formatTestDuration(test.duration_s || 0)}</small></summary>
      <div className="test-body">
        {!test.passed && <FailureDiagnostic message={test.message} location={frames.at(-1)?.location} frames={frames} onLocate={onLocate}/>}
        {test.comparison && <TestComparison comparison={test.comparison} />}
        <p className="test-why"><b>{t("为什么检查这个性质", "Why this property matters")}</b>{test.why || t("见本题测试说明。", "See the problem's test specification.")}</p>
        {test.passed && test.message && <details className="failure-trace"><summary>{t('测试输出','Test output')}</summary><pre>{test.message}</pre></details>}
        {test.artifacts && Object.keys(test.artifacts).length > 0 && <ArtifactCharts data={test.artifacts} />}
      </div>
    </details>;})}
    {!shown.length && <div className="test-filter-empty"><ListFilter size={22} /><p>{query?t('没有找到匹配的测试，试试名称、参数或报错关键词。','No matching tests. Try a name, parameter or error keyword.'):t("当前条件下没有测试项", "No tests match these filters")}</p>{query&&<button type="button" className="text-button" onClick={()=>{searchInput.current?.focus({preventScroll:true});setQuery('');}}>{t('清空搜索','Clear search')}</button>}<button type="button" className="text-button" onClick={() => { filters.current?.focus({preventScroll:true});setOutcome("all"); setLayer("all");setQuery(''); }}>{t("查看全部测试", "Show all tests")}</button></div>}
  </section>;
}
