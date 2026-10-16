import { useEffect, useState } from "react";
import { CheckCircle2, ChevronDown, ChevronUp, ListFilter, XCircle } from "lucide-react";
import { useI18n } from "./i18n";
import { layers, type TestResult } from "./types";
import { reportTestLabel } from "./experimentResults";
import { visibleReportTests, formatTestDuration, testKey, type TestOutcome } from "./testReportView";
import TestComparison from "./TestComparison";
import FailureDiagnostic from './FailureDiagnostic';
import { learnerTraceFrames, type SourceLocation } from "./sourceLocation";
import type { ReportSourceState } from "./reportSource";
import { ArtifactCharts } from "./deferredPages";

export function ReportTests({ tests, source, sourceState, impl, onLocate }: { tests: TestResult[]; source: string; sourceState: ReportSourceState; impl?: string; onLocate?: (location: SourceLocation) => void }) {
  const { t } = useI18n();
  const [outcome, setOutcome] = useState<TestOutcome>(() => tests.some(test => !test.passed) ? 'failed' : 'all');
  const [layer, setLayer] = useState("all");
  const [expanded, setExpanded] = useState<Set<string>>(() => {
    const first = tests.findIndex((test) => !test.passed);
    return new Set(first < 0 ? [] : [testKey(tests[first], first)]);
  });
  const passed = tests.filter((test) => test.passed).length;
  useEffect(() => {
    const first = tests.findIndex(test => !test.passed);
    setOutcome(first >= 0 ? 'failed' : 'all');setLayer('all');
    setExpanded(new Set(first < 0 ? [] : [testKey(tests[first],first)]));
  },[tests]);
  const visible = visibleReportTests(tests, outcome, layer);
  const allExpanded = visible.length > 0 && visible.every(({ key }) => expanded.has(key));
  function toggle(key: string, open: boolean) {
    setExpanded((previous) => {
      if (previous.has(key) === open) return previous;
      const next = new Set(previous);
      if (open) next.add(key); else next.delete(key);
      return next;
    });
  }
  return <section className="report-tests" aria-label={t("测试详情", "Test details")}>
    <div className="report-test-filters">
      <div className="test-outcome-switch" role="group" aria-label={t("按测试结果筛选", "Filter by test outcome")}>
        {([
          ["all", t("全部", "All"), tests.length],
          ["failed", t("未通过", "Failed"), tests.length - passed],
          ["passed", t("已通过", "Passed"), passed],
        ] as const).map(([value, label, count]) => <button key={value} aria-pressed={outcome === value} onClick={() => setOutcome(value)}>{label}<span>{count}</span></button>)}
      </div>
      <label className="test-layer-filter"><ListFilter size={14} aria-hidden="true" /><select aria-label={t("按测试层筛选", "Filter by test layer")} value={layer} onChange={(event) => setLayer(event.target.value)}><option value="all">{t("全部测试层", "All test layers")}</option>{layers.filter((value) => tests.some((test) => test.layer === value)).map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
    </div>
    <div className="report-test-actions"><span role="status">{visible.length} / {tests.length} {t("项测试", "tests")}</span><button className="text-button" disabled={!visible.length} onClick={() => setExpanded((previous) => {
      const next = new Set(previous);
      visible.forEach(({ key }) => { if (allExpanded) next.delete(key); else next.add(key); });
      return next;
    })}>{allExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}{allExpanded ? t("收起当前结果", "Collapse visible") : t("展开当前结果", "Expand visible")}</button></div>
    {visible.map(({ test, key }) => {const frames=test.passed?[]:learnerTraceFrames(test.message,impl,source,sourceState);return <details key={key} className={`test-detail ${test.passed ? "passed" : "failed"}`} open={expanded.has(key)} onToggle={(event) => toggle(key, event.currentTarget.open)}>
      <summary aria-label={`${test.passed ? t("已通过", "Passed") : t("未通过", "Failed")} · ${reportTestLabel(test)} · ${test.layer} · ${formatTestDuration(test.duration_s || 0)}`}>{test.passed ? <CheckCircle2 size={15} aria-hidden="true" /> : <XCircle size={15} aria-hidden="true" />}<span title={reportTestLabel(test)}>{reportTestLabel(test)}</span><b>{test.layer}</b><small>{formatTestDuration(test.duration_s || 0)}</small></summary>
      <div className="test-body">
        {!test.passed && <FailureDiagnostic message={test.message} location={frames.at(-1)?.location} frames={frames} onLocate={onLocate}/>}
        {test.comparison && <TestComparison comparison={test.comparison} />}
        <p className="test-why"><b>{t("为什么检查这个性质", "Why this property matters")}</b>{test.why || t("见本题测试说明。", "See the problem's test specification.")}</p>
        {test.passed && test.message && <details className="failure-trace"><summary>{t('测试输出','Test output')}</summary><pre>{test.message}</pre></details>}
        {test.artifacts && Object.keys(test.artifacts).length > 0 && <ArtifactCharts data={test.artifacts} />}
      </div>
    </details>;})}
    {!visible.length && <div className="test-filter-empty"><ListFilter size={22} /><p>{t("当前条件下没有测试项", "No tests match these filters")}</p><button className="text-button" onClick={() => { setOutcome("all"); setLayer("all"); }}>{t("查看全部测试", "Show all tests")}</button></div>}
  </section>;
}
