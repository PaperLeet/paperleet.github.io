import type { ReactNode } from "react";
import { CheckCircle2, FlaskConical, XCircle } from "lucide-react";
import { useI18n } from "./i18n";
import { isPassed, layers, type Problem, type Report } from "./types";
import { ReportSourceNotice } from "./ReportSourceNotice";
import type { ReportSourceState } from "./reportSource";
import { learnerTraceFrames, type SourceLocation } from "./sourceLocation";
import FailureDiagnostic from './FailureDiagnostic';
import { ReportTests,type TestDetailReveal } from "./ReportTests";
import "./lc-feedback-report.css";

export default function ReportView({
  report,
  problem,
  sourceState,
  source,
  onLocate,
  children,
  accepted,
  revealTest,
}: {
  report?: Report;
  problem: Problem;
  sourceState: ReportSourceState;
  source: string;
  onLocate?: (location: SourceLocation) => void;
  children?: ReactNode;
  /** Immutable account history uses the verdict recorded when this attempt ran. */
  accepted?: boolean;
  revealTest?:TestDetailReveal;
}) {
  const { t, locale } = useI18n();
  const labels = t(
    "接口契约|参考对齐|数学性质|真实行为",
    "Contract|Reference|Properties|Behavior",
  ).split("|");
  if (!report)
    return (
      <div className="empty-state">
        <div className="empty-icon">
          <FlaskConical size={28} />
        </div>
        <h3>{t("用测试验证你的实现", "Validate your implementation")}</h3>
        <p>
          {t(
            "在代码编辑器中完成实现，再点击「运行全部测试」。",
            "Complete your implementation in the code editor, then select Run all tests.",
          )}
          <br />
          {t(
            "测试结果和失败原因会直接显示在这里。",
            "Results and failure details will appear here.",
          )}
        </p>
      </div>
    );
  const passed = report.tests.filter((test) => test.passed).length,
    total = report.tests.length,
    allPassed = !report.error && total > 0 && passed === total;
  const frames=learnerTraceFrames(report.error,report.impl,source,sourceState);
  return (
    <div className="report-view" data-source={sourceState}>
      <div className={`result-head ${allPassed ? sourceState === 'current' ? 'success' : 'previous' : "failure"}`}>
        <div>
          {allPassed ? <CheckCircle2 size={24} /> : <XCircle size={24} />}
          <h3>
            {report.error
              ? t("判题未完成", "Run incomplete")
              : (accepted ?? isPassed(problem, report))
                ? sourceState !== "current" ? t("上次测试通过", "Previous tests passed") : report.runtime === "browser-numpy-wasm" ? t("测试通过", "Tests passed") : "Accepted"
                : allPassed
                  ? t("所选测试通过", "Selected tests passed")
                  : t(`${total - passed} 项测试需要修正`,`${total - passed} tests need attention`)}
          </h3>
        </div>
        <span>
          {passed} / {total} {t("项通过", "tests passed")}
        </span>
      </div>
      <div className="result-summary-meter" aria-hidden="true"><span style={{ width: `${total ? passed / total * 100 : 0}%` }} /></div>
      <ReportSourceNotice state={sourceState} />
      {report.error && <FailureDiagnostic message={report.error} location={frames.at(-1)?.location} frames={frames} onLocate={onLocate}/>}
      {children}
      <div className="report-meta">
        <span>{t("最近一次运行", "Latest run")}</span>
        <span>
          {report.created_at
            ? new Date(report.created_at).toLocaleString(
                locale === "zh" ? "zh-CN" : "en-US",
              )
            : ""}
        </span>
      </div>
      <div className="test-layers">
        {layers.map((l, i) => {
          const [a, b] = report.summary[l] || [0, 0];
          return (
            <div key={l}>
              <div>
                <b>{l}</b>
                <span>
                  {a}/{b}
                </span>
              </div>
              <small>{labels[i]}</small>
              <div className="mini-progress">
                <i style={{ width: `${b ? (a / b) * 100 : 0}%` }} />
              </div>
            </div>
          );
        })}
      </div>
      <ReportTests revealTest={revealTest} key={`${report.problem}:${report.created_at || report.source_sha256 || "latest"}`} tests={report.tests} source={source} sourceState={sourceState} impl={report.impl} onLocate={onLocate} />
      {report.mutation && (
        <section className="mutation">
          <h4>{t("错误变体检查", "Mutation checks")}</h4>
          {Object.entries(report.mutation).map(([id, caught]) => (
            <div key={id}>
              <b>
                {caught.length ? "✓" : "×"} {id}
              </b>
              <span>
                {caught.length
                  ? caught.join(", ")
                  : t("当前测试未捕获", "Not caught by current tests")}
              </span>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
