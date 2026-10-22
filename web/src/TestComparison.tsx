import { useI18n } from "./i18n";
import type { TestComparison as Comparison } from "./comparisons";
import "./lc-feedback-report.css";

export default function TestComparison({ comparison }: { comparison: Comparison }) {
  const { t } = useI18n();
  const shape = (sizes: number[]) => sizes.length ? `[${sizes.join(", ")}]` : t("标量", "Scalar");
  const finiteValues = comparison.kind === "value" && typeof comparison.expected !== "string" && typeof comparison.actual !== "string";
  const nonfinite = comparison.kind === "value" && [comparison.expected, comparison.actual].some(value => typeof value === "string" && ["NaN", "Infinity", "-Infinity"].includes(value));
  return <section className="test-comparison" aria-label={t("数值对比", "Numerical comparison")}>
    <h4>{comparison.kind === "shape" ? t("张量形状不匹配", "Tensor shapes differ") : t("不匹配元素位置（从 0 开始）", "Mismatched element (zero-based index)")}</h4>
    {comparison.kind === "value" && <code className="comparison-index">{comparison.index.length ? `[${comparison.index.join(", ")}]` : t("标量", "Scalar")}</code>}
    <dl className="comparison-values">
      <div><dt>{t("期望值", "Expected")}</dt><dd>{comparison.kind === "shape" ? shape(comparison.expected_shape) : String(comparison.expected)}</dd></div>
      <div><dt>{t("实际值", "Actual")}</dt><dd>{comparison.kind === "shape" ? shape(comparison.actual_shape) : String(comparison.actual)}</dd></div>
    </dl>
    {comparison.kind === "value" && <>
      <dl className="comparison-tolerances">
        <div><dt>{t("此位置绝对误差", "Absolute error here")}</dt><dd>{String(comparison.absolute_error)}</dd></div>
        {finiteValues && <div><dt>{t("允许误差", "Allowed error")}</dt><dd>{String(comparison.atol + comparison.rtol * Math.abs(Number(comparison.expected)))}</dd></div>}
      </dl>
      <p>{finiteValues ? t("允许误差 = atol + rtol × |期望值|。", "Allowed error = atol + rtol × |expected|.") : nonfinite ? t("非有限值需单独检查；NaN 不会通过近似相等检查。", "Check nonfinite values separately; NaN does not pass approximate equality.") : t("较大整数以文本保留精度，容差以测试设定为准。", "Large integers retain exact precision as text; use the test's tolerances.")} <code>atol={comparison.atol}, rtol={comparison.rtol}</code></p>
    </>}
  </section>;
}
