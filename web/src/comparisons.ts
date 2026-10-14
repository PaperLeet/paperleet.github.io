export type ComparisonScalar = number | boolean | string;
type Shapes = { expected_shape: number[]; actual_shape: number[] };
export type TestComparison = Shapes & (
  | { kind: "shape" }
  | { kind: "value"; index: number[]; expected: ComparisonScalar; actual: ComparisonScalar; absolute_error: number | "Infinity"; rtol: number; atol: number }
);
const shape = (value: unknown): value is number[] => Array.isArray(value) && value.length <= 32 && value.every(n => Number.isSafeInteger(n) && n >= 0);
const scalar = (value: unknown): value is ComparisonScalar => typeof value === "boolean" || typeof value === "number" && Number.isFinite(value) ||
  typeof value === "string" && value.length <= 32 && (/^-?\d+$/.test(value) || ["NaN", "Infinity", "-Infinity"].includes(value));

/** Optional feedback is discarded when malformed; the test outcome stays intact. */
export function parseComparison(value: unknown): TestComparison | undefined {
  if (!value || typeof value !== "object") return;
  const c = value as Record<string, unknown>;
  if (!shape(c.expected_shape) || !shape(c.actual_shape)) return;
  const sameShape = c.expected_shape.length === c.actual_shape.length && c.expected_shape.every((n, i) => n === (c.actual_shape as number[])[i]);
  const shapes = { expected_shape: [...c.expected_shape], actual_shape: [...c.actual_shape] };
  if (c.kind === "shape") return sameShape ? undefined : { kind: "shape", ...shapes };
  if (c.kind !== "value" || !sameShape || !Array.isArray(c.index) || c.index.length !== c.actual_shape.length) return;
  if (!c.index.every((position, i) => Number.isSafeInteger(position) && position >= 0 && position < (c.actual_shape as number[])[i])) return;
  if (!scalar(c.expected) || !scalar(c.actual)) return;
  if (c.absolute_error !== "Infinity" && (typeof c.absolute_error !== "number" || !Number.isFinite(c.absolute_error) || c.absolute_error < 0)) return;
  if (typeof c.rtol !== "number" || !Number.isFinite(c.rtol) || c.rtol < 0 || typeof c.atol !== "number" || !Number.isFinite(c.atol) || c.atol < 0) return;
  return { kind: "value", ...shapes, index: [...c.index], expected: c.expected, actual: c.actual, absolute_error: c.absolute_error, rtol: c.rtol, atol: c.atol };
}
