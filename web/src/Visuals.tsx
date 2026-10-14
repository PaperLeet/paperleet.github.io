import { useLabText } from "./labI18n";
import { useState } from "react";
import { useI18n } from "./i18n";
import { difficultyAdvantage, dpoObjective, tokenGradient } from "./concepts";
import type { Aggregation } from "./concepts";

type Point = [number, number];
const colors = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
];
const nums = (n: number, fn: (i: number) => number) =>
  Array.from({ length: n }, (_, i) => fn(i));
export function Slider({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  const tx = useLabText();
  label = tx(label);
  return (
    <label className="slider">
      <span>
        {label}
        <b>{Number(value.toFixed(3))}</b>
      </span>
      <input
        aria-label={label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
      />
    </label>
  );
}
export function Chart({
  series,
  labels = [],
  xLabel = "",
  yLabel = "",
  shade,
  domain,
  focusX,
}: {
  series: Point[][];
  labels?: string[];
  xLabel?: string;
  yLabel?: string;
  shade?: [number, number];
  domain?: [number, number, number, number];
  focusX?: number;
}) {
  const tx = useLabText();
  const all = series.flat().filter((p) => p.every(Number.isFinite));
  if (!all.length) return null;
  const [xmin, xmax, ymin, ymax] = domain || [
    Math.min(...all.map((p) => p[0])),
    Math.max(...all.map((p) => p[0])),
    Math.min(0, ...all.map((p) => p[1])),
    Math.max(0.001, ...all.map((p) => p[1])),
  ];
  const x = (v: number) => 42 + ((v - xmin) / (xmax - xmin || 1)) * 462,
    y = (v: number) => 188 - ((v - ymin) / (ymax - ymin || 1)) * 157;
  return (
    <div className="chart">
      <svg
        viewBox="0 0 536 224"
        role="img"
        aria-label={`${yLabel} / ${xLabel} · ${labels.join("; ")}`}
      >
        {nums(5, (i) => i).map((i) => (
          <g key={i}>
            <line
              x1="42"
              x2="504"
              y1={31 + i * 39.25}
              y2={31 + i * 39.25}
              className="gridline"
            />
            <text x="34" y={35 + i * 39.25} textAnchor="end">
              {Number((ymax - ((ymax - ymin) * i) / 4).toFixed(2))}
            </text>
          </g>
        ))}
        {shade && (
          <rect
            x={x(shade[0])}
            y="31"
            width={Math.max(0, x(shade[1]) - x(shade[0]))}
            height="157"
            fill={colors[0]}
            opacity=".09"
          />
        )}
        {series.map((s, i) => (
          <path
            key={i}
            d={s
              .map((p, j) => p.every(Number.isFinite)
                ? `${j && s[j-1].every(Number.isFinite) ? "L" : "M"}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`
                : "")
              .join(" ")}
            fill="none"
            stroke={colors[i % colors.length]}
            strokeWidth="2.2"
            strokeDasharray={i === 1 ? "5 4" : undefined}
          />
        ))}
        {focusX !== undefined && (
          <g>
            <line
              x1={x(focusX)}
              x2={x(focusX)}
              y1="31"
              y2="188"
              stroke={colors[0]}
              strokeDasharray="3 4"
              opacity=".6"
            />
            {series.slice(0, 1).map((s, i) => {
              const closest = s.reduce((a, b) =>
                Math.abs(a[0] - focusX) < Math.abs(b[0] - focusX) ? a : b,
              );
              return (
                <circle
                  key={i}
                  cx={x(focusX)}
                  cy={y(closest[1])}
                  r="4"
                  fill={colors[0]}
                />
              );
            })}
          </g>
        )}
        {nums(5, (i) => i).map((i) => (
          <text key={i} x={42 + (462 * i) / 4} y="207" textAnchor="middle">
            {Number((xmin + ((xmax - xmin) * i) / 4).toFixed(2))}
          </text>
        ))}
        <text x="273" y="222" textAnchor="middle">
          {xLabel}
        </text>
        <text x="42" y="17">
          {yLabel}
        </text>
      </svg>
      {labels.length > 0 && (
        <div className="legend">
          {labels.map((l, i) => (
            <span key={l}>
              <i style={{ background: colors[i % colors.length] }} />
              {l}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
export function Bars({
  values,
  labels,
}: {
  values: number[];
  labels?: string[];
}) {
  const tx = useLabText();
  const max = Math.max(0.01, ...values.map(Math.abs));
  return (
    <div
      className="bars"
      role="img"
      aria-label={`${tx("柱状图")} ${values.map((v) => v.toFixed(3)).join(", ")}`}
    >
      {values.map((v, i) => (
        <div className="bar-col" key={i}>
          <b>{v.toFixed(2)}</b>
          <div className="bar-track">
            <div
              className="bar-value"
              style={{
                height: `${(Math.abs(v) / max) * 85}%`,
                background: v < 0 ? colors[1] : colors[0],
              }}
            />
          </div>
          <small>{labels?.[i] || i + 1}</small>
        </div>
      ))}
    </div>
  );
}
export function Scatter({
  sets,
  labels = ["采样结果", "真实分布"],
}: {
  sets: Point[][];
  labels?: string[];
}) {
  const tx = useLabText();
  labels = labels.map(tx);
  const all = sets.flat();
  if (!all.length) return null;
  const limit = Math.max(3, ...all.flat().map(Math.abs));
  return (
    <div className="chart">
      <svg
        className="scatter"
        viewBox="0 0 460 280"
        role="img"
        aria-label={`${tx("二维点云")} · ${labels.join(tx("与"))} · ${all.length} ${tx("点")}`}
      >
        <line x1="230" y1="15" x2="230" y2="260" className="gridline" />
        <line x1="60" y1="140" x2="400" y2="140" className="gridline" />
        {sets.map((s, i) => (
          <g key={i} fill={colors[i % colors.length]} opacity=".55">
            {s.slice(0, 1000).map((p, j) => (
              <circle
                key={j}
                cx={230 + (p[0] / limit) * 160}
                cy={140 - (p[1] / limit) * 115}
                r="2"
              />
            ))}
          </g>
        ))}
      </svg>
      <div className="legend">
        {labels.map((l, i) => (
          <span key={l}>
            <i style={{ background: colors[i] }} />
            {l}
          </span>
        ))}
      </div>
    </div>
  );
}
export function Panel({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  const tx = useLabText();
  return (
    <section className="viz-panel">
      <div className="viz-label">INTERACTIVE LAB</div>
      <h3>{title}</h3>
      <p className="viz-caption">{description}</p>
      {children}
    </section>
  );
}
export function PPO() {
  const tx = useLabText();
  const [eps, setEps] = useState(0.2),
    [ratio, setRatio] = useState(1.3);
  return (
    <Panel
      title={tx("Clipping，究竟截断了哪一边？")}
      description={tx(
        "调整 ε 与概率比 r。阴影区域的有利更新被截断，目标梯度变为零。",
      )}
    >
      <div className="controls">
        <Slider
          label={tx("ε · 截断半径")}
          value={eps}
          onChange={setEps}
          min={0.05}
          max={0.5}
        />
        <Slider
          label={tx("r · 概率比")}
          value={ratio}
          onChange={setRatio}
          max={2}
        />
      </div>
      {[1, -1].map((a) => (
        <div key={a}>
          <h4>
            A {a > 0 ? "> 0" : "< 0"}
            <span className="small-badge">
              {tx("当前 ∂L/∂r =")}{" "}
              {(a > 0 && ratio > 1 + eps) || (a < 0 && ratio < 1 - eps) ? 0 : a}
            </span>
          </h4>
          <Chart
            series={[
              nums(101, (i) => i * 0.02).map((x) => [
                x,
                Math.min(x * a, Math.max(1 - eps, Math.min(1 + eps, x)) * a),
              ]),
              [
                [0, 0],
                [2, 2 * a],
              ],
            ]}
            labels={["clipped objective", "r · A"]}
            shade={a > 0 ? [1 + eps, 2] : [0, 1 - eps]}
            domain={[0, 2, a > 0 ? 0 : -2, a > 0 ? 2 : 0]}
            xLabel="r = πθ / πold"
            yLabel="L CLIP"
            focusX={ratio}
          />
        </div>
      ))}
      <div className="formula">L = min(rA, clip(r, 1−ε, 1+ε)A)</div>
    </Panel>
  );
}
export function DPO() {
  const tx = useLabText();
  const [beta, setBeta] = useState(0.2),
    [margin, setMargin] = useState(1),
    [prompt, setPrompt] = useState(3),
    [smoothing, setSmoothing] = useState(0);
  const objective = (x: number) => dpoObjective(x, beta, smoothing);
  return (
    <Panel
      title={tx("从序列概率到偏好目标")}
      description={tx(
        "拖动 β 与标签平滑 ε，观察 loss 及实际梯度怎样变化；调整 prompt 长度，检查右移后的有效 token。",
      )}
    >
      <div className="controls">
        <Slider
          label={tx("β · 偏好温度")}
          value={beta}
          onChange={setBeta}
          min={0.05}
          max={2}
        />
        <Slider
          label={tx("margin · 对数概率比差")}
          value={margin}
          onChange={setMargin}
          min={-5}
          max={5}
          step={0.1}
        />
      </div>
      <Slider
        label="ε · label smoothing"
        value={smoothing}
        onChange={setSmoothing}
        min={0}
        max={0.5}
        step={0.01}
      />
      <Chart
        series={[
          nums(101, (i) => i / 10 - 5).map((x) => [x, objective(x).loss]),
        ]}
        labels={["−(1−ε) log σ(βh) − ε log σ(−βh)"]}
        xLabel="margin"
        yLabel="loss"
        focusX={margin}
      />
      <Chart
        series={[
          nums(101, (i) => i / 10 - 5).map((x) => [x, objective(x).gradient]),
        ]}
        labels={["∂L/∂h = β[σ(βh) − (1−ε)]"]}
        xLabel="margin h"
        yLabel={tx("实际 loss 梯度")}
        focusX={margin}
      />
      <div className="metric-strip">
        <span>
          {tx("当前 loss")}
          <b>{objective(margin).loss.toFixed(4)}</b>
        </span>
        <span>
          ∂L/∂h <b>{objective(margin).gradient.toFixed(4)}</b>
        </span>
      </div>
      <div className="viz-footnote">
        {tx(
          "ε=0 时，负梯度的大小是 βσ(−βh)；ε&gt;0 时，过大的正 margin 也会收到回拉梯度。loss 使用稳定的 softplus 计算。",
        )}
      </div>
      <Slider
        label={tx("prompt token 数")}
        value={prompt}
        onChange={setPrompt}
        min={1}
        max={6}
        step={1}
      />
      <div className="token-grid">
        {Array.from({ length: 8 }, (_, i) => (
          <div key={i} className={i >= prompt ? "active" : ""}>
            <small>{i === 0 ? "BOS" : `token ${i}`}</small>
            <b>{i === 0 ? "—" : i >= prompt ? "1" : "0"}</b>
          </div>
        ))}
      </div>
      <div className="viz-footnote">
        {tx("logits[t−1] 预测 token[t]；只对 mask=1 的回答 token 求和。")}
      </div>
    </Panel>
  );
}
export function GRPO({ dr = false }: { dr?: boolean }) {
  const tx = useLabText();
  const [rewards, setRewards] = useState([0, 0.5, 3, 1, 1.5, 2, 2.5, 0]);
  const [standard, setStandard] = useState(!dr),
    [scale, setScale] = useState(1);
  const [mode, setMode] = useState<Aggregation>(dr ? "Dr. GRPO" : "GRPO");
  const [sharedAdvantage, setSharedAdvantage] = useState(1);
  const rs = rewards.map((r) => r * scale),
    mean = rs.reduce((a, b) => a + b, 0) / rs.length;
  const std = Math.sqrt(
    rs.reduce((a, b) => a + (b - mean) ** 2, 0) / (rs.length - 1),
  );
  const adv = rs.map((r) => (r - mean) / (standard ? std + 1e-4 : 1));
  const lengths = [3, 6, 12, 24],
    maxLength = 24;
  return (
    <>
      <Panel
        title={
          dr ? tx("去掉组标准化，看清长度偏置") : tx("同一组回答，比较相对好坏")
        }
        description={tx(
          "调整 8 个回答各自的奖励，或整体缩放奖励。开关样本标准差归一化，观察 advantage 的变化。",
        )}
      >
        <div className="reward-sliders">
          {rewards.map((r, i) => (
            <Slider
              key={i}
              label={`${tx("回答")} ${i + 1} · reward`}
              value={r}
              min={0}
              max={4}
              step={0.1}
              onChange={(value) =>
                setRewards((prev) =>
                  prev.map((old, j) => (i === j ? value : old)),
                )
              }
            />
          ))}
        </div>
        <Slider
          label={tx("奖励整体缩放")}
          value={scale}
          onChange={setScale}
          min={0.1}
          max={3}
          step={0.1}
        />
        <Bars values={adv} labels={rewards.map((_, i) => `${i + 1}`)} />
        <label className="check">
          <input
            type="checkbox"
            checked={standard}
            onChange={(e) => setStandard(e.target.checked)}
          />
          {tx("除以组内样本标准差（correction=1）")}
        </label>
        <div className="formula">
          {tx("G=8 组均值=")}
          {mean.toFixed(3)}　{tx("样本 std=")}
          {std.toFixed(3)}
          <br />
          {standard ? "Aᵢ = (rᵢ − μ) / (std + 1e−4)" : "Aᵢ = rᵢ − μ"}
        </div>
      </Panel>
      <Panel
        title={tx("相同优势，不同长度：每个 token 分到多少梯度")}
        description={tx(
          "4 条回答的 advantage 相同，以隔离聚合方式的差异。切换 GRPO、DAPO 与 Dr. GRPO，观察长度 3、6、12、24 的逐 token loss 梯度。",
        )}
      >
        <div className="segmented">
          {(["GRPO", "DAPO", "Dr. GRPO"] as Aggregation[]).map((value) => (
            <button
              key={value}
              className={value === mode ? "selected" : ""}
              onClick={() => setMode(value)}
            >
              {value}
            </button>
          ))}
        </div>
        <Slider
          label={tx("所有回答共享的 advantage A")}
          value={sharedAdvantage}
          onChange={setSharedAdvantage}
          min={-2}
          max={2}
          step={0.1}
        />
        {lengths.map((length) => {
          const gradient = tokenGradient(
            mode,
            sharedAdvantage,
            length,
            lengths,
            maxLength,
          );
          return (
            <div className="token-weight-row" key={length}>
              <div>
                <b>L = {length}</b>
                <span>
                  {tx("每个有效 token：")}
                  {gradient.toFixed(5)}
                </span>
              </div>
              <div className="token-weight-cells">
                {Array.from({ length: maxLength }, (_, i) => (
                  <span
                    key={i}
                    aria-label={
                      i < length
                        ? `token ${i + 1} ${tx("梯度")} ${gradient.toFixed(6)}`
                        : `token ${i + 1} padding`
                    }
                    title={
                      i < length
                        ? `∂L/∂logπ = ${gradient.toFixed(6)}`
                        : tx("padding：梯度为零")
                    }
                    style={{
                      background:
                        i < length
                          ? `color-mix(in srgb, ${colors[gradient < 0 ? 0 : 2]} ${(0.1 + Math.min(0.85, Math.abs(gradient) * 4.8)) * 100}%, transparent)`
                          : undefined,
                    }}
                  >
                    {i < length ? "" : "·"}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
        <div className="formula">
          {mode === "GRPO"
            ? "∂L/∂logπᵢ,ₜ = −A / (N · Lᵢ)"
            : mode === "DAPO"
              ? "∂L/∂logπᵢ,ₜ = −A / ΣⱼLⱼ"
              : "∂L/∂logπᵢ,ₜ = −A / (N · Lmax)"}
          <br />
          N=4　Σ Lᵢ=45　Lmax=24　ρ=1　βKL=0
        </div>
        <p className="viz-footnote">
          {tx(
            "这里只比较聚合；Dr. GRPO 去掉组标准差的影响见上方优势图。padding 的梯度始终为零。",
          )}
        </p>
      </Panel>
      {dr && <DifficultyBias />}
    </>
  );
}
export function DifficultyBias() {
  const tx = useLabText();
  const [p, setP] = useState(0.5),
    [sample, setSample] = useState(true);
  const values = difficultyAdvantage(p, sample),
    grid = nums(99, (i) => (i + 1) / 100);
  return (
    <Panel
      title={tx("同一次正确回答，为什么难题的权重更大")}
      description={tx(
        "固定 G=8、二元奖励。拖动组内正确率 p，比较正确回答的优势；切换标准差约定，查看样本修正系数。",
      )}
    >
      <Slider
        label={tx("p · 组内正确率")}
        value={p}
        onChange={setP}
        min={0.125}
        max={0.875}
        step={0.125}
      />
      <label className="check">
        <input
          type="checkbox"
          checked={sample}
          onChange={(e) => setSample(e.target.checked)}
        />
        {tx("本题的样本 std（correction=1）")}
      </label>
      <Chart
        series={[
          grid.map((x) => [x, difficultyAdvantage(x, sample).grpo]),
          grid.map((x) => [x, difficultyAdvantage(x, sample).dr]),
        ]}
        labels={[tx("GRPO · 正确回答 |A|"), "Dr. GRPO · 1 − p"]}
        xLabel={tx("组内正确率 p")}
        yLabel={tx("正确回答 |A|")}
        focusX={p}
      />
      <div className="metric-strip">
        <span>
          GRPO<b>{values.grpo.toFixed(4)}</b>
        </span>
        <span>
          Dr. GRPO<b>{values.dr.toFixed(4)}</b>
        </span>
      </div>
      <div className="formula">
        GRPO = {sample ? "√(7/8) · " : ""}√((1−p)/p)
        <br />
        Dr. GRPO = 1 − p
      </div>
      <p className="viz-footnote">
        {tx(
          "曲线为连续延拓，G=8 时非退化正确率为 1/8 至 7/8；全组奖励相同时优势为零。这里忽略 1e−4 稳定项，样本 std 的修正为 √((G−1)/G)。",
        )}
      </p>
    </Panel>
  );
}
export function Params() {
  const tx = useLabText();
  const [phi, setPhi] = useState(0.7),
    a = Math.cos(phi),
    s = Math.sin(phi);
  return (
    <Panel
      title={tx("三种参数化，同一个旋转坐标系")}
      description={tx(
        "拖动 φ。xₜ 沿四分之一圆移动，v 是它的切向量；观察不同时间的误差放大。",
      )}
    >
      <Slider
        label={tx("φ · 旋转角")}
        value={phi}
        onChange={setPhi}
        max={Math.PI / 2}
        step={0.005}
      />
      <svg
        viewBox="0 0 500 310"
        className="geometry"
        role="img"
        aria-label={`${tx("旋转角")} ${phi.toFixed(2)}, alpha ${a.toFixed(2)}, sigma ${s.toFixed(2)}. ${tx("v 是单位圆的切向量。")}`}
      >
        <defs>
          <marker
            id="arr"
            markerWidth="6"
            markerHeight="6"
            refX="5"
            refY="3"
            orient="auto"
          >
            <path d="M0,0L6,3L0,6" fill={colors[0]} />
          </marker>
        </defs>
        <path d="M95 260H415M95 260V20" className="axis" />
        <path
          d="M305 260 A210 210 0 0 0 95 50"
          fill="none"
          className="gridline"
          strokeDasharray="4 5"
        />
        <path
          d={`M95 260 L${95 + a * 210} ${260 - s * 210}`}
          stroke={colors[0]}
          strokeWidth="3"
          markerEnd="url(#arr)"
        />
        <path
          d={`M${95 + a * 210} ${260 - s * 210}l${-s * 85} ${-a * 85}`}
          stroke={colors[2]}
          strokeWidth="3"
        />
        <path
          d={`M${95 + a * 210} 260V${260 - s * 210}H95`}
          className="gridline"
          strokeDasharray="4 4"
        />
        <circle cx={95 + a * 210} cy={260 - s * 210} r="6" fill={colors[0]} />
        <text x="420" y="265">
          x₀
        </text>
        <text x="86" y="18">
          ε
        </text>
        <text x={100 + a * 210} y={252 - s * 210}>
          xₜ
        </text>
        <text
          x={90 + a * 210 - s * 85}
          y={252 - s * 210 - a * 85}
          fill={colors[2]}
        >
          v
        </text>
        <text x="180" y="290">
          α = {a.toFixed(3)}　σ = {s.toFixed(3)}
        </text>
      </svg>
      <div className="formula">
        xₜ = αx₀ + σε　　v = αε − σx₀
        <br />
        x₀ = αxₜ − σv　　ε = σxₜ + αv
      </div>
      <Chart
        series={[
          nums(99, (i) => (i + 1) / 100).map((x) => [
            x,
            Math.log10(1 / Math.sin((x * Math.PI) / 2)),
          ]),
          nums(99, (i) => (i + 1) / 100).map((x) => [
            x,
            Math.log10(1 / Math.cos((x * Math.PI) / 2)),
          ]),
          nums(99, (i) => (i + 1) / 100).map((x) => [
            x,
            Math.log10(
              Math.cos((x * Math.PI) / 2) / Math.sin((x * Math.PI) / 2),
            ),
          ]),
        ]}
        labels={["log₁₀(1/σ)", "log₁₀(1/α)", "log₁₀(α/σ)"]}
        xLabel="t / T"
        yLabel={tx("误差放大 · log scale")}
      />
    </Panel>
  );
}
function asPoints(v: unknown): Point[] | null {
  if (
    !Array.isArray(v) ||
    !v.length ||
    !v.every(
      (row) =>
        Array.isArray(row) &&
        row.length === 2 &&
        row.every(
          (value) => typeof value === "number" && Number.isFinite(value),
        ),
    )
  )
    return null;
  return v as Point[];
}
function NamedArtifactCharts({ value }: { value: unknown }) {
  if (!Array.isArray(value)) return null;
  return <>{value.map((chart, index) => {
    if (!chart || typeof chart !== "object" || !Array.isArray(chart.x) || !Array.isArray(chart.series)) return null;
    const series = chart.series.filter((s: any) => s && typeof s.name === "string" && Array.isArray(s.values) && s.values.length === chart.x.length && s.values.every((v: unknown) => typeof v === "number" && Number.isFinite(v)));
    if (!series.length) return null;
    if (chart.kind === "line" && chart.x.every((v: unknown) => typeof v === "number" && Number.isFinite(v))) return <section key={index}><h4>{String(chart.title || "")}</h4><Chart series={series.map((s: any) => s.values.map((y: number, i: number) => [chart.x[i], y]))} labels={series.map((s: any) => s.name)} xLabel={typeof chart.x_label === "string" ? chart.x_label : "step"} yLabel={typeof chart.y_label === "string" ? chart.y_label : "value"} /></section>;
    if (chart.kind === "bar") return <section key={index}><h4>{String(chart.title || "")}</h4>{series.map((s: any) => <div key={s.name}><h4>{s.name}</h4><Bars values={s.values} labels={chart.x.map(String)} /></div>)}</section>;
    return null;
  })}</>;
}

export function ArtifactCharts({ data, referenceRun = false, implementationLabel }: { data: Record<string, unknown>; referenceRun?: boolean; implementationLabel?: string }) {
  const tx = useLabText();
  const { t } = useI18n();
  // Named charts and metrics are the presentation contract; the same raw
  // series may also be present as diagnostics for tests and downloaded reports.
  if (Array.isArray(data.charts) && data.charts.length && data.metrics) {
    data = Object.fromEntries(Object.entries(data).filter(([key]) =>
      ["charts", "metrics", "samples", "reference_samples", "provenance"].includes(key)));
  }
  const samples = asPoints(data.samples),
    reference = asPoints(data.reference_samples),
    mmd =
      typeof data.mmd === "number" && Number.isFinite(data.mmd)
        ? data.mmd
        : null,
    threshold =
      typeof data.threshold === "number" &&
      Number.isFinite(data.threshold) &&
      data.threshold > 0
        ? data.threshold
        : null;
  return (
    <div className="artifact-list">
      <NamedArtifactCharts value={data.charts} />
      {samples && reference && (
        <section>
          <h4>{implementationLabel ? t("本次实验与参考对照", "Experiment and reference") : referenceRun ? t("参考实验与目标", "Reference experiment and target") : t("你的实现与参考对照", "Your implementation and reference")}</h4>
          <Scatter
            sets={[samples, reference]}
            labels={[implementationLabel || (referenceRun ? t("参考实现", "Reference implementation") : tx("你的实现")), t("目标 / 参考", "Target / reference")]}
          />
        </section>
      )}
      {mmd !== null && threshold !== null && (
        <div className="mmd-meter">
          <div>
            <span>
              MMD² <b>{mmd.toFixed(5)}</b>
            </span>
            <span>
              {tx("阈值")}
              {threshold.toFixed(5)}
            </span>
          </div>
          <progress
            aria-label={tx("MMD 相对通过阈值")}
            value={Math.max(0, mmd)}
            max={threshold}
            className={mmd > threshold ? "exceeded" : ""}
          />
          <small>
            {mmd <= threshold ? tx("在质量阈值以内") : tx("超过质量阈值")} ·{" "}
            {threshold > 0 ? ((mmd / threshold) * 100).toFixed(1) : "—"}%
          </small>
        </div>
      )}
      {Object.entries(data)
        .filter(
          ([key]) =>
            key !== "x_label" && key !== "charts" &&
            !(
              samples &&
              reference &&
              ["samples", "reference_samples"].includes(key)
            ),
        )
        .map(([key, value]) => {
          if (key === 'comparison_loss_curves' && value && typeof value === 'object' && !Array.isArray(value)) {
            const entries = Object.entries(value), series = entries.map(([, points]) => asPoints(points));
            return entries.length && series.every((points) => points !== null) ? (
              <section key={key}>
                <h4>{t('训练损失对比', 'Training loss comparison')}</h4>
                <Chart series={series as Point[][]} labels={entries.map(([name]) => name)}
                  xLabel={typeof data.x_label === 'string' ? data.x_label : 'step'} yLabel={t('损失', 'Loss')} />
              </section>
            ) : (
              <details key={key}><summary>{t('训练损失对比', 'Training loss comparison')}</summary><pre>{JSON.stringify(value, null, 2).slice(0, 16000)}</pre></details>
            );
          }
          const p = asPoints(value);
          if (p)
            return (
              <section key={key}>
                <h4>{key}</h4>
                {/loss|curve|error|training|history/.test(key) ? (
                  <Chart
                    series={[p]}
                    labels={[key]}
                    xLabel={typeof data.x_label === "string" ? data.x_label : "step"}
                    yLabel={key}
                  />
                ) : (
                  <Scatter sets={[p]} labels={[key]} />
                )}
              </section>
            );
          if (
            Array.isArray(value) &&
            value.length &&
            value.every((v) => typeof v === "number" && Number.isFinite(v))
          )
            return (
              <section key={key}>
                <h4>{key}</h4>
                <Chart
                  series={[(value as number[]).map((y, x) => [x, y])]}
                  labels={[key]}
                  xLabel="index / step"
                  yLabel={key}
                />
              </section>
            );
          if (
            Array.isArray(value) &&
            value.length &&
            value.every(
              (row) =>
                row !== null && typeof row === "object" && !Array.isArray(row),
            )
          ) {
            const records = value as Record<string, unknown>[],
              numeric = [...new Set(records.flatMap(Object.keys))].filter(
                (k) => records.some((row) => typeof row[k] === "number" && Number.isFinite(row[k])) &&
                  records.every((row) => row[k] == null || typeof row[k] === "number" && Number.isFinite(row[k])),
              ),
              xkey = ["step", "epoch", "lam", "t", "index"].find((k) =>
                numeric.includes(k) && records.every((row) => typeof row[k] === "number" && Number.isFinite(row[k])),
              ),
              keys = numeric.filter((k) => k !== xkey);
            if (keys.length)
              return (
                <section key={key}>
                  <h4>{key}</h4>
                  {keys.map((k) => (
                    <Chart
                      key={k}
                      series={[
                        records.map((r, i) => [
                          xkey ? Number(r[xkey]) : i,
                          r[k] == null ? NaN : Number(r[k]),
                        ]),
                      ]}
                      labels={[k]}
                      xLabel={xkey || "index"}
                      yLabel={k}
                    />
                  ))}
                </section>
              );
          }
          if (key === "display_only" && value === true)
            return <p className="notice" key={key}>{t("此实验的趋势仅供展示，不按单次趋势判分。", "This experiment displays a trend; a single-run trend is not graded.")}</p>;
          if (typeof value === "number" && Number.isFinite(value))
            return (
              <div className="artifact-number" key={key}>
                <span>{key}</span>
                <b>{Number(value.toPrecision(5))}</b>
              </div>
            );
          if (value && typeof value === "object" && !Array.isArray(value))
            return (
              <details key={key} open>
                <summary>{key}</summary>
                <ArtifactCharts data={value as Record<string, unknown>} referenceRun={referenceRun} implementationLabel={implementationLabel} />
              </details>
            );
          return (
            <details key={key}>
              <summary>{key}</summary>
              <pre>
                {(JSON.stringify(value, null, 2) ?? String(value)).slice(
                  0,
                  16000,
                )}
              </pre>
            </details>
          );
        })}
    </div>
  );
}
