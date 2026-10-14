import { status } from "./types";
import type { Problem, Saved } from "./types";

// Start with small, self-contained Transformer building blocks before adaptations.
const foundations = [
  "architecture/attention",
  "architecture/layer_norm",
  "architecture/rms_norm",
  "architecture/positional_encoding",
  "architecture/rope",
  "architecture/positionwise_ffn",
  "architecture/multihead_attention",
  "architecture/gqa",
  "finetuning/lora",
];

/** Resume an unfinished draft first; otherwise suggest an unsolved core module. */
export function recommendPractice(problems: Problem[], saved: Record<string, Saved>): Problem | undefined {
  const remaining = problems.filter((p) => !p.module.optional && status(p, saved[p.id]) !== "done");
  const started = remaining.filter((p) => status(p, saved[p.id]) === "started");
  if (started.length) {
    return started.sort((a, b) => (saved[b.id]?.updated || "").localeCompare(saved[a.id]?.updated || "") || a.number - b.number)[0];
  }
  const rank = (p: Problem) => {
    const index = foundations.indexOf(p.id);
    return index < 0 ? foundations.length : index;
  };
  const difficulty: Record<string, number> = { E: 0, M: 1, H: 2 };
  return remaining.sort((a, b) => rank(a) - rank(b) || (difficulty[a.difficulty] ?? 3) - (difficulty[b.difficulty] ?? 3) || a.number - b.number)[0];
}

/** Continue the paper in its declared order before suggesting another paper. */
export function nextPractice(current: Problem, moduleIds: string[], problems: Problem[], saved: Record<string, Saved>): Problem | undefined {
  const index = moduleIds.indexOf(current.id);
  const ordered = index < 0 ? moduleIds : [...moduleIds.slice(index + 1), ...moduleIds.slice(0, index)];
  const byId = new Map(problems.map((p) => [p.id, p]));
  for (const id of ordered) {
    const p = byId.get(id);
    if (p && p.id !== current.id && !p.module.optional && status(p, saved[p.id]) !== "done") return p;
  }
  return recommendPractice(problems.filter((p) => p.id !== current.id), saved);
}
