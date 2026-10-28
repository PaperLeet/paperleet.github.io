import type { Paper } from "./types";

export type PaperCollection = {
  id: string;
  kind?: "course";
  name: string;
  name_en: string;
  paperIds: string[];
  /** Complete reading list size; paperIds continues to contain executable papers only. */
  readingCount?: number;
};

// Featured collections can overlap; compact category filters remain independent.
export const paperCollections: PaperCollection[] = [
  {
    id: "llm-basics",
    kind: "course",
    name: "LLM 基础",
    name_en: "LLM Basics",
    paperIds: ["llm-training-generation"],
  },
  {
    id: "llm-frontier",
    name: "LLM 前沿",
    name_en: "LLM Frontier",
    paperIds: ["transformer", "rmsnorm", "roformer", "gqa", "adamw", "flash-attention", "lora", "kimi-k3", "glm-5", "es-at-scale", "neural-thickets"],
  },
  {
    id: "multimodal",
    readingCount: 16,
    name: "多模态",
    name_en: "Multimodal",
    paperIds: ["clip", "siglip"],
  },
  {
    id: "robotics",
    readingCount: 16,
    name: "机器人学习",
    name_en: "Robotics",
    paperIds: ["diffusion-policy", "supermimic"],
  },
  {
    id: "ai-infra",
    readingCount: 16,
    name: "AI 基础设施",
    name_en: "AI Infra",
    paperIds: ["flash-attention", "paged-attention", "gqa", "qlora", "svdquant"],
  },
  {
    id: "diffusion-rl",
    readingCount: 14,
    name: "扩散强化学习",
    name_en: "Diffusion RL",
    paperIds: ["diffusion-dpo", "flow-grpo", "diffusion-nft"],
  },
  {
    id: "self-evolving",
    name: "自演化",
    name_en: "Self-Evolving",
    paperIds: ["r-zero", "g-zero", "visplay", "mm-zero"],
  },
  {
    id: "retrieval",
    name: "检索与推荐",
    name_en: "Retrieval",
    paperIds: ["bpr", "lightgcn", "graphsage", "sasrec", "clip", "colbert", "rotation-trick", "tiger"],
  },
  {
    id: "rl",
    name: "强化学习",
    name_en: "Reinforcement Learning",
    paperIds: [
      "gae", "ppo", "deepseekmath", "dr-grpo", "gspo", "dapo",
      "glm-5", "sdpo", "pkpo", "diffusion-nft",
    ],
  },
  {
    id: "diffusion",
    name: "扩散与 Flow",
    name_en: "Diffusion & Flow",
    paperIds: [
      "ddpm", "ddim", "progressive-distillation", "flow-matching",
      "consistency", "dmd", "dmd2", "meanflow", "jit", "repa",
      "block-diffusion", "svdquant", "daps", "diffusion-nft", "fd-loss",
    ],
  },
  {
    id: "post-training",
    name: "后训练",
    name_en: "Post-training",
    paperIds: ["lora", "qlora", "dpo", "gkd", "opsd", "sdpo", "distillm-2", "es-at-scale", "visual-domain-prompt", "neural-thickets"],
  },
  {
    id: "jepa-world-models",
    name: "JEPA / 世界模型",
    name_en: "JEPA / World Models",
    paperIds: ["vicreg", "i-jepa", "lejepa", "visreg", "lewm", "causal-jepa"],
  },
  {
    id: "visual-generation",
    name: "视觉生成",
    name_en: "Visual Generation",
    paperIds: ["clip", "simclr", "moco", "rotation-trick", "var", "repa", "drifting", "fd-loss"],
  },
];

export function papersInCollection(papers: Paper[], collection?: PaperCollection): Paper[] {
  if (!collection) return papers;
  const byId = new Map(papers.map((paper) => [paper.id, paper]));
  return collection.paperIds.flatMap((id) => {
    const paper = byId.get(id);
    return paper ? [paper] : [];
  });
}

export function collectionIdFromRoute(route: string): string | undefined {
  // Keep previously shared learning-list links usable.
  return /^(?:collections|lists)\/([a-z0-9_-]+)$/.exec(route.split('?')[0])?.[1];
}
