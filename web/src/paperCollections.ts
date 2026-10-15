import type { Paper } from "./types";

export type PaperCollection = {
  id: string;
  name: string;
  name_en: string;
  description: string;
  description_en: string;
  paperIds: string[];
};

// Featured collections can overlap; compact category filters remain independent.
export const paperCollections: PaperCollection[] = [
  {
    id: "llm-basics",
    name: "LLM 基础",
    name_en: "LLM Basics",
    description: "Attention · Transformer · LoRA",
    description_en: "Attention · Transformer · LoRA",
    paperIds: ["transformer", "rmsnorm", "roformer", "gqa", "adamw", "flash-attention", "lora"],
  },
  {
    id: "retrieval",
    name: "检索与推荐",
    name_en: "Retrieval",
    description: "序列推荐 · 对比检索 · 生成式召回",
    description_en: "Sequential recommendation · Contrastive retrieval · Generative retrieval",
    paperIds: ["bpr", "lightgcn", "sasrec", "clip", "colbert", "rotation-trick", "tiger"],
  },
  {
    id: "rl",
    name: "强化学习",
    name_en: "Reinforcement Learning",
    description: "策略梯度 · 奖励优化 · 推理",
    description_en: "Policy gradients · Rewards · Reasoning",
    paperIds: [
      "gae", "ppo", "deepseekmath", "dr-grpo", "gspo", "dapo",
      "glm-5", "sdpo", "pkpo", "diffusion-nft",
    ],
  },
  {
    id: "diffusion",
    name: "扩散与 Flow",
    name_en: "Diffusion & Flow",
    description: "去噪 · Flow Matching · 蒸馏",
    description_en: "Denoising · Flow matching · Distillation",
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
    description: "高效微调 · 对齐 · 蒸馏",
    description_en: "Fine-tuning · Alignment · Distillation",
    paperIds: ["lora", "qlora", "dpo", "gkd", "opsd", "sdpo", "distillm-2"],
  },
  {
    id: "jepa-world-models",
    name: "JEPA / 世界模型",
    name_en: "JEPA / World Models",
    description: "表征学习 · 潜空间预测 · 规划",
    description_en: "Representations · Prediction · Planning",
    paperIds: ["vicreg", "i-jepa", "lejepa", "visreg", "lewm", "causal-jepa"],
  },
  {
    id: "visual-generation",
    name: "视觉生成",
    name_en: "Visual Generation",
    description: "视觉表征 · 多尺度生成",
    description_en: "Visual features · Multiscale generation",
    paperIds: ["clip", "rotation-trick", "var", "repa", "drifting", "fd-loss"],
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
