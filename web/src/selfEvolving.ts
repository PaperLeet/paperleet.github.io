// Original teaching diagrams. Paper figures and method equations are linked per flow.
type Text = readonly [string, string];
export type FlowStep = {title: Text; detail: Text; node: number; update?: number};
export type EvolutionFlow = {name: string; source: string; roles: string[]; nodes: Text[]; steps: FlowStep[]};
export const evolutionFlows: Record<string, EvolutionFlow> = {
  'r-zero': {
    name: 'R-Zero', source: 'https://arxiv.org/html/2508.05004v4#S2.F2', roles: ['Challenger', 'Solver'],
    nodes: [['生成问题', 'Generate questions'], ['采样答案', 'Sample answers'], ['奖励与共识', 'Reward & consensus'], ['筛选训练集', 'Filter training data']],
    steps: [
      {title: ['出题', 'Propose'], node: 0, detail: ['Challenger 生成问题。两种角色从同一个预训练模型初始化，但分别维护参数。', 'The Challenger generates questions. Both roles start from one pretrained model but maintain separate parameters.']},
      {title: ['解题', 'Solve'], node: 1, detail: ['冻结的 Solver 对每道题采样多个答案；最常见答案的占比为一致率 p。', 'The frozen Solver samples multiple answers per question. The most frequent answer defines consistency p.']},
      {title: ['更新出题者', 'Train Challenger'], node: 2, update: 0, detail: ['用不确定性减去重复惩罚，再截断到零。GRPO 更新 Challenger，Solver 保持冻结。', 'Subtract repetition from uncertainty and floor at zero. GRPO updates the Challenger while the Solver remains frozen.']},
      {title: ['形成课程', 'Build curriculum'], node: 3, detail: ['更新后的 Challenger 生成新问题。冻结的 Solver 重新投票；只保留 0.25 ≤ p ≤ 0.75，并固定多数答案作为伪标签。', 'The updated Challenger generates fresh questions. The frozen Solver votes again; keep 0.25 ≤ p ≤ 0.75 and fix the majority answers as pseudo-labels.']},
      {title: ['更新解题者', 'Train Solver'], node: 1, update: 1, detail: ['GRPO 将 Solver 的新答案与固定伪标签比较；Challenger 保持冻结。更新后的 Solver 为下一轮出题提供反馈。', 'GRPO compares new Solver responses with the fixed pseudo-labels; the Challenger stays frozen. The updated Solver provides feedback for the next round.']},
    ],
  },
  'g-zero': {
    name: 'G-Zero', source: 'https://arxiv.org/html/2605.09959#S3.F2', roles: ['Proposer', 'Generator'],
    nodes: [['问题 + 提示', 'Question + hint'], ['有 / 无提示生成', 'Generate with / without hint'], ['Hint-δ 反馈', 'Hint-δ feedback'], ['偏好数据对', 'Preference pairs']],
    steps: [
      {title: ['出题与提示', 'Propose'], node: 0, detail: ['Proposer 生成问题 q 和提示 h。Generator 先只看问题，生成无提示答案 y。', 'The Proposer creates a question q and hint h. The Generator first answers q alone, producing unassisted response y.']},
      {title: ['对齐评分', 'Score same tokens'], node: 1, detail: ['对同一个无提示答案 y，在 q 和 q+h 两种上下文下评分；逐 token 的 log p(y|q) − log p(y|q,h) 均值就是 Hint-δ。', 'Score the same unassisted y under q and q+h. Hint-δ is the token mean of log p(y|q) − log p(y|q,h).']},
      {title: ['更新出题者', 'Train Proposer'], node: 2, update: 0, detail: ['δ 减去超长提示与重复惩罚，用 GRPO 更新 Proposer。Generator 冻结。δ 是概率变化，不是答案质量的实测提升。', 'GRPO updates the Proposer using δ minus hint-length and repetition penalties. The Generator is frozen. δ measures probability shift, not an observed quality gain.']},
      {title: ['筛选偏好对', 'Filter pairs'], node: 3, detail: ['新池子先计算全池 δ 中位数，保留 δ ≤ 中位数，再做结构检查。有提示答案为 chosen，无提示答案为 rejected；训练输入只含 q。', 'Compute the full fresh pool’s δ median, keep δ ≤ median, then apply structural checks. Assisted answers are chosen, unassisted answers rejected; the training prompt contains only q.']},
      {title: ['更新生成者', 'Train Generator'], node: 1, update: 1, detail: ['DPO 更新 Generator；轮次开始时的 Generator 副本作为冻结 reference。Proposer 冻结，下一轮使用更新后的 Generator。', 'DPO updates the Generator against a frozen round-start Generator reference. The Proposer remains frozen; the next round uses the updated Generator.']},
    ],
  },
  visplay: {
    name: 'VisPlay', source: 'https://arxiv.org/html/2511.15661#S1.F2', roles: ['Questioner', 'Reasoner'],
    nodes: [['图像条件出题', 'Image-conditioned questions'], ['视觉推理', 'Visual reasoning'], ['图像内奖励', 'Per-image reward'], ['图文伪标签', 'Image / question / label']],
    steps: [
      {title: ['读取图像', 'Read image'], node: 0, detail: ['输入来自外部的无标注图像，Questioner 为每张图生成多个问题。VisPlay 需要图像数据。', 'Start from external unlabeled images. The Questioner generates multiple questions per image. VisPlay requires image data.']},
      {title: ['采样答案', 'Sample answers'], node: 1, detail: ['冻结的 Reasoner 同时看图像和问题，多次作答得到一致率 p。', 'The frozen Reasoner sees both image and question, sampling answers to obtain consistency p.']},
      {title: ['更新出题者', 'Train Questioner'], node: 2, update: 0, detail: ['不确定性鼓励中等难度；重复比例只在同一张图的问题中计算。GRPO 更新 Questioner，Reasoner 冻结。', 'Uncertainty favors intermediate difficulty. Repetition is counted only among questions for the same image. GRPO updates the Questioner; the Reasoner is frozen.']},
      {title: ['筛选图文对', 'Filter image tasks'], node: 3, detail: ['新生成的问题经多数投票标注，保留 0.25 ≤ p ≤ 0.75 的图像、问题与伪标签三元组。', 'Majority voting labels newly generated questions. Keep image, question and pseudo-label triples with 0.25 ≤ p ≤ 0.75.']},
      {title: ['更新推理者', 'Train Reasoner'], node: 1, update: 1, detail: ['用固定伪标签通过 GRPO 更新 Reasoner；Questioner 冻结。两种角色在下一轮继续交替。', 'GRPO trains the Reasoner against fixed pseudo-labels while the Questioner is frozen. The two roles alternate again next round.']},
    ],
  },
  'mm-zero': {
    name: 'MM-Zero', source: 'https://arxiv.org/html/2603.09206#S2.F1', roles: ['Proposer', 'Coder', 'Solver'],
    nodes: [['概念与问题', 'Concept & questions'], ['代码 → 图像', 'Code → image'], ['解题反馈', 'Solver feedback'], ['筛选图文任务', 'Filter visual tasks']],
    steps: [
      {title: ['设计任务', 'Propose'], node: 0, detail: ['Proposer 生成图像描述、简单问题及其答案、困难问题。不需要外部种子图像，但需要预训练模型与渲染环境。', 'The Proposer creates a visual description, easy question and answer, and hard question. No external seed images are needed; pretrained models and a renderer are given.']},
      {title: ['渲染', 'Render'], node: 1, detail: ['Coder 将描述转换成可执行图形代码，渲染多张候选图像；记录每次成功与失败。', 'The Coder turns descriptions into executable graphics code and renders candidate images, recording every successful and failed attempt.']},
      {title: ['衡量可解性', 'Measure feedback'], node: 2, detail: ['冻结的 Solver 回答简单与困难问题。简单题与给定答案比较，困难题使用答案一致率；一致不等于正确。', 'The frozen Solver answers easy and hard questions. Easy answers are checked against the supplied answer; hard answers yield consistency, which is not correctness.']},
      {title: ['更新提案者', 'Train Proposer'], node: 2, update: 0, detail: ['所有渲染尝试共同构成分母，失败项贡献零；简单题得分上限 0.5，再加入难度与多样性等调整。仅更新 Proposer。', 'Average over all render attempts, including failures as zeros. Cap the easy score at 0.5 and add difficulty and diversity adjustments. Only the Proposer updates.']},
      {title: ['更新编码者', 'Train Coder'], node: 1, update: 1, detail: ['Coder 接收执行成功奖励、未截断的简单题得分、难度与错误惩罚；Proposer 和 Solver 保持冻结。', 'The Coder receives execution reward, uncapped easy score, difficulty and error penalties. The Proposer and Solver remain frozen.']},
      {title: ['构造解题数据', 'Build Solver data'], node: 3, detail: ['用当前 Proposer 与 Coder 生成并筛选可用图文任务，由冻结 Solver 形成多数伪标签。', 'Use the current Proposer and Coder to generate and filter usable visual tasks; the frozen Solver supplies majority pseudo-labels.']},
      {title: ['更新解题者', 'Train Solver'], node: 2, update: 2, detail: ['GRPO 用 0.9 × 多数答案一致 + 0.1 × 格式正确来训练 Solver。其他角色冻结，再进入下一轮。', 'GRPO trains the Solver with 0.9 × majority agreement + 0.1 × valid format. Other roles stay frozen before the next round.']},
    ],
  },
};

export function consensusExample(p: number, duplicates: number, valid: boolean) {
  const uncertainty = 1 - 2 * Math.abs(p - .5);
  return {uncertainty, reward: valid ? Math.max(0, uncertainty - duplicates / 10) : 0, keep: valid && p >= .25 && p <= .75};
}
export function hintExample(delta: number, length: number, valid: boolean) {
  // The other three records stay fixed; quality checks occur AFTER this full-pool median.
  const pool = [-.2, .1, .6, delta].sort((a, b) => a - b);
  const median = (pool[1] + pool[2]) / 2;
  const lengthPenalty = .03 * Math.max(length - 200, 0) / 100;
  return {median, lengthPenalty, reward: (valid ? delta - lengthPenalty : -1) - .1, keep: valid && delta <= median};
}
export function renderExample(successes: number, easy: number, consistency: number) {
  const difficulty = Math.min(consistency, 1 - consistency);
  // Four attempts, identical successful-image feedback; upstream adjustments are zero here.
  return {difficulty, proposer: successes / 4 * (Math.min(easy, .5) + difficulty), coder: 1 + easy + difficulty, failedCoder: -.1};
}
