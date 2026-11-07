/** Local project format. Review quotes and experiment evidence stay separate from drafts. */
export type Reviewer = {
  id: string; name: string; text: string;
  summary?: string; strengths?: string; rating?: string; confidence?: string;
  aiSummary?: { zh: string; en: string };
};
export type Issue = {
  id: string; reviewerId: string; title: string; quote: string;
  category: "experiment" | "clarification" | "writing";
  priority: "high" | "medium" | "low";
  status: "todo" | "drafting" | "ready";
  suggestion: string; outline: string; response: string;
};
export type Experiment = {
  id: string; issueId: string; title: string; protocol: string; metric: string;
  owner: string; status: "planned" | "running" | "done"; result: string;
};
export type ProjectSnapshot = { id: string; createdAt: string; label: string; markdown: string };
export type RebuttalProject = {
  id: string; title: string; venue: string; context: string;
  reviews: Reviewer[]; issues: Issue[]; experiments: Experiment[];
  opening: string; closing: string; wordLimit: number; updatedAt: string;
  createdAt?: string; archived?: boolean; snapshots?: ProjectSnapshot[];
};
export type RemovedContent = {
  projectId: string;
  issue?: { value: Issue; index: number };
  experiments: { value: Experiment; index: number }[];
};
export type Workspace = { version: 1; activeProjectId: string; projects: RebuttalProject[] };
export const WORKSPACE_KEY = "paperleet.rebuttal.v1";
type StorageLike = Pick<Storage, "getItem" | "setItem">;

export function makeId(): string {
  return globalThis.crypto?.randomUUID?.() || `rb-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

export function newProject(): RebuttalProject {
  const now = new Date().toISOString();
  return {
    id: makeId(), title: "Untitled rebuttal", venue: "OpenReview", context: "",
    reviews: [{ id: makeId(), name: "Reviewer 1", text: "" }], issues: [], experiments: [],
    opening: "", closing: "", wordLimit: 0, updatedAt: now,
    createdAt: now, archived: false, snapshots: [],
  };
}

/** Keep undo data outside the saved schema; deleting a concern also removes its experiments. */
export function removeConcern(project: RebuttalProject, id: string): { project: RebuttalProject; removed: RemovedContent | null } {
  const index = project.issues.findIndex(issue => issue.id === id);
  if (index < 0) return { project, removed: null };
  const experiments = project.experiments.flatMap((value, index) => value.issueId === id ? [{ value: { ...value }, index }] : []);
  return {
    project: { ...project, issues: project.issues.filter(issue => issue.id !== id), experiments: project.experiments.filter(experiment => experiment.issueId !== id) },
    removed: { projectId: project.id, issue: { value: { ...project.issues[index] }, index }, experiments },
  };
}

/** Removing one experiment leaves its concern and other experiments intact. */
export function removeExperiment(project: RebuttalProject, id: string): { project: RebuttalProject; removed: RemovedContent | null } {
  const index = project.experiments.findIndex(experiment => experiment.id === id);
  if (index < 0) return { project, removed: null };
  return {
    project: { ...project, experiments: project.experiments.filter(experiment => experiment.id !== id) },
    removed: { projectId: project.id, experiments: [{ value: { ...project.experiments[index] }, index }] },
  };
}

/** Restore original positions without overwriting edits or creating orphaned records. */
export function restoreRemovedContent(project: RebuttalProject, removed: RemovedContent): RebuttalProject {
  if (project.id !== removed.projectId) return project;
  let issues = project.issues;
  const issue = removed.issue;
  if (issue && !issues.some(value => value.id === issue.value.id) && project.reviews.some(review => review.id === issue.value.reviewerId)) {
    issues = [...issues];
    issues.splice(Math.max(0, issue.index), 0, { ...issue.value });
  }
  const issueIds = new Set(issues.map(value => value.id));
  const experimentIds = new Set(project.experiments.map(value => value.id));
  let experiments = project.experiments;
  for (const item of [...removed.experiments].sort((a, b) => a.index - b.index)) {
    if (!issueIds.has(item.value.issueId) || experimentIds.has(item.value.id)) continue;
    if (experiments === project.experiments) experiments = [...experiments];
    experiments.splice(Math.max(0, item.index), 0, { ...item.value });
    experimentIds.add(item.value.id);
  }
  if (issues.length > 2000 || experiments.length > 2000) return project;
  return issues === project.issues && experiments === project.experiments ? project : { ...project, issues, experiments };
}

function emptyWorkspace(): Workspace {
  const project = newProject();
  return { version: 1, activeProjectId: project.id, projects: [project] };
}

const experimentPattern = /\b(ablation|baseline|experiment|benchmark|runtime|latency|memory|scalability|robustness|variance|seeds?|statistical|significance|evaluation|compare|comparison|generaliz\w*|generalis\w*)\b|实验|消融|基线|对比|比较|泛化|鲁棒|显存|耗时|效率|随机种子|显著性|方差/i;
const writingPattern = /\b(typo|grammar|notation|readability|writing|figure|caption|table formatting|proofread)\b|错别字|排版|拼写|语法|可读性|符号|图注|措辞/i;
const highPriorityPattern = /\b(critical|major|missing|lack|insufficient|unfair|invalid|not convincing|concern)\b|主要|缺少|缺乏|不足|不公平|关键|重大|无法|不充分/i;

function planExperiment(quote: string, issueId: string, language: "zh" | "en"): Experiment {
  const zh = language === "zh";
  let title = zh ? "补充评估" : "Additional evaluation";
  let protocol = zh ? "针对审稿问题明确假设、对照组和评估数据；固定训练预算与评估设置，记录可复现命令。" : "Define the hypothesis, controls and evaluation data for this concern. Match training budgets and evaluation settings; record reproducible commands.";
  let metric = zh ? "填写主指标、误差范围和计算预算" : "Specify primary metric, uncertainty and compute budget";
  if (/ablation|消融/i.test(quote)) {
    title = zh ? "关键组件消融" : "Component ablation";
    protocol = zh ? "逐一移除被质疑的组件；保持数据、训练预算和评估过程一致，与完整方法比较。" : "Remove each questioned component individually. Keep data, training budget and evaluation identical; compare with the full method.";
  } else if (/runtime|latency|memory|scalability|显存|耗时|效率/i.test(quote)) {
    title = zh ? "效率与资源开销" : "Efficiency and resource cost";
    protocol = zh ? "在相同硬件、批大小和精度下评估方法与基线；记录预热、重复次数和输入规模。" : "Measure the method and baselines on identical hardware, batch size and precision. Record warm-up, repetitions and input scale.";
    metric = zh ? "延迟、吞吐量、峰值显存" : "Latency, throughput, peak memory";
  } else if (/variance|seeds?|statistical|significance|随机种子|显著性|方差/i.test(quote)) {
    title = zh ? "多随机种子复现" : "Multi-seed evaluation";
    protocol = zh ? "对方法和主要基线使用相同的一组独立随机种子，保留逐次结果并报告不确定性。" : "Run the method and primary baselines with the same independent seeds. Retain per-run results and report uncertainty.";
    metric = zh ? "主指标均值、标准差或置信区间" : "Primary metric mean, standard deviation or confidence interval";
  } else if (/baseline|compar|基线|对比|比较/i.test(quote)) {
    title = zh ? "补充基线对比" : "Baseline comparison";
    protocol = zh ? "确定审稿人要求的基线，使用相同数据划分、训练预算和评估协议，记录实现与调参设置。" : "Identify the requested baseline. Match data splits, training budgets and evaluation protocol; record implementations and tuning settings.";
  } else if (/robust|generaliz|generalis|泛化|鲁棒/i.test(quote)) {
    title = zh ? "泛化与鲁棒性评估" : "Generalization and robustness";
    protocol = zh ? "明确分布变化或扰动条件；在不使用测试集调参的情况下，对方法和基线采用同一评估流程。" : "Specify the distribution shift or perturbation. Evaluate the method and baselines identically without tuning on test data.";
  }
  return { id: makeId(), issueId, title, protocol, metric, owner: "", status: "planned", result: "" };
}

type Section = "concern" | "skip";
const inlineConcernHeading = /^\s*(?:#{1,6}\s*)?(?:(?:\d+[.)])\s+)?(?:\*\*|__)?(?:(?:critical\s+)?weakness(?:es)?(?:\s+and\s+questions)?|questions?(?:\s+and\s+suggestions)?(?:\s+(?:for|to)\s+(?:the\s+)?authors?)?|concerns?|limitations?|suggestions?(?:\s+for\s+improvement)?|弱点|不足(?:之处)?|缺点|问题|局限(?:性)?|改进建议)(?:\*\*|__)?\s*[:：](?:\*\*|__)?\s*(\S.*?)\s*$/i;
function sectionHeading(line: string): Section | undefined {
  const cleaned = line.trim().replace(/^#{1,6}\s*/, "").replace(/\*\*|__/g, "").replace(/^\d+[.)]\s*/, "");
  if (/^(?:(?:critical\s+)?weakness(?:es)?(?:\s+and\s+questions)?|questions?(?:\s+and\s+suggestions)?(?:\s+(?:for|to)\s+(?:the\s+)?authors?)?|concerns?|limitations?|suggestions?(?:\s+for\s+improvement)?|弱点|不足(?:之处)?|缺点|问题|局限(?:性)?|改进建议)\s*[:：]?\s*$/i.test(cleaned)) return "concern";
  if (/^(?:(?:paper\s+)?summary(?:\s+of\s+(?:the\s+)?paper)?|(?:critical\s+)?strengths?|overall\s+(?:assessment|rating|recommendation)|rating|confidence|soundness|presentation|informativeness|contribution|(?:flag\s+for\s+)?ethics(?:\s+review)?|评分|置信度|总结|摘要|优点|优势|总体评价)\s*(?:[:：].*)?$/i.test(cleaned)) return "skip";
  return undefined;
}

/** Basic mode uses explicit review structure; every quote is a substring of the source. */
function reviewPoints(text: string): string[] {
  const lines = [...text.matchAll(/[^\r\n]*(?:\r\n|\r|\n|$)/g)]
    .filter(match => match[0].length)
    .map(match => ({ content: match[0].replace(/[\r\n]+$/, ""), start: match.index! }));
  const hasSections = lines.some(line => sectionHeading(line.content) === "concern" || inlineConcernHeading.test(line.content));
  let section: Section = hasSections ? "skip" : "concern";
  let start = -1, end = -1;
  const points: string[] = [];
  const flush = () => {
    if (start < 0) return;
    const exact = text.slice(start, end).trim();
    // Separate multiple direct questions while keeping each verbatim source span.
    const questions = exact.match(/[\s\S]+?(?:[?？](?=\s|[\p{Script=Han}]|$)|$)/gu) || [];
    for (const question of questions) if (question.trim()) points.push(question.trim());
    start = -1;
  };
  for (const line of lines) {
    const inline = inlineConcernHeading.exec(line.content);
    if (inline) {
      flush(); section = "concern";
      start = line.start + line.content.lastIndexOf(inline[1]);
      end = start + inline[1].length;
      continue;
    }
    const heading = sectionHeading(line.content);
    if (heading) { flush(); section = heading; continue; }
    if (section === "skip") continue;
    if (!line.content.trim()) { flush(); continue; }
    if (/^\s*(?:[-*•‣]\s+|(?:\d+|[A-Z])[.)]\s+|\[(?:W|Q)?\d+\]\s*|(?:W|Q)\d+\s*[.．:：]\s*)/i.test(line.content)) flush();
    if (start < 0) start = line.start;
    end = line.start + line.content.length;
  }
  flush();
  return points;
}

export function splitReviews(reviews: Reviewer[], language: "zh" | "en"): { issues: Issue[]; experiments: Experiment[] } {
  const issues: Issue[] = [], experiments: Experiment[] = [];
  for (const reviewer of reviews) {
    for (const quote of reviewPoints(reviewer.text)) {
      const category = experimentPattern.test(quote) ? "experiment" : writingPattern.test(quote) ? "writing" : "clarification";
      const plain = quote.replace(/^\s*(?:[-*•‣]\s+|(?:\d+|[A-Z])[.)]\s+|\[(?:W|Q)?\d+\]\s*|(?:W|Q)\d+\s*[.．:：]\s*)/i, "").replace(/\s+/g, " ");
      if (/^(?:none|n\/?a|not applicable|no (?:questions|weaknesses|concerns)|无|暂无|没有问题|无问题|无明显不足)[.。!！]?$/i.test(plain.trim())) continue;
      const id = makeId();
      const titleCharacters = [...plain];
      const issue: Issue = {
        id, reviewerId: reviewer.id, title: titleCharacters.length > 110 ? `${titleCharacters.slice(0, 107).join("")}…` : plain,
        quote, category, priority: highPriorityPattern.test(quote) ? "high" : category === "writing" ? "low" : "medium",
        status: "todo", outline: "", response: "",
        suggestion: language === "zh"
          ? category === "experiment" ? "先明确实验要验证的结论，补充可复现的证据，再写逐点回复。" : category === "writing" ? "给出具体修改内容及论文中的位置。" : "直接回答核心问题，并引用论文中的定义、推导或已有结果。"
          : category === "experiment" ? "Define the claim to test, collect reproducible evidence, then write a point-by-point response." : category === "writing" ? "State the concrete revision and its location in the paper." : "Answer the core question directly and refer to definitions, derivations or existing results.",
      };
      issues.push(issue);
      if (category === "experiment") experiments.push(planExperiment(quote, id, language));
    }
  }
  return { issues, experiments };
}

export type ReviewerOverview = { summary: string; strengths: string; rating: string; confidence: string };

/** The local overview quotes named review sections; it does not infer a score or stance. */
export function reviewerOverview(reviewer: Reviewer, issues: Issue[]): ReviewerOverview {
  const sections: ReviewerOverview = { summary: "", strengths: "", rating: "", confidence: "" };
  let current: keyof ReviewerOverview | undefined;
  const heading = /^\s*(?:#{1,6}\s*)?(?:\*\*|__)?((?:paper\s+)?summary(?:\s+of\s+(?:the\s+)?paper)?|(?:critical\s+)?strengths?|rating|overall\s+(?:rating|recommendation)|confidence|总结|摘要|论文总结|优点|优势|评分|置信度)(?:\*\*|__)?\s*(?:[:：](?:\*\*|__)?\s*(.*))?\s*$/i;
  for (const line of reviewer.text.split(/\r\n|\r|\n/)) {
    const match = heading.exec(line);
    if (match) {
      const name = match[1].toLowerCase();
      current = /summary|总结|摘要/.test(name) ? "summary"
        : /strength|优点|优势/.test(name) ? "strengths"
          : /confidence|置信度/.test(name) ? "confidence" : "rating";
      if (match[2]?.trim()) sections[current] += `${sections[current] ? "\n" : ""}${match[2].trim()}`;
      continue;
    }
    if (sectionHeading(line) || inlineConcernHeading.test(line)) { current = undefined; continue; }
    if (current) sections[current] += `${sections[current] ? "\n" : ""}${line}`;
  }
  const concernSummary = issues.filter(issue => issue.reviewerId === reviewer.id)
    .slice(0, 3).map(issue => issue.title.trim()).filter(Boolean).join("; ");
  const characters = [...concernSummary];
  return {
    summary: reviewer.summary ?? (sections.summary.trim() || (characters.length > 600 ? `${characters.slice(0, 597).join("")}…` : concernSummary)),
    strengths: reviewer.strengths ?? sections.strengths.trim(),
    rating: reviewer.rating ?? sections.rating.trim(),
    confidence: reviewer.confidence ?? sections.confidence.trim(),
  };
}

/** Import exported review notes without confusing a project backup with reviewer feedback. */
export function importReviewFile(value: unknown): Reviewer[] {
  const rows = Array.isArray(value) ? value : list(object(value).notes, "OpenReview notes", 100);
  if (rows.length > 100) throw Error("Too many reviewers");
  const reviews: Reviewer[] = [];
  const fieldText = (field: unknown): string => {
    if (typeof field === "string") return field;
    if (typeof field === "number" && Number.isFinite(field)) return String(field);
    if (field && typeof field === "object" && !Array.isArray(field)) {
      const wrapped = (field as Record<string, unknown>).value;
      if (typeof wrapped === "string") return wrapped;
      if (typeof wrapped === "number" && Number.isFinite(wrapped)) return String(wrapped);
    }
    return "";
  };
  for (const raw of rows) {
    const row = object(raw);
    let text: string, name: string;
    let metadata: Partial<ReviewerOverview> & Pick<Reviewer, "aiSummary"> = {};
    if (typeof row.text === "string") {
      text = string(row.text, "review text");
      name = typeof row.name === "string" ? string(row.name, "reviewer name", 300) : `Reviewer ${reviews.length + 1}`;
      metadata = reviewerMetadata(row);
    } else {
      if (!row.content || typeof row.content !== "object" || Array.isArray(row.content)) continue;
      const content = row.content as Record<string, unknown>;
      const normalized = new Map(Object.entries(content).map(([key, field]) => [key.toLowerCase().replace(/[ _-]+/g, " ").trim(), fieldText(field)]));
      const sections: string[] = [];
      let substantive = false;
      const fields: [string[], string, boolean][] = [
        [["summary", "paper summary", "summary of the paper"], "Summary", false],
        [["strengths", "strength", "critical strengths"], "Strengths", false],
        [["weaknesses", "weakness", "critical weaknesses", "limitations", "concerns"], "Weaknesses", true],
        [["questions", "questions for authors", "questions to authors"], "Questions", true],
        [["review", "main review", "detailed review"], "", true],
      ];
      for (const [keys, heading, concern] of fields) {
        for (const key of keys) {
          const body = normalized.get(key)?.trim();
          if (!body) continue;
          sections.push(heading ? `${heading}:\n${body}` : `Questions:\n${body}`);
          substantive ||= concern;
        }
      }
      if (!substantive) continue;
      const first = (...keys: string[]) => keys.map(key => normalized.get(key)?.trim()).find(Boolean);
      const importedMetadata = {
        summary: first("summary", "paper summary", "summary of the paper"),
        strengths: first("strengths", "strength", "critical strengths"),
        rating: first("rating", "overall rating", "recommendation", "overall recommendation"),
        confidence: first("confidence"),
      };
      metadata = reviewerMetadata(importedMetadata);
      if (metadata.rating) sections.push(`Rating: ${metadata.rating}`);
      if (metadata.confidence) sections.push(`Confidence: ${metadata.confidence}`);
      text = string(sections.join("\n\n"), "review text");
      name = fieldText(content.reviewer) || (typeof row.name === "string" ? row.name : "") || `Reviewer ${reviews.length + 1}`;
      name = string(name, "reviewer name", 300);
    }
    if (!text.trim()) continue;
    const proposed = typeof row.id === "string" && /^[a-zA-Z0-9_-]{1,150}$/.test(row.id) ? row.id : makeId();
    const id = reviews.some(review => review.id === proposed) ? makeId() : proposed;
    reviews.push({ id, name: name.trim() || `Reviewer ${reviews.length + 1}`, text, ...metadata });
  }
  if (!reviews.length) throw Error("No reviewer feedback found. Import OpenReview notes or an array of {name, text} reviews.");
  return reviews;
}

export function assembleResponse(project: RebuttalProject, reviewerId?: string): string {
  const sections: string[] = [];
  if (project.opening.trim()) sections.push(project.opening.trim());
  for (const reviewer of project.reviews) {
    if (reviewerId && reviewer.id !== reviewerId) continue;
    const drafts = project.issues.filter(issue => issue.reviewerId === reviewer.id && issue.response.trim());
    if (!drafts.length) continue;
    const responses = drafts.map((issue, index) => [
      `**${index + 1}. ${issue.title}**`,
      issue.quote.split(/\r?\n/).map(line => `> ${line}`).join("\n"),
      issue.response.trim(),
    ].join("\n\n"));
    sections.push([`## ${reviewer.name}`, ...responses].join("\n\n"));
  }
  if (project.closing.trim()) sections.push(project.closing.trim());
  return sections.join("\n\n");
}

/** Save an immutable export version; later draft edits never change this Markdown. */
export function saveResponseSnapshot(project: RebuttalProject, label?: string): RebuttalProject {
  const markdown = assembleResponse(project);
  if (!markdown.trim()) throw Error("Write a response before saving a version.");
  const snapshots = project.snapshots || [];
  if (snapshots.length >= 100) throw Error("A paper can contain up to 100 saved versions.");
  const now = new Date().toISOString();
  const snapshot: ProjectSnapshot = {
    id: makeId(), createdAt: now,
    label: string(label?.trim() || `Version ${snapshots.length + 1}`, "version label", 300),
    markdown: string(markdown, "version Markdown", 2_000_000),
  };
  return { ...project, updatedAt: now, snapshots: [...snapshots, snapshot] };
}

/** Approximate editorial count: CJK characters and other language words. */
export function countWords(text: string): number {
  const cjk = text.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) || [];
  const remainder = text.replace(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu, " ");
  return cjk.length + (remainder.match(/[\p{L}\p{N}]+(?:['’_-][\p{L}\p{N}]+)*/gu) || []).length;
}

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw Error("Invalid rebuttal workspace object");
  return value as Record<string, unknown>;
}
function string(value: unknown, name: string, max = 200_000): string {
  if (typeof value !== "string" || value.length > max) throw Error(`Invalid ${name}`);
  return value;
}
function reviewerMetadata(value: Record<string, unknown>): Partial<ReviewerOverview> & Pick<Reviewer, "aiSummary"> {
  const result: Partial<ReviewerOverview> & Pick<Reviewer, "aiSummary"> = {};
  for (const name of ["summary", "strengths", "rating", "confidence"] as const) {
    if (value[name] !== undefined) result[name] = string(value[name], `reviewer ${name}`, name === "rating" || name === "confidence" ? 2000 : 200_000);
  }
  if (value.aiSummary !== undefined) {
    const summary = object(value.aiSummary);
    if (Object.keys(summary).length !== 2 || !("zh" in summary) || !("en" in summary)) throw Error("Invalid bilingual reviewer summary");
    const zh = string(summary.zh, "Chinese reviewer summary", 6000);
    const en = string(summary.en, "English reviewer summary", 6000);
    if (!zh.trim() || !en.trim()) throw Error("Both reviewer summary languages are required");
    result.aiSummary = { zh, en };
  }
  return result;
}
function date(value: unknown, name: string): string {
  const text = string(value, name, 50);
  if (!Number.isFinite(Date.parse(text))) throw Error(`Invalid ${name}`);
  return text;
}
function identifier(value: unknown): string {
  const id = string(value, "identifier", 150);
  if (!/^[a-zA-Z0-9_-]+$/.test(id)) throw Error("Invalid identifier");
  return id;
}
function list(value: unknown, name: string, max: number): unknown[] {
  if (!Array.isArray(value) || value.length > max) throw Error(`Invalid ${name}`);
  return value;
}
function option<T extends string>(value: unknown, values: readonly T[], name: string): T {
  if (!values.includes(value as T)) throw Error(`Invalid ${name}`);
  return value as T;
}
function uniqueIds(items: { id: string }[], name: string) {
  if (new Set(items.map(item => item.id)).size !== items.length) throw Error(`Duplicate ${name} identifiers`);
}

/** Reconstruct known fields so imported JSON cannot carry unexpected executable or inherited data. */
export function validateWorkspace(value: unknown): Workspace {
  const data = object(value);
  if (data.version !== 1) throw Error("Unsupported rebuttal workspace version");
  const projects = list(data.projects, "projects", 100).map(raw => {
    const project = object(raw);
    const reviews: Reviewer[] = list(project.reviews, "reviewers", 100).map(rawReview => {
      const review = object(rawReview);
      return { id: identifier(review.id), name: string(review.name, "reviewer name", 300), text: string(review.text, "review text"), ...reviewerMetadata(review) };
    });
    uniqueIds(reviews, "reviewer");
    const reviewerIds = new Set(reviews.map(review => review.id));
    const issues: Issue[] = list(project.issues, "issues", 2000).map(rawIssue => {
      const issue = object(rawIssue), reviewerId = identifier(issue.reviewerId);
      if (!reviewerIds.has(reviewerId)) throw Error("Issue refers to an unknown reviewer");
      return {
        id: identifier(issue.id), reviewerId, title: string(issue.title, "issue title", 2000), quote: string(issue.quote, "review quote"),
        category: option(issue.category, ["experiment", "clarification", "writing"] as const, "issue category"),
        priority: option(issue.priority, ["high", "medium", "low"] as const, "issue priority"),
        status: option(issue.status, ["todo", "drafting", "ready"] as const, "issue status"),
        suggestion: string(issue.suggestion, "suggestion"), outline: string(issue.outline, "outline"), response: string(issue.response, "response"),
      };
    });
    uniqueIds(issues, "issue");
    const issueIds = new Set(issues.map(issue => issue.id));
    const experiments: Experiment[] = list(project.experiments, "experiments", 2000).map(rawExperiment => {
      const experiment = object(rawExperiment), issueId = identifier(experiment.issueId);
      if (!issueIds.has(issueId)) throw Error("Experiment refers to an unknown issue");
      return {
        id: identifier(experiment.id), issueId, title: string(experiment.title, "experiment title", 2000),
        protocol: string(experiment.protocol, "experiment protocol"), metric: string(experiment.metric, "experiment metric", 10_000),
        owner: string(experiment.owner, "experiment owner", 300),
        status: option(experiment.status, ["planned", "running", "done"] as const, "experiment status"),
        result: string(experiment.result, "experiment result"),
      };
    });
    uniqueIds(experiments, "experiment");
    if (!Number.isInteger(project.wordLimit) || (project.wordLimit as number) < 0 || (project.wordLimit as number) > 100_000) throw Error("Invalid word limit");
    const updatedAt = date(project.updatedAt, "update date");
    const extra: Pick<RebuttalProject, "createdAt" | "archived" | "snapshots"> = {};
    if (project.createdAt !== undefined) extra.createdAt = date(project.createdAt, "creation date");
    if (project.archived !== undefined) {
      if (typeof project.archived !== "boolean") throw Error("Invalid archived state");
      extra.archived = project.archived;
    }
    if (project.snapshots !== undefined) {
      extra.snapshots = list(project.snapshots, "saved versions", 100).map(rawSnapshot => {
        const snapshot = object(rawSnapshot);
        return { id: identifier(snapshot.id), createdAt: date(snapshot.createdAt, "version date"),
          label: string(snapshot.label, "version label", 300), markdown: string(snapshot.markdown, "version Markdown", 2_000_000) };
      });
      uniqueIds(extra.snapshots, "version");
    }
    return {
      id: identifier(project.id), title: string(project.title, "project title", 2000), venue: string(project.venue, "venue", 300),
      context: string(project.context, "paper context"), reviews, issues, experiments,
      opening: string(project.opening, "opening"), closing: string(project.closing, "closing"), wordLimit: project.wordLimit as number, updatedAt, ...extra,
    };
  });
  uniqueIds(projects, "project");
  const activeProjectId = identifier(data.activeProjectId);
  if (!projects.some(project => project.id === activeProjectId)) throw Error("Active project is missing");
  return { version: 1, activeProjectId, projects };
}

export function readWorkspace(storage: StorageLike): { workspace: Workspace; error: boolean } {
  try {
    const raw = storage.getItem(WORKSPACE_KEY);
    if (raw === null) return { workspace: emptyWorkspace(), error: false };
    if (raw.length > 20_000_000) throw Error("Workspace is too large");
    return { workspace: validateWorkspace(JSON.parse(raw)), error: false };
  } catch {
    return { workspace: emptyWorkspace(), error: true };
  }
}

export function writeWorkspace(storage: StorageLike, workspace: Workspace): boolean {
  try {
    const serialized = JSON.stringify(validateWorkspace(workspace));
    if (serialized.length > 20_000_000) return false;
    storage.setItem(WORKSPACE_KEY, serialized);
    return true;
  } catch { return false; }
}

export function demoProject(): RebuttalProject {
  const project = newProject();
  project.title = "Demo · Efficient adaptation with sparse updates";
  project.venue = "ICLR · Demo";
  project.context = "Illustrative example only. This project demonstrates how reviewer concerns, experiment plans and response drafts connect. It does not contain real research results.";
  project.reviews = [
    { id: makeId(), name: "Reviewer 1", text: "Summary:\nThe paper studies efficient adaptation with sparse parameter updates.\n\nStrengths:\nThe motivation is clear.\n\nWeaknesses:\n1. The ablation study is missing. How much does the update selection rule contribute?\n2. The comparison with a dense fine-tuning baseline needs a matched training budget.\n\nQuestions:\n3. How are the active parameters selected at each step?" },
    { id: makeId(), name: "Reviewer 2", text: "Weaknesses:\n1. Please report peak memory and runtime on the same hardware as the baselines.\n\nQuestions:\n2. The notation for the update mask in Section 3 is inconsistent." },
  ];
  const parsed = splitReviews(project.reviews, "en");
  project.issues = parsed.issues;
  project.experiments = parsed.experiments;
  const clarification = project.issues.find(issue => issue.category === "clarification");
  if (clarification) {
    clarification.status = "ready";
    clarification.outline = "Explain selection at each step; point to Algorithm 1. This is illustrative text, to be replaced with the paper's actual rule.";
    clarification.response = "[DEMO — replace with your method] At each optimization step, we rank parameter groups by the selection score and update the selected groups. Algorithm 1 specifies the score and selection budget. We will make the timing of this selection explicit in Section 3.";
  }
  if (project.experiments[0]) {
    project.experiments[0].status = "running";
    project.experiments[0].owner = "Author A";
  }
  project.opening = "Thank you for the careful review. We address each point below.";
  project.wordLimit = 1000;
  return project;
}
