import { useId, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { Archive, ArrowDownToLine, ArrowRight, FileText, Plus, Search, Undo2, Upload, X } from "lucide-react";
import { useI18n } from "./i18n";
import type { RebuttalProject, Workspace } from "./rebuttalModel";
import "./rebuttalLibrary.css";

type Props = {
  workspace: Workspace;
  onOpen: (id: string) => void;
  onCreate: (title: string, venue: string) => void;
  onArchive: (id: string, archived: boolean) => void;
  onImport: () => void;
  onExport: () => void;
  onSample: () => void;
  userControl?: ReactNode;
};

function isPristineTemplate(project: RebuttalProject) {
  return project.title === "Untitled rebuttal" && !project.archived &&
    !project.issues.length && !project.experiments.length && !project.snapshots?.length &&
    !project.context.trim() && !project.opening.trim() && !project.closing.trim() &&
    !project.reviews.some(review => review.text.trim() || review.summary?.trim() || review.aiSummary ||
      review.strengths?.trim() || review.rating?.trim() || review.confidence?.trim());
}

export default function RebuttalLibrary({
  workspace, onOpen, onCreate, onArchive, onImport, onExport, onSample, userControl,
}: Props) {
  const { locale, t } = useI18n();
  const [query, setQuery] = useState("");
  const [archived, setArchived] = useState(false);
  const [creating, setCreating] = useState(false);
  const [title, setTitle] = useState("");
  const [venue, setVenue] = useState("");
  const createButton = useRef<HTMLButtonElement>(null);
  const createTitleId = useId();
  const papers = workspace.projects.filter(project => !isPristineTemplate(project));
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const visible = papers.filter(project => Boolean(project.archived) === archived &&
    `${project.title} ${project.venue} ${project.reviews.map(review => review.name).join(" ")}`
      .toLocaleLowerCase().includes(normalizedQuery))
    .sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
  const dateFormatter = new Intl.DateTimeFormat(locale === "zh" ? "zh-CN" : "en-US", {
    year: "numeric", month: "short", day: "numeric",
  });

  function closeCreate() {
    setCreating(false);
    requestAnimationFrame(() => createButton.current?.focus());
  }
  function createPaper(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!title.trim()) return;
    onCreate(title.trim(), venue.trim());
    setTitle(""); setVenue(""); setCreating(false); setArchived(false); setQuery("");
  }

  return <section className="rsl-library" aria-label={t("论文工作区", "Paper workspace")}>
    <header className="rsl-heading">
      <h1>{t("我的论文", "My papers")}</h1>
      <div className="rsl-heading-actions">
        {userControl && <div className="rsl-user-control">{userControl}</div>}
        <button ref={createButton} className="rst-button rst-primary" onClick={() => setCreating(true)}>
          <Plus size={16} />{t("新建论文", "New paper")}
        </button>
      </div>
    </header>

    {creating && <form className="rsl-create" onSubmit={createPaper} aria-labelledby={createTitleId}
      onKeyDown={event => { if (event.key === "Escape") { event.preventDefault(); closeCreate(); } }}>
      <header><h2 id={createTitleId}>{t("新建论文", "New paper")}</h2>
        <button type="button" className="rst-icon-button" aria-label={t("取消新建", "Cancel new paper")} onClick={closeCreate}><X size={18} /></button>
      </header>
      <div className="rsl-create-fields">
        <label><span>{t("论文标题", "Paper title")}</span><input autoFocus required maxLength={2000}
          value={title} onChange={event => setTitle(event.target.value)} /></label>
        <label><span>{t("会议 / 期刊", "Venue")}</span><input maxLength={300} placeholder="ICLR / ICML / …"
          value={venue} onChange={event => setVenue(event.target.value)} /></label>
      </div>
      <footer><button type="button" className="rst-button" onClick={closeCreate}>{t("取消", "Cancel")}</button>
        <button type="submit" className="rst-button rst-primary" disabled={!title.trim()}>{t("创建并打开", "Create & open")}<ArrowRight size={15} /></button>
      </footer>
    </form>}

    <div className="rsl-toolbar">
      <nav aria-label={t("论文状态", "Paper status")}>
        <button className={!archived ? "active" : ""} aria-pressed={!archived} onClick={() => setArchived(false)}>{t("进行中", "Active")}</button>
        <button className={archived ? "active" : ""} aria-pressed={archived} onClick={() => setArchived(true)}>{t("历史论文", "Archived")}</button>
      </nav>
      <label className="rsl-search"><Search size={16} /><input aria-label={t("搜索论文", "Search papers")}
        placeholder={t("搜索论文…", "Search papers…")} value={query} onChange={event => setQuery(event.target.value)} />
      </label>
      <div className="rsl-backup-actions">
        <button className="rst-icon-button" onClick={onImport} aria-label={t("导入备份", "Import backup")} title={t("导入备份", "Import backup")}><Upload size={17} /></button>
        <button className="rst-icon-button" onClick={onExport} aria-label={t("导出备份", "Export backup")} title={t("导出备份", "Export backup")}><ArrowDownToLine size={17} /></button>
      </div>
    </div>

    {visible.length ? <div className="rsl-paper-list">{visible.map(project => {
      const ready = project.issues.filter(issue => issue.status === "ready").length;
      const reviewers = project.reviews.filter(review => review.text.trim() || review.summary?.trim() || review.aiSummary ||
        project.issues.some(issue => issue.reviewerId === review.id)).length;
      const updatedAt = new Date(project.updatedAt);
      return <article className="rsl-paper" key={project.id}>
        <button className="rsl-paper-open" onClick={() => onOpen(project.id)} aria-label={`${t("打开", "Open")} ${project.title || t("未命名论文", "Untitled paper")}`}>
          <span className="rsl-paper-icon"><FileText size={21} /></span>
          <span className="rsl-paper-info"><strong>{project.title || t("未命名论文", "Untitled paper")}</strong>
            <span className="rsl-paper-meta">{project.venue && <span>{project.venue}</span>}<span>{reviewers} {t("位 reviewer", "reviewers")}</span></span>
          </span>
          <span className={`rsl-paper-progress ${project.issues.length && ready === project.issues.length ? "complete" : ""}`}>
            {ready} / {project.issues.length} {t("条回复就绪", "responses ready")}
          </span>
          <time dateTime={project.updatedAt} title={updatedAt.toLocaleString(locale === "zh" ? "zh-CN" : "en-US")}>
            {dateFormatter.format(updatedAt)}
          </time>
          <ArrowRight className="rsl-open-arrow" size={17} />
        </button>
        <button className="rst-icon-button rsl-archive" onClick={() => onArchive(project.id, !archived)}
          title={archived ? t("恢复到进行中", "Restore to active") : t("归档论文", "Archive paper")}
          aria-label={`${archived ? t("恢复", "Restore") : t("归档", "Archive")} ${project.title || t("未命名论文", "Untitled paper")}`}>
          {archived ? <Undo2 size={17} /> : <Archive size={17} />}
        </button>
      </article>;
    })}</div> : <div className="rsl-empty">
      {archived ? <Archive size={29} /> : <FileText size={29} />}
      <h2>{normalizedQuery ? t("没有匹配的论文", "No matching papers") : archived ? t("暂无历史论文", "No archived papers") : t("还没有论文", "No papers yet")}</h2>
      {normalizedQuery ? <button className="rst-button" onClick={() => setQuery("")}>{t("清除搜索", "Clear search")}</button>
        : !archived && !creating && <div className="rsl-empty-actions"><button className="rst-button" onClick={() => setCreating(true)}><Plus size={15} />{t("新建论文", "New paper")}</button><button className="rsl-sample" onClick={onSample}>{t("查看示例", "View example")}</button></div>}
    </div>}
  </section>;
}
