import { useEffect, useId, useRef, useState } from "react";
import { CheckCircle2, ChevronDown, ChevronLeft, ChevronRight, Circle } from "lucide-react";
import {readInterviewRoute} from './interviewNavigation';
import {preparationProgress} from './interviewPreparation';
import { withLibraryContext } from "./libraryNavigation";
import { useI18n } from "./i18n";
import { status, type Problem, type Saved } from "./types";

export function ModuleNavigator({ current, modules: paperModules, allModules, saved, ready, returnTo }: { current: Problem; modules: Problem[]; allModules?:Problem[]; saved: Record<string, Saved>; ready: boolean; returnTo?: string }) {
  const { t, locale } = useI18n();
  const track=returnTo?readInterviewRoute(returnTo)?.track:undefined;
  const progress=track?preparationProgress(track,allModules||paperModules,saved):undefined;
  const scoped=progress?.modules.some(module=>module.id===current.id)?progress:undefined;
  const modules=scoped?.modules||paperModules;
  const heading=scoped?(locale==='en'?scoped.path.name_en:scoped.path.name):t('本篇论文','This paper');
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const popupId = useId();
  const index = modules.findIndex((module) => module.id === current.id);
  const core = scoped?modules:modules.filter((module) => !module.module.optional);
  const solved = core.filter((module) => status(module, saved[module.id]) === "done").length;
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setOpen(false);
      trigger.current?.focus({ preventScroll: true });
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => { document.removeEventListener("pointerdown", outside); document.removeEventListener("keydown", escape); };
  }, [open]);
  return <div className="problem-navigation module-navigator" ref={root} onBlur={(event) => { if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget as Node)) setOpen(false); }}>
    <button className="icon-button" aria-label={t("上一个模块", "Previous module")} disabled={index <= 0} onClick={() => { location.hash = withLibraryContext(modules[index - 1].number, returnTo).slice(1); }}><ChevronLeft size={15} /></button>
    <button className="module-picker-trigger" ref={trigger} aria-expanded={open} aria-controls={popupId} aria-label={t(`选择模块，当前第 ${index + 1} 个，共 ${modules.length} 个`, `Choose module, ${index + 1} of ${modules.length}`)} onClick={() => setOpen((value) => !value)} onKeyDown={(event) => {
      if (event.key !== "ArrowDown") return;
      event.preventDefault();
      setOpen(true);
      requestAnimationFrame(() => root.current?.querySelector<HTMLElement>('.module-picker a[aria-current="page"]')?.focus());
    }}><span>{index + 1} / {modules.length}</span><ChevronDown size={12} /></button>
    <button className="icon-button" aria-label={t("下一个模块", "Next module")} disabled={index < 0 || index >= modules.length - 1} onClick={() => { location.hash = withLibraryContext(modules[index + 1].number, returnTo).slice(1); }}><ChevronRight size={15} /></button>
    {open && <nav className="module-picker" id={popupId} aria-label={heading}>
      <div className="module-picker-heading"><b>{heading}</b><span>{ready ? `${solved} / ${scoped?.total||core.length} ${scoped?t("路径模块已通过","path modules solved"):t("核心模块已通过", "core modules solved")}` : t("正在加载进度…", "Loading progress…")}</span></div>
      <div className="module-picker-links">{modules.map((module) => {
        const state = ready ? status(module, saved[module.id]) : "todo";
        const active = module.id === current.id;
        return <a key={module.id} href={withLibraryContext(module.number, returnTo)} aria-current={active ? "page" : undefined} onClick={() => setOpen(false)}>
          <span className={`module-picker-state ${state}`} role="img" aria-label={!ready ? t("正在加载进度", "Loading progress") : state === "done" ? t("已通过", "Solved") : state === "started" ? t("进行中", "In progress") : t("未开始", "Not started")}>{state === "done" ? <CheckCircle2 size={16} /> : <Circle size={16} />}</span>
          <span><b>{module.number}. {locale === "en" ? module.translations?.en?.title || module.title : module.title}</b><small>{module.estimated_minutes || 25} min{module.module.optional ? ` · ${t("可选拓展", "Optional extension")}` : ""}{active ? ` · ${t("当前模块", "Current module")}` : ""}</small></span>
          {active && <span className="module-picker-current" aria-hidden="true" />}
        </a>;
      })}</div>
    </nav>}
  </div>;
}
