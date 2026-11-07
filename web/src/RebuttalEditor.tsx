import { useId, useLayoutEffect, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { Bold, Italic, Heading1, Heading2, Palette, Quote, Minus, Square, Code2, Table2, List, ListOrdered, Link2, Sigma, Undo2, Redo2, Copy, Check } from "lucide-react";
import RebuttalPreview from "./RebuttalPreview";
import { formatRebuttal, rebuttalShortcut, REBUTTAL_COLORS, toOpenReviewMarkdown } from "./rebuttalFormatting";
import type { RebuttalFormat } from "./rebuttalFormatting";
import "./rebuttalEditor.css";

type Props = { value: string; onChange: (value: string) => void; readOnly?: boolean; locale: "zh" | "en"; actions?: ReactNode; footer?: ReactNode };
type Selection = { value: string; start: number; end: number; direction: "forward" | "backward" | "none"; top: number; left: number };
const HISTORY_LIMIT = 100;

export default function RebuttalEditor({ value, onChange, readOnly = false, locale, actions, footer }: Props) {
  const zh = locale === "zh";
  const id = useId();
  const [preview, setPreview] = useState(readOnly);
  const [color, setColor] = useState<string>(REBUTTAL_COLORS[0].value);
  const [message, setMessage] = useState("");
  const [copied, setCopied] = useState(false);
  const [, setRevision] = useState(0);
  const area = useRef<HTMLTextAreaElement>(null);
  const container = useRef<HTMLDivElement>(null);
  const lastValue = useRef(value);
  const selection = useRef<Selection>({ value, start: 0, end: 0, direction: "none", top: 0, left: 0 });
  const pending = useRef<Selection | null>(null);
  const undo = useRef<Selection[]>([]);
  const redo = useRef<Selection[]>([]);
  const composing = useRef(false);
  const beforeInput = useRef<Selection | null>(null);
  const typedAt = useRef(0);
  const t = (cn: string, en: string) => zh ? cn : en;
  const remember = (): Selection => {
    const element = area.current;
    const snapshot = element ? { value: lastValue.current, start: element.selectionStart, end: element.selectionEnd, direction: element.selectionDirection, top: element.scrollTop, left: element.scrollLeft } : selection.current;
    if (snapshot.start !== selection.current.start || snapshot.end !== selection.current.end) typedAt.current = 0;
    selection.current = snapshot;
    return snapshot;
  };
  const refresh = () => setRevision(current => current + 1);
  useLayoutEffect(() => {
    if (value !== lastValue.current) {
      // The parent replaced the document (or rejected an edit). Never undo across that boundary.
      undo.current = []; redo.current = []; pending.current = null; typedAt.current = 0; setCopied(false);
      lastValue.current = value;
      selection.current = { ...selection.current, value, start: Math.min(value.length, selection.current.start), end: Math.min(value.length, selection.current.end) };
      refresh();
    }
    const next = pending.current, element = area.current;
    if (!next || preview || !element || next.value !== value) return;
    pending.current = null;
    const active = document.activeElement;
    if (!active || active === document.body || container.current?.contains(active)) element.focus({ preventScroll: true });
    element.setSelectionRange(next.start, next.end, next.direction);
    element.scrollTop = next.top; element.scrollLeft = next.left;
    selection.current = next;
  }, [value, preview]);
  const pushUndo = (snapshot: Selection) => {
    undo.current.push(snapshot);
    if (undo.current.length > HISTORY_LIMIT) undo.current.shift();
    redo.current = [];
  };
  const commit = (next: Selection) => {
    lastValue.current = next.value; pending.current = next; selection.current = next;
    onChange(next.value); setMessage(""); setCopied(false); refresh();
  };
  const apply = (format: RebuttalFormat, chosenColor = color) => {
    if (readOnly || composing.current) return;
    const previous = remember();
    try {
      const edit = formatRebuttal(previous.value, format, previous.start, previous.end, chosenColor, locale);
      if (edit.value === previous.value) return;
      pushUndo(previous); typedAt.current = 0;
      commit({ ...previous, value: edit.value, start: edit.start, end: edit.end });
      setPreview(false);
    } catch (error) {
      setMessage(error instanceof Error && error.message === "short-text"
        ? t("颜色、字号和方框请选择单行短文字（最多 200 字）。", "Select a short, single-line label for color, size or a box (up to 200 characters).")
        : t("内容超出编辑范围，未修改原文。", "This edit exceeds the supported range. Your text is unchanged."));
    }
  };
  const moveHistory = (direction: "undo" | "redo") => {
    if (readOnly || composing.current) return;
    const source = direction === "undo" ? undo : redo, destination = direction === "undo" ? redo : undo;
    const next = source.current.pop();
    if (!next) return;
    destination.current.push(remember()); typedAt.current = 0;
    commit(next); setPreview(false);
  };
  const handleKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    beforeInput.current = remember();
    if (readOnly || composing.current || event.nativeEvent.isComposing || event.keyCode === 229) return;
    const action = rebuttalShortcut({ ...event, isComposing: event.nativeEvent.isComposing });
    if (!action) return;
    event.preventDefault();
    if (action === "undo" || action === "redo") moveHistory(action); else apply(action);
  };
  const toolbarKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!(event.target instanceof HTMLButtonElement) || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    const items = [...event.currentTarget.querySelectorAll<HTMLButtonElement | HTMLSelectElement>("button:not(:disabled),select:not(:disabled)")];
    const index = items.indexOf(event.target);
    if (index < 0 || !items.length) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? items.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + items.length) % items.length;
    items[next].focus();
  };
  const changeMode = (next: boolean) => {
    if (next) remember(); else pending.current = selection.current;
    setPreview(next);
  };
  const copy = async () => {
    try { await navigator.clipboard.writeText(toOpenReviewMarkdown(value)); setCopied(true); setMessage(""); }
    catch { setMessage(t("复制失败，请在 Markdown 中选择原文复制。", "Copy failed. Select and copy the Markdown source.")); }
  };
  const tool = (format: RebuttalFormat, label: string, icon: ReactNode, shortcut?: string) => <button
    type="button" key={format} aria-label={label} title={shortcut ? `${label} (${shortcut})` : label}
    onMouseDown={event => event.preventDefault()} onClick={() => apply(format)}
  >{icon}</button>;
  const mod = "Ctrl/⌘";
  return <div className={`rbe-editor${readOnly ? " rbe-readonly" : ""}`} ref={container}>
    <div className="rbe-head">
      <div className="rbe-modes" role="tablist" aria-label={t("回复编辑模式", "Response editor mode")}>
        <button type="button" id={`${id}-write`} role="tab" aria-selected={!preview} aria-controls={`${id}-source`} onClick={() => changeMode(false)}>{readOnly ? "Markdown" : t("编辑", "Write")}</button>
        <button type="button" id={`${id}-preview`} role="tab" aria-selected={preview} aria-controls={`${id}-rendered`} onClick={() => changeMode(true)}>{t("预览", "Preview")}</button>
      </div>
      <button type="button" className="rbe-copy" onClick={copy} disabled={!value} title={t("复制 OpenReview Markdown", "Copy OpenReview Markdown")}>{copied ? <Check size={15} /> : <Copy size={15} />}{copied ? t("已复制", "Copied") : t("复制", "Copy")}</button>
    </div>
    {!readOnly && <div className="rbe-toolbar" role="toolbar" aria-label={t("文字格式", "Text formatting")} onKeyDown={toolbarKey} hidden={preview}>
      <div className="rbe-tool-group">
        {tool("bold", t("粗体", "Bold"), <Bold size={16} />, `${mod}+B`)}
        {tool("italic", t("斜体", "Italic"), <Italic size={16} />, `${mod}+I`)}
        {tool("h1", t("一级标题", "Heading 1"), <Heading1 size={18} />, `${mod}+Shift+1`)}
        {tool("h2", t("二级标题", "Heading 2"), <Heading2 size={18} />, `${mod}+Shift+2`)}
      </div>
      <div className="rbe-tool-group">
        {tool("color", t("文字颜色", "Text color"), <Palette size={16} style={{ color }} />, `${mod}+Shift+C`)}
        <select value="" aria-label={t("选择文字颜色", "Choose text color")} title={t("选择文字颜色", "Choose text color")} onChange={event => { const next = event.target.value; setColor(next); apply("color", next); }}>
          <option value="" disabled>{t("颜色", "Color")}</option>{REBUTTAL_COLORS.map(item => <option key={item.value} value={item.value}>{zh ? item.zh : item.en}</option>)}
        </select>
        <select value="" aria-label={t("字号", "Text size")} title={`${t("字号", "Text size")} (${mod}+Shift+− / +)`} onChange={event => apply(event.target.value as "small" | "normal" | "large")}>
          <option value="" disabled>{t("字号", "Size")}</option><option value="small">{t("小", "Small")}</option><option value="normal">{t("正常", "Normal")}</option><option value="large">{t("大", "Large")}</option>
        </select>
      </div>
      <div className="rbe-tool-group">
        {tool("quote", t("引用", "Quote"), <Quote size={16} />, `${mod}+Shift+Q`)}
        {tool("hr", t("分隔线", "Horizontal rule"), <Minus size={16} />, `${mod}+Shift+H`)}
        {tool("boxed", t("文字方框", "Boxed text"), <Square size={16} />, `${mod}+Shift+B`)}
        {tool("list", t("项目列表", "Bullet list"), <List size={17} />)}
        {tool("ordered", t("编号列表", "Numbered list"), <ListOrdered size={17} />)}
      </div>
      <div className="rbe-tool-group">
        {tool("link", t("链接", "Link"), <Link2 size={16} />, `${mod}+K`)}
        {tool("code", t("代码", "Code"), <Code2 size={17} />)}
        {tool("table", t("表格", "Table"), <Table2 size={16} />)}
        {tool("math", t("公式", "Math"), <Sigma size={17} />)}
      </div>
      <div className="rbe-tool-group rbe-history">
        <button type="button" aria-label={t("撤销", "Undo")} title={`${t("撤销", "Undo")} (${mod}+Z)`} disabled={!undo.current.length} onMouseDown={event => event.preventDefault()} onClick={() => moveHistory("undo")}><Undo2 size={17} /></button>
        <button type="button" aria-label={t("重做", "Redo")} title={`${t("重做", "Redo")} (${mod}+Shift+Z)`} disabled={!redo.current.length} onMouseDown={event => event.preventDefault()} onClick={() => moveHistory("redo")}><Redo2 size={17} /></button>
      </div>
    </div>}
    <div id={`${id}-source`} role="tabpanel" aria-labelledby={`${id}-write`} hidden={preview}>
      <textarea ref={area} value={value} readOnly={readOnly} maxLength={200000} aria-label={t("回复 Markdown", "Response Markdown")} spellCheck
        placeholder={t("逐条回复审稿人的问题…", "Respond to the reviewer's concern…")}
        onSelect={remember} onScroll={remember} onBlur={remember} onKeyDown={handleKey} onBeforeInput={() => { beforeInput.current = remember(); }}
        onCompositionStart={() => { composing.current = true; beforeInput.current = remember(); }} onCompositionEnd={() => { composing.current = false; }}
        onChange={event => {
          if (readOnly) return;
          const element = event.currentTarget;
          const previous = beforeInput.current?.value === lastValue.current ? beforeInput.current : { ...selection.current, value: lastValue.current };
          const now = Date.now();
          if (!typedAt.current || now - typedAt.current > 700 || redo.current.length) pushUndo(previous);
          typedAt.current = now; beforeInput.current = null;
          const next = { value: element.value, start: element.selectionStart, end: element.selectionEnd, direction: element.selectionDirection, top: element.scrollTop, left: element.scrollLeft };
          lastValue.current = next.value; selection.current = next;
          onChange(next.value); setMessage(""); setCopied(false); refresh();
        }} />
    </div>
    <div id={`${id}-rendered`} className="rbe-rendered" role="tabpanel" aria-labelledby={`${id}-preview`} hidden={!preview}>
      {preview && (value.trim() ? <RebuttalPreview text={value} /> : <p className="rbe-empty">{t("暂无回复", "No response yet")}</p>)}
    </div>
    {message && <p className="rbe-message" role="status">{message}</p>}
    {!readOnly && actions && <div className="rbe-actions">{actions}</div>}
    {footer && <div className="rbe-footer">{footer}</div>}
  </div>;
}
