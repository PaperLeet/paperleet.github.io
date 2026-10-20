import AppLink from './AppLink';
import {useId,useRef,type ReactNode} from "react";
import ReadingAppearance from './ReadingAppearance';
import {webStatementReadme,type StatementExecution} from './webStatementReadme';
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { FlaskConical } from "lucide-react";
import { useI18n } from "./i18n";
import { Highlight } from "./SyntaxHighlight";
import { readingOutline } from './readingOutline';
import "katex/dist/katex.min.css";
import './reading.css';

export function Prose({
  content,
  problemId,
  onViz,
  statementExecution,
}: {
  content: string;
  problemId?: string;
  onViz?: () => void;
  statementExecution?: StatementExecution;
}) {
  const { t } = useI18n();
  const readingRoot=useRef<HTMLDivElement>(null),readingId=useId();
  const displayed=statementExecution?webStatementReadme(content,statementExecution,t):content;
  const processed = displayed.replace(
    /<viz\s+id=["']([^"']+)["']\s*\/>/g,
    `[${t("打开交互实验", "Open interactive lab")}](#viz-$1)`,
  );
  function Link({
    href,
    children,
  }: {
    href?: string;
    children?: ReactNode;
  }) {
    if (href?.startsWith("#viz-"))
      return (
        <button className="text-button inline-lab-link" onClick={onViz}>
          <FlaskConical size={13} />
          {children}
        </button>
      );
    if (href && /^https?:/.test(href))
      return (
        <AppLink href={href} target="_blank" rel="noreferrer">
          {children}
        </AppLink>
      );
    if (href?.includes("README.md") && problemId) {
      const resolved = new URL(
        href,
        `https://local/problems/${problemId}/README.md`,
      ).pathname;
      const match = /\/problems\/([a-z_]+)\/([^/]+)\/README(?:\.en)?\.md$/.exec(
        resolved,
      );
      if (match) return <AppLink href={`#/${match[1]}/${match[2]}`}>{children}</AppLink>;
    }
    return <AppLink href={href}>{children}</AppLink>;
  }
  return (
    <div ref={readingRoot} id={readingId} className="prose" onClick={event => {
      if (!(event.target instanceof Element)) return;
      const button = event.target.closest<HTMLButtonElement>('button[data-reading-section]');
      const menu = event.currentTarget.querySelector<HTMLDetailsElement>('.reading-outline');
      if (!button) {
        if (menu && !menu.contains(event.target)) menu.open = false;
        return;
      }
      const index = Number(button.dataset.readingSection);
      if (!Number.isSafeInteger(index) || index < 0) return;
      const heading = event.currentTarget.querySelector<HTMLElement>(`[data-reading-anchor="${index}"]`);
      if (!heading) return;
      if (menu) menu.open = false;
      heading.focus({preventScroll:true});
      heading.scrollIntoView({block:'start', behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
    }} onKeyDown={event => {
      if (event.key !== 'Escape' || event.defaultPrevented || event.repeat || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || event.nativeEvent.isComposing || event.keyCode === 229) return;
      const menu = event.currentTarget.querySelector<HTMLDetailsElement>('.reading-outline[open]');
      if (!menu) return;
      event.preventDefault(); event.stopPropagation();
      menu.open = false;
      menu.querySelector('summary')?.focus({preventScroll:true});
    }} onBlur={event => {
      const menu = event.currentTarget.querySelector<HTMLDetailsElement>('.reading-outline[open]');
      if (menu && event.relatedTarget instanceof Node && !menu.contains(event.relatedTarget)) menu.open = false;
    }}>
      {displayed.length>=1200&&<ReadingAppearance target={readingRoot} controlsId={readingId} scope={displayed}/>}
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[readingOutline, {title:t('本文导航', 'On this page'), sectionLabel:t('节', 'sections')}], rehypeKatex]}
        components={{
          a: Link,
          pre: ({ children }) => <>{children}</>,
          code: ({ className, children, ...props }) => {
            const lang = /language-(\w+)/.exec(className || "")?.[1];
            return lang ? (
              <Highlight
                code={String(children).replace(/\n$/, "")}
                lang={lang}
              />
            ) : (
              <code {...props} className={className}>
                {children}
              </code>
            );
          },
        }}
      >
        {processed}
      </Markdown>
    </div>
  );
}
