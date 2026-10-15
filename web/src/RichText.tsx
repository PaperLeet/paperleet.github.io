import type { ReactNode } from "react";
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
}: {
  content: string;
  problemId?: string;
  onViz?: () => void;
}) {
  const { t } = useI18n();
  const processed = content.replace(
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
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    if (href?.includes("README.md") && problemId) {
      const resolved = new URL(
        href,
        `https://local/problems/${problemId}/README.md`,
      ).pathname;
      const match = /\/problems\/([a-z_]+)\/([^/]+)\/README(?:\.en)?\.md$/.exec(
        resolved,
      );
      if (match) return <a href={`#/${match[1]}/${match[2]}`}>{children}</a>;
    }
    return <a href={href}>{children}</a>;
  }
  return (
    <div className="prose" onClick={event => {
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
      if (event.key !== 'Escape') return;
      const menu = event.currentTarget.querySelector<HTMLDetailsElement>('.reading-outline[open]');
      if (!menu) return;
      event.preventDefault(); event.stopPropagation();
      menu.open = false;
      menu.querySelector('summary')?.focus({preventScroll:true});
    }} onBlur={event => {
      const menu = event.currentTarget.querySelector<HTMLDetailsElement>('.reading-outline[open]');
      if (menu && event.relatedTarget instanceof Node && !menu.contains(event.relatedTarget)) menu.open = false;
    }}>
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
