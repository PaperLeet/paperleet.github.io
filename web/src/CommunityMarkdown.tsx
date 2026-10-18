import AppLink from './AppLink';
import { Children, createElement, isValidElement, memo, useMemo } from 'react';
import type { ReactNode } from 'react';
import Markdown from 'react-markdown';
import type {Components} from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import CommunityCodeBlock from './CommunityCodeBlock';
export {default as CommunityCodeBlock} from './CommunityCodeBlock';
import { useI18n } from './i18n';
import {headingAnchor} from './markdownOutline';
import {usePracticeMentions} from './PracticeMentions';
import {transformPracticeMentions} from './practiceMentionMatcher';
import {withLibraryContext} from './libraryNavigation';
import 'katex/dist/katex.min.css';

export function CommunityFormatHelp() {
  const { t } = useI18n();
  return <details className="community-format-help">
    <summary>{t('公式与代码写法', 'Math and code formatting')}</summary>
    <p>{t('行内公式用单个美元符号包裹，例如 ', 'Wrap inline math in single dollar signs, for example ')}<code>{'$\\alpha/r$'}</code>{t('。', '.')}</p>
    <p>{t('独立公式的 $$ 各占一行；代码块使用三个反引号，并注明语言。', 'Put each display-math $$ on its own line. Use triple backticks and a language for code blocks.')}</p>
    <pre>{'$$\n\\Delta W = \\frac{\\alpha}{r}BA\n$$\n\n```python\ndelta = (alpha / rank) * (b @ a)\n```'}</pre>
  </details>;
}

const components = {
  a: ({ href, children,className }: { href?: string; children?: ReactNode;className?:string }) => className==='practice-mention-link'
    ? <AppLink href={href} className={className} data-reading-link="practice">{children}</AppLink>
    : <AppLink href={href} target="_blank" rel="noopener noreferrer nofollow">{children}</AppLink>,
  img: ({ alt }: { alt?: string }) => <span className="community-image-alt">{alt ? `[${alt}]` : '[image]'}</span>,
  pre: ({ children }: { children?: ReactNode }) => {
    const child = Children.toArray(children)[0];
    if (!isValidElement<{ className?: string; children?: ReactNode }>(child)) return <pre>{children}</pre>;
    const language = /language-(\S+)/.exec(child.props.className || '')?.[1] || 'text';
    return <CommunityCodeBlock code={String(child.props.children ?? '').replace(/\n$/, '')} language={language} />;
  },
};
const remarkPlugins = [remarkGfm, remarkMath];
// Keep macros local to each render; public math cannot inject HTML or fetch images.
const katexOptions = { trust: false, strict: 'ignore' as const, maxExpand: 100, maxSize: 20 };

const CommunityMarkdown = memo(function CommunityMarkdown({ body,headingScope }: { body: string;headingScope?:string }) {
  const mentions=usePracticeMentions();
  const plugins=useMemo(()=>mentions?[...remarkPlugins,()=> (tree:Parameters<typeof transformPracticeMentions>[0])=>{transformPracticeMentions(tree,mentions.index,topic=>withLibraryContext(topic.id,mentions.returnTo));}]:remarkPlugins,[mentions]);
  const readingComponents=useMemo<Components>(()=>{
    if(!headingScope)return components;
    const result:Components={...components};
    for(const tag of ['h1','h2','h3','h4','h5','h6'] as const)result[tag]=({children,node,...props})=>createElement(tag,{...props,id:headingAnchor(headingScope,node?.position?.start.offset),tabIndex:-1,className:'discussion-section-heading'},children);
    return result;
  },[headingScope]);
  return <div className="prose community-prose">
    <Markdown remarkPlugins={plugins} rehypePlugins={[[rehypeKatex, katexOptions]]} components={readingComponents}>{body}</Markdown>
  </div>;
});
export default CommunityMarkdown;
