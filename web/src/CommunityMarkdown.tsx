import { Children, isValidElement, memo, useState } from 'react';
import type { ReactNode } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Check, Copy } from 'lucide-react';
import { Highlight } from './SyntaxHighlight';
import { useI18n } from './i18n';
import 'katex/dist/katex.min.css';

export function CommunityCodeBlock({ code, language = 'text' }: { code: string; language?: string }) {
  const { t } = useI18n();
  const [feedback, setFeedback] = useState<{ code: string; state: 'copying' | 'copied' | 'failed' }>();
  const state = feedback?.code === code ? feedback.state : undefined;
  async function copy() {
    setFeedback({ code, state: 'copying' });
    try {
      await navigator.clipboard.writeText(code);
      setFeedback({ code, state: 'copied' });
    } catch { setFeedback({ code, state: 'failed' }); }
  }
  return <div className="community-code-block">
    <div className="community-code-header">
      <span>{language}</span>
      <button type="button" disabled={state === 'copying'} onClick={() => void copy()} aria-label={t(`复制 ${language} 代码`, `Copy ${language} code`)}>
        {state === 'copied' ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />}
        {state === 'copied' ? t('已复制', 'Copied') : t('复制代码', 'Copy code')}
      </button>
    </div>
    <Highlight code={code} lang={language} />
    {state === 'copied' && <span className="sr-only" role="status">{t('代码已复制。', 'Code copied.')}</span>}
    {state === 'failed' && <p className="community-copy-error" role="alert">{t('无法自动复制，请选择代码后手动复制。', 'Could not copy automatically. Select the code and copy it manually.')}</p>}
  </div>;
}

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
  a: ({ href, children }: { href?: string; children?: ReactNode }) => <a href={href} target="_blank" rel="noopener noreferrer nofollow">{children}</a>,
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

const CommunityMarkdown = memo(function CommunityMarkdown({ body }: { body: string }) {
  return <div className="prose community-prose">
    <Markdown remarkPlugins={remarkPlugins} rehypePlugins={[[rehypeKatex, katexOptions]]} components={components}>{body}</Markdown>
  </div>;
});
export default CommunityMarkdown;
