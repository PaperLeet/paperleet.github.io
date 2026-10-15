import { useEffect, useId, useState } from 'react';
import { Link2, RefreshCw } from 'lucide-react';
import { useI18n } from './i18n';
import { socialApi } from './social';
import { canonicalArxivUrl, readPaperPreview, type PaperPreview } from './communityPapers';
import PaperPreviewCard from './PaperPreviewCard';

export default function PaperLinkField({ value, onChange, owner, available, required = false, disabled = false, initialPaper, onUseTitle }: {
  value: string; onChange: (value: string) => void; owner: string; available: boolean; required?: boolean; disabled?: boolean; initialPaper?: unknown; onUseTitle?: (title: string) => void;
}) {
  const { t } = useI18n(), hint = useId();
  const url = canonicalArxivUrl(value);
  const [result, setResult] = useState<PaperPreview | undefined>(() => readPaperPreview(initialPaper, url));
  const [request, setRequest] = useState<{ url: string; state: 'loading' | 'failed' | 'done' }>();
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!url || !available || !owner || disabled) return;
    const controller = new AbortController();
    let live = true;
    const timer = window.setTimeout(() => {
      setRequest({ url, state: 'loading' });
      void socialApi<{ paper: unknown }>('community/papers/preview', { url, expected_user_id: owner }, controller.signal).then(data => {
        if (!live) return;
        const paper = readPaperPreview(data.paper, url);
        if (!paper) throw Error('Invalid paper preview');
        setResult(paper);
        setRequest({ url, state: paper.status === 'ready' ? 'done' : 'failed' });
      }).catch(() => { if (live) setRequest({ url, state: 'failed' }); });
    }, 800);
    return () => { live = false; clearTimeout(timer); controller.abort(); };
  }, [url, available, owner, retry, disabled]);
  const paper = result?.url === url ? result : undefined;
  const state = request && request.url === url ? request.state : undefined;
  const invalid = !!value.trim() && !url;
  return <div className="paper-link-field">
    <label><span><Link2 size={14} />{t('arXiv 论文链接', 'arXiv paper link')}{!required && <small>{t('（可选）', ' (optional)')}</small>}</span>
      <input type="url" value={value} onChange={event => onChange(event.target.value)} required={required} disabled={disabled || !available} maxLength={2048} placeholder="https://arxiv.org/abs/1706.03762" aria-invalid={invalid || undefined} aria-describedby={hint} />
    </label>
    <p id={hint} className={invalid ? 'paper-link-error' : 'paper-link-hint'}>{!available
      ? t('论文预览暂不可用，可以先在正文中分享链接。', 'Paper previews are unavailable. You can share the link in your post for now.')
      : invalid ? t('请粘贴有效的 arxiv.org 摘要页或 PDF 链接。', 'Paste a valid arxiv.org abstract page or PDF link.')
      : t('粘贴摘要页或 PDF 链接，自动读取标题、作者和论文预览。', 'Paste an abstract page or PDF link to load its title, authors, and preview.')}</p>
    {available && url && <>
      <div className="paper-link-status" role="status">{state === 'loading' ? <><span className="spinner" />{t('正在解析论文…', 'Loading paper preview…')}</> : state === 'failed' ? <>{t('预览暂时不可用，论文链接仍会随帖子保存。', 'Preview unavailable. The paper link will still be saved with your post.')}<button type="button" disabled={disabled} onClick={() => setRetry(value => value + 1)}><RefreshCw size={12} />{t('重试', 'Retry')}</button></> : paper ? t('论文已关联', 'Paper attached') : t('正在准备论文预览…', 'Preparing paper preview…')}</div>
      {(paper || state === 'failed') && <PaperPreviewCard url={url} paper={paper} />}
      {paper?.title && onUseTitle && <button type="button" className="text-button paper-use-title" disabled={disabled} onClick={() => onUseTitle(paper.title!)}>{t('使用论文标题作为帖子标题', 'Use paper title for this discussion')}</button>}
    </>}
  </div>;
}
