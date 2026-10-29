import AppLink from './AppLink';
import { useState } from 'react';
import { ExternalLink, FileText } from 'lucide-react';
import { apiUrl } from './apiClient';
import { useI18n } from './i18n';
import { parseArxivUrl, readPaperPreview } from './communityPapers';
import './community-papers.css';

/** `thumbnail` is a cover bundled with the site (roadmap papers), used only when the API preview has no thumbnail.
 *  `linksUnderCover` stacks the cover and its Paper page / Open PDF links in one side column instead of ending the text with the links. */
export default function PaperPreviewCard({ url, paper, compact = false, thumbnail, linksUnderCover = false }: { url?: string | null; paper?: unknown; compact?: boolean; thumbnail?: string; linksUnderCover?: boolean }) {
  const { t, locale } = useI18n();
  const [failedImage, setFailedImage] = useState<string>();
  const link = parseArxivUrl(url);
  if (!link) return null;
  const preview = readPaperPreview(paper, link.url);
  const image = preview?.thumbnail_url ? apiUrl(preview.thumbnail_url) : thumbnail;
  const date = preview?.published ? new Date(preview.published) : undefined;
  const cover = <AppLink className="paper-preview-cover" href={link.pdfUrl} target="_blank" rel="noreferrer" aria-label={t('打开论文 PDF', 'Open paper PDF')}>
    {image && failedImage !== image
      ? <img src={image} alt={t('论文 PDF 首页', 'First page of the paper PDF')} loading="lazy" decoding="async" referrerPolicy="no-referrer" onError={() => setFailedImage(image)} />
      : <span><FileText size={30} /><small>PDF</small></span>}
  </AppLink>;
  const links = <div className="paper-preview-links"><AppLink href={link.url} target="_blank" rel="noreferrer">{t('论文页面', 'Paper page')}<ExternalLink size={12} /></AppLink><AppLink href={link.pdfUrl} target="_blank" rel="noreferrer">{t('打开 PDF', 'Open PDF')}<ExternalLink size={12} /></AppLink></div>;
  return <aside className={`paper-preview-card${compact ? ' compact' : ''}`} aria-label={t('论文预览', 'Paper preview')}>
    {linksUnderCover ? <div className="paper-preview-side">{cover}{links}</div> : cover}
    <div className="paper-preview-copy">
      <div className="paper-preview-source"><span>arXiv:{link.arxivId}</span>{date && Number.isFinite(date.getTime()) && <time dateTime={preview!.published}>{date.toLocaleDateString(locale === 'zh' ? 'zh-CN' : 'en-US', { year: 'numeric', month: 'short', day: 'numeric' })}</time>}</div>
      <AppLink className="paper-preview-title" href={link.url} target="_blank" rel="noreferrer">{preview?.title || t('查看 arXiv 论文', 'View paper on arXiv')}</AppLink>
      {!!preview?.authors?.length && <p className="paper-preview-authors">{preview.authors.slice(0, 5).join(', ')}{preview.authors.length > 5 ? ' et al.' : ''}</p>}
      {!compact && preview?.abstract && <p className="paper-preview-abstract">{preview.abstract}</p>}
      {!preview?.title && <p className="paper-preview-unavailable">{t('暂时无法读取论文信息，仍可通过链接查看原文。', 'Paper details are unavailable. You can still open the original paper.')}</p>}
      {!linksUnderCover && links}
    </div>
  </aside>;
}
