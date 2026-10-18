/** Only these arXiv links may be attached to community posts. */
import {parseArxivUrl,canonicalArxivUrl} from './arxivLinks';
export {parseArxivUrl,canonicalArxivUrl} from './arxivLinks';
export type {ArxivLink} from './arxivLinks';
export type PaperPreview = {
  arxiv_id: string;
  url: string;
  pdf_url: string;
  title?: string;
  authors?: string[];
  abstract?: string;
  published?: string;
  thumbnail_url?: string;
  status: 'ready' | 'unavailable';
};

const thumbnailPath = /^\/api\/community\/papers\/thumbnails\/[a-f0-9]{64}\.png$/;

/** Validate API metadata before it becomes a clickable URL or image source. */
export function readPaperPreview(value: unknown, expectedUrl?: string | null): PaperPreview | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value) || expectedUrl === null) return;
  const data = value as Record<string, unknown>;
  const link = parseArxivUrl(data.url);
  if (!link || data.url !== link.url || data.pdf_url !== link.pdfUrl || data.arxiv_id !== link.arxivId ||
      (expectedUrl !== undefined && canonicalArxivUrl(expectedUrl) !== link.url) ||
      (data.status !== 'ready' && data.status !== 'unavailable')) return;
  const preview: PaperPreview = {
    arxiv_id: link.arxivId, url: link.url, pdf_url: link.pdfUrl, status: data.status,
  };
  for (const [key, maxLength] of [['title', 2000], ['abstract', 30000], ['published', 80]] as const) {
    if (data[key] === undefined || data[key] === null) continue;
    if (typeof data[key] !== 'string' || data[key].length > maxLength) return;
    if (data[key].trim()) preview[key] = data[key];
  }
  if (data.authors !== undefined && data.authors !== null) {
    if (!Array.isArray(data.authors) || data.authors.length > 2000 ||
        !data.authors.every(author => typeof author === 'string' && author.length <= 300)) return;
    preview.authors = data.authors.filter(author => author.trim());
  }
  // An unavailable thumbnail must not hide an otherwise useful paper link.
  if (typeof data.thumbnail_url === 'string' && thumbnailPath.test(data.thumbnail_url)) {
    preview.thumbnail_url = data.thumbnail_url;
  }
  return preview;
}
