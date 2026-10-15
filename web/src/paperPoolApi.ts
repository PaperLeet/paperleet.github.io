import { apiFetch } from './apiClient';
import { currentQuotaMessage } from './publicationQuota';

/** Accept arXiv abstract and PDF URLs; the server remains the duplicate authority. */
export function arxivPaperUrl(value: string): string | undefined {
  try {
    if (value.length > 512) return;
    const raw = value.trim();
    if (!raw || /[\x00-\x20\x7f\\%?#]/.test(raw)) return;
    const authority = /^[a-z]+:\/\/([^/]+)/i.exec(raw)?.[1].toLowerCase();
    const url = new URL(raw);
    if (!['https:', 'http:'].includes(url.protocol) || !authority || !['arxiv.org', 'www.arxiv.org', 'export.arxiv.org'].includes(authority)) return;
    const path = /^\/(abs|pdf)\/(.+)$/.exec(url.pathname);
    if (!path) return;
    const identifier = path[1] === 'pdf' ? path[2].replace(/\.pdf$/, '') : path[2];
    const match = /^([0-9]{2}(?:0[1-9]|1[0-2])\.[0-9]{4,5}|[a-z][a-z0-9.\-]{0,39}\/[0-9]{2}(?:0[1-9]|1[0-2])[0-9]{3})(?:v[1-9][0-9]{0,3})?$/.exec(identifier);
    if (match) return `https://arxiv.org/abs/${identifier}`;
  } catch { /* A partial draft is not a submission. */ }
}

export async function poolApi<T>(path = '', body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await apiFetch(`/api/paper-pool${path}`, {
    cache: 'no-store', signal,
    ...(body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}),
  });
  let data;
  try { data = await response.json(); }
  catch { throw Error('The paper pool service is unavailable. Please try again later.'); }
  if (!response.ok) throw Error(currentQuotaMessage(data) || data.error || `Request failed (${response.status})`);
  return data as T;
}

export function poolHref(id?: string) {
  return id ? `#/submit?proposal=${encodeURIComponent(id)}` : '#/submit?view=pool';
}

export function poolRoute(hash: string) {
  const query = new URLSearchParams(hash.split('?')[1] || '');
  const id = query.get('proposal');
  return {id: id && /^[a-f0-9]{32}$/.test(id) ? id : undefined, pool: query.get('view') === 'pool'};
}
