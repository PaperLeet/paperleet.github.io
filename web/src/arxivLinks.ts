/** Strict arXiv links shared by search and community publication. */
export type ArxivLink = { arxivId: string; url: string; pdfUrl: string };
const arxivId = /^(?:[0-9]{2}(?:0[1-9]|1[0-2])\.[0-9]{4,5}|[a-z][a-z0-9.-]{0,39}\/[0-9]{2}(?:0[1-9]|1[0-2])[0-9]{3})(?:v[1-9][0-9]{0,3})?$/;

export function parseArxivUrl(value: unknown): ArxivLink | undefined {
  if (typeof value !== 'string') return;
  const input = value.trim();
  // Check the original input too: URL parsers silently discard embedded controls.
  if (!input || input.length > 512 || /[\u0000-\u001f\u007f\\%?#]/.test(value)) return;
  const match = /^(https?):\/\/((?:www\.|export\.)?arxiv\.org)(\/.*)$/i.exec(input);
  if (!match) return;
  // A strict authority match excludes credentials and every explicit port.
  const path = /^\/(abs|pdf)\/(.+)$/.exec(match[3]);
  if (!path) return;
  const id = path[1] === 'pdf' ? path[2].replace(/\.pdf$/, '') : path[2];
  if (!arxivId.test(id)) return;
  return { arxivId: id, url: `https://arxiv.org/abs/${id}`, pdfUrl: `https://arxiv.org/pdf/${id}` };
}

export function canonicalArxivUrl(value: unknown): string | undefined {
  return parseArxivUrl(value)?.url;
}

