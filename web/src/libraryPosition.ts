import { libraryHref, readLibraryRoute } from './libraryNavigation';

type Position = { y: number; anchor?: { id: string; top: number } };
const key = 'paperleet.library.positions.v1', limit = 8;
const memory = new Map<string, Position>();
let hydrated = false;
function canonical(href: string) {
  const view = readLibraryRoute(href);
  return view && libraryHref(view.collectionId, view.filters);
}
function valid(value: unknown): value is Position {
  if (!value || typeof value !== 'object') return false;
  const row = value as Position;
  return Number.isFinite(row.y) && row.y >= 0 && row.y <= 1e7 && (!row.anchor || typeof row.anchor.id === 'string' && /^[a-z0-9_-]{1,80}(?:\/[a-z0-9_-]{1,80})?$/.test(row.anchor.id)
    && Number.isFinite(row.anchor.top) && Math.abs(row.anchor.top) <= 1e5);
}
export function rememberLibraryPosition(href: string, value: Position) {
  hydrate();
  const name = canonical(href);
  if (!name || !valid(value)) return;
  memory.delete(name); memory.set(name, value);
  while (memory.size > limit) memory.delete(memory.keys().next().value!);
  try { sessionStorage.setItem(key, JSON.stringify([...memory])); } catch { /* Filtering and return links also work without browser storage. */ }
}
function hydrate() {
  if (hydrated) return;
  hydrated = true;
  try {
    const text = sessionStorage.getItem(key);
    if (!text || text.length > 65536) return;
    const rows: unknown = JSON.parse(text);
    if (!Array.isArray(rows)) return;
    for (const row of rows.slice(-limit)) if (Array.isArray(row) && row.length === 2 && typeof row[0] === 'string' && valid(row[1])) {
      const name = canonical(row[0]);
      if (name) memory.set(name, row[1]);
    }
  } catch { /* Corrupt or blocked storage cannot break the library. */ }
}
export function readLibraryPosition(href: string): Position | undefined {
  hydrate();
  const name = canonical(href);
  return name ? memory.get(name) : undefined;
}

/** Optional enhancement; install only while this exact library view is mounted. */
export function watchLibraryPosition(href: string, restore: boolean) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const here = () => canonical(location.hash) === href;
  const rows = () => Array.from(document.querySelectorAll<HTMLElement>('[data-paper-id],[data-module-id]'));
  const rowId = (element?: HTMLElement) => element?.dataset.paperId || element?.dataset.moduleId;
  const position = restore && readLibraryPosition(href);
  let anchorId = position && position.anchor?.id;
  const visible = (element: HTMLElement) => { const rect = element.getBoundingClientRect(); return rect.bottom > 100 && rect.top < innerHeight; };
  function capture(preferred?: HTMLElement) {
    if (!here()) return;
    const candidates = rows(), retained = candidates.find(element => rowId(element) === anchorId && visible(element));
    // Keep the selected row as the anchor until the reader scrolls away.
    const row = rowId(preferred) ? preferred : retained || candidates.find(visible);
    anchorId = rowId(row);
    rememberLibraryPosition(href, { y: Math.max(0, scrollY), ...(row ? { anchor: { id: rowId(row)!, top: row.getBoundingClientRect().top } } : {}) });
  }
  const frame = position ? requestAnimationFrame(() => {
    if (!here()) return;
    const row = position.anchor && rows().find(element => rowId(element) === position.anchor!.id);
    scrollTo(0, Math.max(0, row && position.anchor ? scrollY + row.getBoundingClientRect().top - position.anchor.top : position.y));
  }) : undefined;
  const scroll = () => { clearTimeout(timer); timer = setTimeout(() => capture(), 120); };
  const hide = () => capture();
  const click = (event: MouseEvent) => { if (event.target instanceof Element) {
    const target = event.target.closest<HTMLElement>('a[href],button');
    if (target) capture(target);
  } };
  window.addEventListener('scroll', scroll, { passive: true });
  window.addEventListener('pagehide', hide);
  document.addEventListener('click', click, true);
  return () => { clearTimeout(timer); if (frame !== undefined) cancelAnimationFrame(frame); window.removeEventListener('scroll', scroll); window.removeEventListener('pagehide', hide); document.removeEventListener('click', click, true); };
}
