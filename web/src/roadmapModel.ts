import { parseArxivUrl } from './arxivLinks';
import type { CollectionRoadmap } from './collectionRoadmaps';
import type { Paper } from './types';

/** Grid geometry in px; gutter holds year labels, top holds the start tag. */
export const ROADMAP = { colWidth: 150, rowHeight: 86, nodeWidth: 126, nodeHeight: 50, gutter: 58, top: 34, pad: 16 };

type Cell = { row: number; col: number };
type Pt = { x: number; y: number };

export function nodeBox(node: Cell) {
  const { colWidth, rowHeight, nodeWidth, nodeHeight, gutter, top } = ROADMAP;
  return { x: gutter + node.col * colWidth + (colWidth - nodeWidth) / 2, y: top + node.row * rowHeight + (rowHeight - nodeHeight) / 2, width: nodeWidth, height: nodeHeight };
}

export function roadmapSize(r: CollectionRoadmap) {
  const maxCol = Math.max(0, ...r.nodes.map(n => n.col)), maxRow = Math.max(0, ...r.nodes.map(n => n.row));
  return { width: ROADMAP.gutter + (maxCol + 1) * ROADMAP.colWidth + ROADMAP.pad, height: ROADMAP.top + (maxRow + 1) * ROADMAP.rowHeight + ROADMAP.pad };
}

/** Start tag size in px, with its gap above the node (no incoming arrow) or `side` px left of it (with one). */
export const START_TAG = { width: 96, height: 22, gap: 6, side: 8 };

/** Tag side, CSS anchor (`left`/`top`, positioned with translate) and estimated box of the start node's tag. */
export function startTag(r: CollectionRoadmap) {
  const n = r.nodes.find(x => x.start)!, b = nodeBox(n), { width, height, gap, side } = START_TAG;
  if (r.edges.some(e => e[1] === n.id)) return { side: 'left' as const, left: b.x - side, top: b.y + b.height / 2, box: { x: b.x - side - width, y: b.y + (b.height - height) / 2, width, height } };
  return { side: 'above' as const, left: b.x + b.width / 2, top: b.y, box: { x: b.x + (b.width - width) / 2, y: b.y - gap - height, width, height } };
}

/** Bottom-center of `from`, top-center of `to`, vertical control points at half the gap. */
function ends(from: Cell, to: Cell) {
  const a = nodeBox(from), b = nodeBox(to);
  return { x1: a.x + a.width / 2, y1: a.y + a.height, x2: b.x + b.width / 2, y2: b.y };
}

export function edgePath(from: Cell, to: Cell): string {
  const { x1, y1, x2, y2 } = ends(from, to), ym = (y1 + y2) / 2;
  return `M ${x1},${y1} C ${x1},${ym} ${x2},${ym} ${x2},${y2}`;
}

export function edgePoints(from: Cell, to: Cell, samples = 24): Pt[] {
  const { x1, y1, x2, y2 } = ends(from, to), ym = (y1 + y2) / 2;
  return Array.from({ length: samples + 1 }, (_, i) => {
    const t = i / samples, u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return { x: (a + b) * x1 + (c + d) * x2, y: a * y1 + (b + c) * ym + d * y2 };
  });
}

/** First row of each distinct year, ascending. */
export function yearRows(r: CollectionRoadmap): { year: number; row: number }[] {
  const first = new Map<number, number>();
  for (const n of r.nodes) first.set(n.year, Math.min(first.get(n.year) ?? n.row, n.row));
  return [...first].map(([year, row]) => ({ year, row })).sort((a, b) => a.year - b.year);
}

const bare = (id: string) => id.replace(/v\d+$/, '');

/** Match by paper id, else by version-less arXiv id; course papers never match. */
export function matchPaper(node: { paper?: string; arxiv?: string }, papers: Paper[]): Paper | undefined {
  const pool = papers.filter(p => p.kind !== 'course');
  if (node.paper) return pool.find(p => p.id === node.paper);
  const want = node.arxiv && bare(node.arxiv);
  if (!want) return;
  return pool.find(p => (p.arxiv && bare(p.arxiv) === want) || (() => { const id = parseArxivUrl(p.url)?.arxivId; return !!id && bare(id) === want; })());
}

/** The id itself plus direct predecessors and successors. */
export function neighbors(r: CollectionRoadmap, id: string): Set<string> {
  const out = new Set([id]);
  for (const [a, b] of r.edges) { if (a === id) out.add(b); if (b === id) out.add(a); }
  return out;
}
