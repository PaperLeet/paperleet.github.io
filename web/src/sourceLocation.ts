import type { ReportSourceState } from './reportSource';

export type SourceLocation = { line: number; column?: number };
export type LearnerTraceFrame = { row: number; location: SourceLocation };
/** Without a location, focus the existing selection; a claim is a single-use permission. */
export type EditorFocusRequest = { source: string; location?: SourceLocation; claimFocus?: () => boolean };

export const editorSource = (source: string) => source.replace(/\r\n|\r/g, '\n');

/** Python columns count Unicode characters; editor offsets count UTF-16 units. */
export function sourceOffset(source: string, { line, column = 1 }: SourceLocation): number | undefined {
  if (!Number.isSafeInteger(line) || line < 1 || !Number.isSafeInteger(column) || column < 1) return;
  const lines = source.split('\n');
  // CPython can locate an incomplete statement immediately after the last line.
  if (line === lines.length + 1 && column === 1) return source.length;
  if (line > lines.length) return;
  const before = lines.slice(0, line - 1).reduce((length, text) => length + text.length + 1, 0);
  return before + Array.from(lines[line - 1]).slice(0, column - 1).join('').length;
}

/** Advisory navigation only. Never infer a learner location from test/reference frames. */
export function learnerTraceLocation(message: string | undefined, impl: string | undefined, source: string, state: ReportSourceState): SourceLocation | undefined {
  return learnerTraceFrames(message,impl,source,state).at(-1)?.location;
}

/** Preserve raw traceback rows, including repeated calls, for explicit navigation. */
export function learnerTraceFrames(message: string | undefined, impl: string | undefined, source: string, state: ReportSourceState): LearnerTraceFrame[] {
  if (state !== 'current' || !message || !impl || impl.length > 2048) return [];
  const normalize = (path: string) => path.replace(/\\/g, '/').replace(/^\.\//, '');
  const implementation = normalize(impl);
  if (!implementation.includes('/') && implementation !== 'submission.py') return [];
  const lineCount = editorSource(source).split('\n').length;
  const frames:LearnerTraceFrame[]=[];
  for (const [row,text] of message.split(/\r\n|\r|\n/).entries()) {
    const frame = /^\s*(.+):([1-9]\d*): in \S.*$/.exec(text) || /^\s*File ["'](.+)["'], line ([1-9]\d*), in (?:[A-Za-z_]\w*|<[^>\r\n]+>)\s*$/.exec(text);
    if (!frame) continue;
    const file = normalize(frame[1]);
    const matches = implementation.endsWith('/submission.py') || implementation === 'submission.py'
      ? file === 'submission.py' || file.endsWith('/submission.py')
      : file === implementation || (!file.startsWith('/') && !/^[A-Za-z]:/.test(file) && file.includes('/') && implementation.endsWith('/' + file));
    const line = Number(frame[2]);
    if (matches && Number.isSafeInteger(line) && line <= lineCount) frames.push({row,location:{line}});
  }
  return frames;
}
