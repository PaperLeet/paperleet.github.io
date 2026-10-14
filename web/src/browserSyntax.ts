export type SyntaxDiagnostic = {kind:'SyntaxError'|'IndentationError'|'TabError';message:string;line:number;column:number};
export type BrowserSyntaxFailure = {source:string;diagnostic:SyntaxDiagnostic};

/** Syntax feedback is separate from reports; it can never establish a pass. */
export function parseSyntaxDiagnostic(value: unknown, source: string): SyntaxDiagnostic | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return;
  const row=value as Record<string,unknown>;
  if (typeof row.kind !== 'string' || !['SyntaxError','IndentationError','TabError'].includes(row.kind) ||
      typeof row.message !== 'string' || !row.message || row.message.length > 500 ||
      !Number.isSafeInteger(row.line) || Number(row.line)<1 || Number(row.line)>source.split(/\r\n|\r|\n/).length+1 ||
      !Number.isSafeInteger(row.column) || Number(row.column)<1 || Number(row.column)>source.length+1) return;
  return {kind:row.kind as SyntaxDiagnostic['kind'],message:row.message,line:Number(row.line),column:Number(row.column)};
}
