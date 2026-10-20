import type { BrowserSyntaxFailure } from './browserSyntax';
import type { Report } from './types';
import type { ReportSourceState } from './reportSource';
import { editorSource, learnerTraceLocation, type SourceLocation } from './sourceLocation';
import { reportTestLabel } from './experimentResults';
import { parseComparison, type TestComparison } from './comparisons';

/** Quote a bounded diagnostic; never turn a traceback into a guessed diagnosis. */
export function failureExcerpt(message:string|undefined) {
  const lines = (message || '').slice(-24_000).replace(/\u001b\[[0-9;]*m/g,'').split(/\r\n|\r|\n/).map(line => line.replace(/^E\s{2,}/,'').trim()).filter(Boolean);
  const errors = lines.filter(line => /^(?:[\w.]*Error|[\w.]*Exception):/.test(line));
  const text = errors.at(-1) || lines.filter(line => /^assert\s/.test(line)).at(-1) || lines.at(-1) || '';
  return text.length > 360 ? text.slice(0,360) + '…' : text;
}
export type FailureTarget = {id:string;label:string;message:string;location?:SourceLocation;comparison?:TestComparison;testIndex?:number};
export function reportFailureTargets(report:Report|undefined, syntax:BrowserSyntaxFailure|undefined, source:string, state:ReportSourceState):FailureTarget[] {
  if (syntax?.source === source) return [{id:'syntax',label:syntax.diagnostic.kind,message:syntax.diagnostic.message,location:{line:syntax.diagnostic.line,column:syntax.diagnostic.column}}];
  if (!report) return [];
  const targets:FailureTarget[] = report.error ? [{id:'run-error',label:'',message:failureExcerpt(report.error),location:learnerTraceLocation(report.error,report.impl,source,state)}] : [];
  report.tests.forEach((test,index) => { if (!test.passed) targets.push({id:`test-${index}`,testIndex:index,label:reportTestLabel(test),message:failureExcerpt(test.message),location:learnerTraceLocation(test.message,report.impl,source,state),comparison:state==='current'?parseComparison(test.comparison):undefined}); });
  return targets;
}
export function failureCodeContext(source:string, location?:SourceLocation) {
  if (!location) return [];
  const lines = editorSource(source).split('\n');
  if (location.line < 1 || location.line > lines.length + 1) return [];
  const index = Math.min(location.line - 1,lines.length - 1);
  return lines.slice(Math.max(0,index - 2),Math.min(lines.length,index + 3)).map((text,offset) => ({line:Math.max(0,index - 2) + offset + 1,text:text.length > 260 ? text.slice(0,260) + '…' : text,active:Math.max(0,index - 2) + offset + 1 === location.line}));
}
