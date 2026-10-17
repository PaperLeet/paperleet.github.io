import type {Problem,Report} from './types';
import {submissionDestination} from './submissionNavigation';

export type SubmissionCompletion={owner:string;problem:string;number:number;runId:string;href:string};
export type CompletionAction='inline'|'saved'|'ignore';
export type SubmissionCompletionProps={notice:SubmissionCompletion;inline:boolean;onView:(inline:boolean,modified?:boolean)=>CompletionAction;onDismiss:(target?:HTMLElement)=>void};

/** Only a confirmed account submission receives an immutable saved-result action. */
export function submissionCompletion(report:Report,owner:string,entry:Pick<Problem,'id'|'number'>):SubmissionCompletion|undefined {
  const destination=submissionDestination(report,owner,entry);
  if(!destination||!Number.isSafeInteger(entry.number)||entry.number<1)return;
  return {owner,problem:entry.id,number:entry.number,runId:report.run_id!,href:destination.href};
}
export function resultPanelVisible(panel:HTMLElement|null) {
  return !!panel&&!panel.closest('[hidden]')&&panel.getClientRects().length>0;
}
