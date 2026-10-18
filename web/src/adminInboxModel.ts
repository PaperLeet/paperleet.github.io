import {parseArxivUrl} from './arxivLinks';

export type InboxReport={post_id:string;title:string;pending_reports:number;total_reports:number;last_report_at:string;post_available:boolean};
export type InboxPaper={id:string;arxiv_id:string;version:number;status:string;created_at:string};
type Group<T>={total:number;items:T[]};
export type AdminReviewInboxData={owner_id:string;reports:Group<InboxReport>;intake:Group<InboxPaper>;candidates:Group<InboxPaper>};
const record=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const count=(value:unknown):value is number=>Number.isSafeInteger(value)&&Number(value)>=0;
const id=(value:unknown):value is string=>typeof value==='string'&&/^[a-f0-9]{32}$/.test(value);
const date=(value:unknown)=>typeof value==='string'&&value.length<=40&&/^\d{4}-\d{2}-\d{2}T/.test(value)&&Number.isFinite(Date.parse(value));
function group(value:unknown,valid:(row:Record<string,unknown>)=>boolean):boolean{
 return record(value)&&count(value.total)&&Array.isArray(value.items)&&value.items.length===Math.min(3,value.total)&&value.items.every(row=>record(row)&&valid(row));
}
export function isAdminReviewInbox(value:unknown):value is AdminReviewInboxData{
 if(!record(value)||typeof value.owner_id!=='string')return false;
 const paper=(row:Record<string,unknown>)=>id(row.id)&&typeof row.arxiv_id==='string'&&!!parseArxivUrl('https://arxiv.org/abs/'+row.arxiv_id)&&count(row.version)&&date(row.created_at);
 if(!group(value.reports,row=>id(row.post_id)&&typeof row.title==='string'&&row.title.length>0&&row.title.length<=1000&&count(row.pending_reports)&&row.pending_reports>0&&count(row.total_reports)&&row.total_reports>=row.pending_reports&&date(row.last_report_at)&&typeof row.post_available==='boolean')||
    !group(value.intake,row=>paper(row)&&row.status==='pending_review')||
    !group(value.candidates,row=>paper(row)&&Number(row.version)>0&&['pool','ready'].includes(String(row.status))))return false;
 const data=value as AdminReviewInboxData;
 return new Set(data.reports.items.map(row=>row.post_id)).size===data.reports.items.length&&
  new Set([...data.intake.items,...data.candidates.items].map(row=>row.id)).size===data.intake.items.length+data.candidates.items.length;
}
