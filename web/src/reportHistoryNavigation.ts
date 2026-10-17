export type ReportHistoryState='all'|'pending'|'reviewed';
export const REPORT_HISTORY_PAGE_SIZE=20,MAX_REPORT_HISTORY_PAGE=5000;
export const reportHistoryHref=(state:ReportHistoryState='all',page=1)=>`#/community/reports${state==='all'?'':`?state=${state}`}${page>1?`${state==='all'?'?':'&'}page=${page}`:''}`;
/** Closed canonical filters only: no owner, report text or nested destinations. */
export function readReportHistory(route:string){
 const value=route.replace(/^#\/?/,''),match=/^community\/reports(?:\?state=(pending|reviewed))?(?:[?&]page=([1-9]\d{0,3}))?$/.exec(value);
 if(!match)return;
 const state=(match[1]||'all') as ReportHistoryState,page=Number(match[2]||1);
 return page<=MAX_REPORT_HISTORY_PAGE&&reportHistoryHref(state,page)==='#/'+value?{state,page}:undefined;
}
