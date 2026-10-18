export type ReviewNotice={kind:'paper_review'|'report_result'|'post_removed';resource_id:string;decision:string;title:string;feedback:string;phase:string;version:number};
export function reviewNotice(value:unknown):ReviewNotice|undefined{
 if(!value||typeof value!=='object')return;
 const row=value as ReviewNotice;
 if(!/^[a-f0-9]{32}$/.test(row.resource_id)||typeof row.title!=='string'||Array.from(row.title).length>240||typeof row.feedback!=='string'||Array.from(row.feedback).length>1000||!Number.isSafeInteger(row.version)||row.version<0)return;
 if(row.kind==='paper_review'){
  if(!['approve','changes','reject'].includes(row.decision)||!['submission','candidate'].includes(row.phase)||row.phase==='candidate'&&row.version<1)return;
 }else if(row.kind==='report_result'){
  if(!['dismiss','remove','unavailable'].includes(row.decision)||row.phase!==''||row.version!==0||row.feedback!=='')return;
 }else if(row.kind==='post_removed'){
  if(row.decision!=='remove'||row.phase!==''||row.version!==0||row.feedback!=='')return;
 }else return;
 return {kind:row.kind,resource_id:row.resource_id,decision:row.decision,title:row.title,feedback:row.feedback,phase:row.phase,version:row.version};
}
