import type {Post} from './communityDiscussion';
export const reportReasons=[['spam','垃圾广告','Spam or advertising'],['harassment','骚扰或攻击','Harassment or abuse'],['privacy','泄露隐私','Private information'],['misinformation','误导或不实内容','Misleading content'],['copyright','侵权','Copyright'],['other','其他问题','Other']] as const;
export type ReportReason=typeof reportReasons[number][0];
export type OwnReport={owner_id:string;post_id:string;reported:boolean;report_id:string|null;request_id?:string};
export type ReportHistoryItem={id:string;post_id:string;title:string;reason:ReportReason;details:string;created_at:string;status:'pending'|'reviewed';reviewed_at:string|null;post_available:boolean;user_message?:string};
export type ReportHistory={owner_id:string;state:'all'|'pending'|'reviewed';limit:number;offset:number;total:number;summary:{all:number;pending:number;reviewed:number};items:ReportHistoryItem[]};
export type ReportCommand={expected_user_id:string;request_id:string;reason:ReportReason;details:string};
export type ReviewCommand={expected_user_id:string;request_id:string;expected_revision:number;expected_content_version:string|null;action:'dismiss'|'remove';note:string;user_message?:string};
export type ReviewPost=Pick<Post,'id'|'title'|'body'|'category'|'anonymous'|'solution'|'problem_id'|'author'|'edit_version'|'updated_at'|'tags'>;
export type ReviewRow={post_id:string;title:string;revision:number;state:'pending'|'dismissed'|'removed';total_reports:number;pending_reports:number;last_report_at:string;post:Omit<ReviewPost,'body'>|null};
export type ReviewCase=Omit<ReviewRow,'title'|'last_report_at'|'post'>&{post:ReviewPost|null;owner_id:string;offset:number;limit:number;reports:{id:string;reason:ReportReason;details:string;created_at:string;pending:boolean;snapshot:ReviewPost}[];reviews:{id:string;action:'dismiss'|'remove';note:string;created_at:string;user_message?:string}[]};
export const reportRequestId=()=>crypto.randomUUID().replaceAll('-','');
export function confirmedReport(value:OwnReport,owner:string,post:string,request?:string){
 if(!value||value.owner_id!==owner||value.post_id!==post||typeof value.reported!=='boolean'||value.reported&&!/^[a-f0-9]{32}$/.test(value.report_id||'')||request&&(value.request_id!==request||!value.reported))throw Error('The report has not been confirmed. Retry to confirm it.');
 return value;
}
