import type {AdminDraftIntent} from './adminDraftStore';
/** One captured operation per administrator. Private tab storage never crosses accounts. */
export type AdminCommand={draft?:AdminDraftIntent;path:string;label:string;resource:string;kind:string;payload:Record<string,unknown>&{expected_user_id:string;expected_revision:number;request_id:string}};
const held=new Map<string,AdminCommand>(),retired=new Map<string,Set<string>>();
const key=(owner:string)=>'paperleet.admin.command.v1:'+encodeURIComponent(owner);
function valid(value:unknown,owner:string):value is AdminCommand{
 if(!value||typeof value!=='object')return false;const c=value as AdminCommand,p=c.payload;
 if(c.draft&&(!/^[a-f0-9]{32}$/.test(c.draft.workspace)||!Number.isSafeInteger(c.draft.stamp)||c.draft.stamp<1||typeof c.draft.key!=='string'||! /^(?:(?:boards|tags):(?:\$new|[a-z][a-z0-9_-]{0,39})|(?:users|papers):[a-f0-9]{32}|highlights:(?:global|board:[a-z][a-z0-9_-]{0,39}):[1-3])$/.test(c.draft.key)))return false;
 if(c.draft&&c.draft.key!==c.kind+':'+c.resource&&!(c.draft.key===c.kind+':$new'&&['boards','tags'].includes(c.kind)))return false;
 if(!p||p.expected_user_id!==owner||!Number.isSafeInteger(p.expected_revision)||p.expected_revision<0||!/^[a-f0-9]{32}$/.test(p.request_id)||typeof p.reason!=='string'||!p.reason.trim()||p.reason.length>1000||typeof c.label!=='string'||c.label.length>100||typeof c.resource!=='string')return false;
 if(c.kind==='highlights')return typeof p.category==='string'&&(p.category===''||/^[a-z][a-z0-9_-]{0,39}$/.test(p.category))&&typeof p.slot==='number'&&[1,2,3].includes(p.slot)&&(p.post_id===null?p.expected_content_version===null:typeof p.post_id==='string'&&/^[a-f0-9]{32}$/.test(p.post_id)&&typeof p.expected_content_version==='string'&&/^[a-f0-9]{64}$/.test(p.expected_content_version))&&/^(?:global|board:[a-z][a-z0-9_-]{0,39}):[1-3]$/.test(c.resource)&&c.path==='admin/highlights'&&c.resource===(p.category?'board:'+p.category:'global')+':'+p.slot;
 return (['boards','tags'].includes(c.kind)&&/^[a-z][a-z0-9_-]{0,39}$/.test(c.resource)&&c.path==='admin/'+c.kind&&p.id===c.resource)||(['users','papers'].includes(c.kind)&&/^[a-f0-9]{32}$/.test(c.resource)&&c.path===`admin/${c.kind}/${c.resource}/${c.kind==='users'?'status':'review'}`);
}
export function readAdminCommand(owner:string):AdminCommand|undefined{
 if(held.has(owner))return structuredClone(held.get(owner));
 try{const raw=sessionStorage.getItem(key(owner));if(!raw||raw.length>20000)return;const value=JSON.parse(raw);if(valid(value,owner)&&!retired.get(owner)?.has(value.payload.request_id)){held.set(owner,structuredClone(value));return value;}}catch{/* Storage may be unavailable. */}
}
export function holdAdminCommand(command:AdminCommand):boolean{
 const owner=command.payload.expected_user_id;if(!valid(command,owner))throw Error('Invalid administrator operation');held.set(owner,structuredClone(command));try{sessionStorage.setItem(key(owner),JSON.stringify(command));return true;}catch{return false;}
}
export function clearAdminCommand(command:AdminCommand){const owner=command.payload.expected_user_id;if(readAdminCommand(owner)?.payload.request_id!==command.payload.request_id)return;held.delete(owner);if(!retired.has(owner))retired.set(owner,new Set());retired.get(owner)!.add(command.payload.request_id);try{sessionStorage.removeItem(key(owner));}catch{/* The captured in-page operation has retired. */}}
