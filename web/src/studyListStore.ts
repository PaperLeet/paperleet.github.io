export const STUDY_LIST_LIMIT=30,STUDY_LIST_ITEM_LIMIT=200;
export type StudyList={id:string;title:string;created_at:string;updated_at:string;items:{problem_id:string;created_at:string}[]};
export type StudyListsSnapshot={owner_id:string;revision:number;lists:StudyList[]};
export type StudyListOperation={list_id:string}&({action:'create';title:string;problem_ids?:string[]}|{action:'rename';title:string}|{action:'delete'}|{action:'add';problem_ids:string[]}|{action:'remove';problem_id:string}|{action:'reorder';problem_ids:string[]});
export type StudyListCommand=StudyListOperation&{expected_user_id:string;expected_revision:number;request_id:string};
export type StudyListPending={command:StudyListCommand;busy:boolean;error?:string;conflict?:boolean;rejected?:boolean};
export type StudyListsState={snapshot?:StudyListsSnapshot;loading:boolean;error:string;pending?:StudyListPending;confirmation?:{request_id:string;list_id:string;revision:number}};
const id=(value:unknown):value is string=>typeof value==='string'&&/^[a-f0-9]{32}$/.test(value);
const problem=(value:unknown):value is string=>typeof value==='string'&&value.length<=180&&/^[a-z0-9_-]+\/[a-z0-9_-]+$/.test(value);
const date=(value:unknown)=>typeof value==='string'&&Number.isFinite(Date.parse(value));
export const newStudyListId=()=>crypto.randomUUID().replaceAll('-','');
export function studyListsSnapshot(value:unknown,owner:string):StudyListsSnapshot{
 const s=value as StudyListsSnapshot;
 if(!s||s.owner_id!==owner||!Number.isSafeInteger(s.revision)||s.revision<0||!Array.isArray(s.lists)||s.lists.length>STUDY_LIST_LIMIT||new Set(s.lists.map(row=>row?.id)).size!==s.lists.length||s.lists.some(row=>!row||!id(row.id)||typeof row.title!=='string'||!row.title.trim()||row.title!==row.title.trim()||Array.from(row.title).length>60||/[\u0000-\u001f\u007f]/.test(row.title)||!date(row.created_at)||!date(row.updated_at)||!Array.isArray(row.items)||row.items.length>STUDY_LIST_ITEM_LIMIT||row.items.some(item=>!item||!problem(item.problem_id)||!date(item.created_at))||new Set(row.items.map(item=>item.problem_id)).size!==row.items.length))throw Error('Could not confirm your private study lists.');
 return {owner_id:owner,revision:s.revision,lists:s.lists.map(row=>({...row,items:row.items.map(item=>({...item}))}))};
}
function operation(command:StudyListCommand){const {request_id,expected_user_id,expected_revision,...op}=command;return op;}
function same(left:unknown,right:unknown):boolean{
 if(left===right)return true;
 if(!left||!right||typeof left!=='object'||typeof right!=='object'||Array.isArray(left)!==Array.isArray(right))return false;
 const a=Object.entries(left),b=Object.entries(right);return a.length===b.length&&a.every(([key,value])=>Object.hasOwn(right,key)&&same(value,(right as Record<string,unknown>)[key]));
}
function applied(snapshot:StudyListsSnapshot,op:StudyListOperation){
 const list=snapshot.lists.find(row=>row.id===op.list_id);
 if(op.action==='delete')return !list;if(!list)return false;
 if(op.action==='create')return list.title===op.title&&same(list.items.map(item=>item.problem_id),op.problem_ids||[]);
 if(op.action==='rename')return list.title===op.title;
 const ids=list.items.map(item=>item.problem_id);
 if(op.action==='add')return op.problem_ids.every(id=>ids.includes(id));
 if(op.action==='remove')return !ids.includes(op.problem_id);
 return same(ids,op.problem_ids);
}
export function createStudyListStore(owner:string,request:(body?:StudyListCommand,signal?:AbortSignal)=>Promise<unknown>,changed:()=>void=()=>{}){
 let state:StudyListsState={loading:false,error:''},read:Promise<boolean>|undefined;
 const listeners=new Set<()=>void>(),emit=(next:StudyListsState)=>{state=next;listeners.forEach(fn=>fn());};
 const accept=(snapshot:StudyListsSnapshot)=>!state.snapshot||snapshot.revision>=state.snapshot.revision?snapshot:state.snapshot;
 async function refresh(){
  if(read)return read;emit({...state,loading:true,error:''});
  read=(async()=>{try{const snapshot=studyListsSnapshot(await request(undefined,AbortSignal.timeout(20_000)),owner);emit({...state,snapshot:accept(snapshot),error:''});return true;}catch(error){emit({...state,error:error instanceof Error?error.message:'Could not load study lists.'});return false;}finally{read=undefined;emit({...state,loading:false});}})();return read;
 }
 async function run(command:StudyListCommand){
  emit({...state,pending:{command,busy:true}});
  try{
   const response=await request(command,AbortSignal.timeout(20_000)),snapshot=studyListsSnapshot(response,owner),ack=(response as {ack?:{request_id:string;operation:StudyListOperation;revision:number}}).ack;
   if(!ack||ack.request_id!==command.request_id||!same(ack.operation,operation(command))||!Number.isSafeInteger(ack.revision)||ack.revision<command.expected_revision||(['create','delete'].includes(command.action)&&ack.revision===command.expected_revision)||ack.revision>snapshot.revision||snapshot.revision===ack.revision&&!applied(snapshot,operation(command)))throw Error('The study-list change has not been confirmed. Retry to confirm it.');
   emit({...state,snapshot:accept(snapshot),pending:undefined,error:'',confirmation:{request_id:command.request_id,list_id:command.list_id,revision:ack.revision}});changed();return true;
  }catch(error){const e=error as {status?:number;code?:string};emit({...state,pending:{command,busy:false,error:error instanceof Error?error.message:'Could not save the study list.',conflict:e.code==='study_list_changed',rejected:!!e.status&&e.status>=400&&e.status<500&&![408,429].includes(e.status)&&e.code!=='study_list_changed'}});return false;}
 }
 return {subscribe:(fn:()=>void)=>{listeners.add(fn);return()=>{listeners.delete(fn);};},getSnapshot:()=>state,refresh,
  async mutate(op:StudyListOperation){if(state.pending||state.loading||state.error||!state.snapshot)return false;const captured=JSON.parse(JSON.stringify(op)) as StudyListOperation;return run({...captured,expected_user_id:owner,expected_revision:state.snapshot.revision,request_id:newStudyListId()});},
  async retry(){const pending=state.pending;if(!pending||pending.busy||pending.conflict||pending.rejected)return false;return run(pending.command);},
  async loadLatest(){if(state.pending?.busy)return false;const pending=state.pending;if(!pending?.conflict)return refresh();const ok=await refresh();if(ok&&state.pending===pending)emit({...state,pending:undefined});return ok;},
  dismissRejected(){if(state.pending?.rejected&&!state.pending.busy)emit({...state,pending:undefined});}
 };
}
