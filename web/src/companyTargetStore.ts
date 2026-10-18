export const MAX_COMPANY_TARGETS=12;
export type CompanyTarget={company_id:string;created_at:string};
export type CompanyTargetsSnapshot={owner_id:string;revision:number;items:CompanyTarget[]};
export type CompanyTargetPending={active:boolean;busy:boolean;error?:string};
export type CompanyTargetsState={snapshot?:CompanyTargetsSnapshot;loading:boolean;error:string;accountError:boolean;pending:Record<string,CompanyTargetPending>};
type Payload={expected_user_id:string;company_id:string;active:boolean};
type Request=(body:Payload|undefined,signal:AbortSignal)=>Promise<unknown>;
const identifier=(value:unknown):value is string=>typeof value==='string'&&/^[a-z][a-z0-9-]{0,39}$/.test(value);
class TargetAccountError extends Error{}
const accountFailure=(error:unknown)=>error instanceof TargetAccountError||!!error&&typeof error==='object'&&(('status'in error&&[401,403].includes(Number(error.status)))||('code'in error&&error.code==='account_changed'));
export function companyTargetsSnapshot(value:unknown,owner:string):CompanyTargetsSnapshot{
 const row=value as CompanyTargetsSnapshot;
 if(!row||row.owner_id!==owner)throw new TargetAccountError('Your target companies could not be matched to this account.');
 if(!Number.isSafeInteger(row.revision)||row.revision<0||!Array.isArray(row.items)||row.items.length>MAX_COMPANY_TARGETS||row.items.some(item=>!item||!identifier(item.company_id)||typeof item.created_at!=='string'||item.created_at.length>40||!Number.isFinite(Date.parse(item.created_at)))||new Set(row.items.map(item=>item.company_id)).size!==row.items.length)throw Error('Your target companies could not be confirmed.');
 return {owner_id:owner,revision:row.revision,items:row.items.map(({company_id,created_at})=>({company_id,created_at}))};
}
export function createCompanyTargetStore(owner:string,request:Request,changed:()=>void=()=>{}){
 let state:CompanyTargetsState={loading:false,error:'',accountError:false,pending:{}},live=true,epoch=0,read:Promise<void>|undefined;
 const listeners=new Set<()=>void>(),controllers=new Set<AbortController>();
 const update=(next:CompanyTargetsState)=>{if(live){state=next;listeners.forEach(listener=>listener());}};
 function accept(value:CompanyTargetsSnapshot){
  if(!state.snapshot||value.revision>state.snapshot.revision)return value;
  if(value.revision===state.snapshot.revision&&JSON.stringify(value.items)!==JSON.stringify(state.snapshot.items))throw Error('Conflicting target-company data. Load the latest state.');
  return state.snapshot;
 }
 async function send(body?:Payload){
  const controller=new AbortController();controllers.add(controller);const timer=setTimeout(()=>controller.abort(),20_000);
  try{return await request(body,controller.signal);}finally{clearTimeout(timer);controllers.delete(controller);}
 }
 async function refresh(){
  if(!live)return;if(read)return read;
  const generation=epoch;update({...state,loading:true,error:''});
  const task=(async()=>{
   try{const value=await send();if(!live||generation!==epoch)return;const snapshot=accept(companyTargetsSnapshot(value,owner));update({...state,snapshot,error:'',accountError:false});}
   catch(error){if(live&&generation===epoch){const accountError=accountFailure(error);update({...state,...(accountError?{snapshot:undefined,pending:{}}:{}),accountError,error:error instanceof Error?error.message:'Target companies are unavailable.'});}}
   finally{if(live&&generation===epoch){read=undefined;update({...state,loading:false});}}
  })();read=task;return task;
 }
 async function set(company_id:string,active:boolean){
  if(!live||!identifier(company_id)||typeof active!=='boolean'||state.pending[company_id]?.busy||!state.snapshot||state.error||state.accountError)return false;
  active=state.pending[company_id]?.active??active;
  const generation=epoch;update({...state,pending:{...state.pending,[company_id]:{active,busy:true}}});
  try{
   const value=await send({company_id,active,expected_user_id:owner});if(!live||generation!==epoch)return false;
   const snapshot=companyTargetsSnapshot(value,owner),ack=value as {company_id:unknown;active:unknown};
   if(ack.company_id!==company_id||ack.active!==active||snapshot.items.some(row=>row.company_id===company_id)!==active)throw Error('The target-company change is unconfirmed. Retry the original action.');
   const accepted=accept(snapshot),pending={...state.pending};delete pending[company_id];update({...state,snapshot:accepted,pending});changed();return true;
  }catch(error){if(live&&generation===epoch){const accountError=accountFailure(error);update(accountError?{loading:false,snapshot:undefined,pending:{},accountError:true,error:'Refresh your account before managing target companies.'}:{...state,pending:{...state.pending,[company_id]:{active,busy:false,error:error instanceof Error?error.message:'The change is unconfirmed.'}}});}return false;}
 }
 return {getSnapshot:()=>state,subscribe:(listener:()=>void)=>{listeners.add(listener);return()=>{listeners.delete(listener);};},refresh,set,
  start(){if(!live){live=true;epoch++;state={loading:false,error:'',accountError:false,pending:{}};}},
  stop(){live=false;epoch++;read=undefined;for(const controller of controllers)controller.abort();controllers.clear();}
 };
}
