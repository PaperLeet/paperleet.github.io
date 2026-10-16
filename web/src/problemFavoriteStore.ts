export type FavoriteItem={problem_id:string;created_at:string};
export type FavoriteSnapshot={owner_id:string;revision:number;items:FavoriteItem[]};
export type FavoritePending={active:boolean;busy:boolean;error?:string};
export type FavoriteState={snapshot?:FavoriteSnapshot;loading:boolean;error:string;pending:Record<string,FavoritePending>};
type Request=(body?:{problem_id:string;active:boolean;expected_user_id:string},signal?:AbortSignal)=>Promise<unknown>;
export function favoriteSnapshot(value:unknown,owner:string):FavoriteSnapshot {
  const row=value as FavoriteSnapshot;
  if(!row||row.owner_id!==owner||!Number.isSafeInteger(row.revision)||row.revision<0||!Array.isArray(row.items)
    ||row.items.some(item=>!item||typeof item.problem_id!=='string'||!item.problem_id||item.problem_id.length>180||typeof item.created_at!=='string'||!Number.isFinite(Date.parse(item.created_at)))
    ||new Set(row.items.map(item=>item.problem_id)).size!==row.items.length)throw Error('Could not confirm your saved problems.');
  return {owner_id:owner,revision:row.revision,items:row.items.map(item=>({problem_id:item.problem_id,created_at:item.created_at}))};
}
export function createFavoriteStore(owner:string,request:Request,changed:()=>void=()=>{}) {
  let state:FavoriteState={loading:false,error:'',pending:{}},read:Promise<void>|undefined;
  const listeners=new Set<()=>void>();
  const update=(next:FavoriteState)=>{state=next;listeners.forEach(listener=>listener());};
  const accept=(snapshot:FavoriteSnapshot)=>!state.snapshot||snapshot.revision>=state.snapshot.revision?snapshot:state.snapshot;
  async function refresh(){
    if(read)return read;
    update({...state,loading:true,error:''});
    read=(async()=>{try{const snapshot=favoriteSnapshot(await request(undefined,AbortSignal.timeout(20_000)),owner);update({...state,snapshot:accept(snapshot),error:''});}
      catch(error){update({...state,error:error instanceof Error?error.message:'Could not load saved problems.'});}
      finally{read=undefined;update({...state,loading:false});}})();
    return read;
  }
  async function set(id:string,active:boolean){
    if(state.pending[id]?.busy||!state.snapshot||state.error)return;
    // A retry keeps its original direction, including after an uncertain transport failure.
    active=state.pending[id]?.active??active;
    update({...state,pending:{...state.pending,[id]:{active,busy:true}}});
    try{
      const value=await request({problem_id:id,active,expected_user_id:owner},AbortSignal.timeout(20_000));
      const snapshot=favoriteSnapshot(value,owner),ack=value as {problem_id:string;active:boolean};
      if(ack.problem_id!==id||ack.active!==active||snapshot.items.some(item=>item.problem_id===id)!==active)throw Error('Could not confirm the bookmark change.');
      const pending={...state.pending};delete pending[id];update({...state,snapshot:accept(snapshot),pending});changed();
    }catch(error){update({...state,pending:{...state.pending,[id]:{active,busy:false,error:error instanceof Error?error.message:'Could not save the bookmark.'}}});}
  }
  return {getSnapshot:()=>state,subscribe:(listener:()=>void)=>{listeners.add(listener);return()=>{listeners.delete(listener);};},refresh,set};
}
