export type CollectionFavoriteSnapshot={owner_id:string;revision:number;items:{collection_id:string;created_at:string}[]};
type Pending={active:boolean;busy:boolean;error?:string};
export type CollectionFavoriteState={snapshot?:CollectionFavoriteSnapshot;loading:boolean;error:string;pending:Record<string,Pending>};
type Request=(body?:{collection_id:string;active:boolean;expected_user_id:string},signal?:AbortSignal)=>Promise<unknown>;

function snapshot(value:unknown,owner:string):CollectionFavoriteSnapshot{
 const row=value as CollectionFavoriteSnapshot;
 if(!row||row.owner_id!==owner||!Number.isSafeInteger(row.revision)||row.revision<0||!Array.isArray(row.items)
  ||row.items.some(item=>!item||typeof item.collection_id!=='string'||!/^[a-z0-9][a-z0-9_-]{0,79}$/.test(item.collection_id)||typeof item.created_at!=='string'||!Number.isFinite(Date.parse(item.created_at)))
  ||new Set(row.items.map(item=>item.collection_id)).size!==row.items.length)throw Error('Could not confirm your saved collections.');
 return {owner_id:owner,revision:row.revision,items:row.items.map(item=>({...item}))};
}

/** Stable partition preserves the curated order within both groups. */
export function savedCollectionsFirst<T extends {id:string}>(collections:readonly T[],saved:ReadonlySet<string>):T[]{
 return [...collections.filter(item=>saved.has(item.id)),...collections.filter(item=>!saved.has(item.id))];
}

export function createCollectionFavoriteStore(owner:string,request:Request,changed:()=>void=()=>{}){
 let state:CollectionFavoriteState={loading:false,error:'',pending:{}},read:Promise<void>|undefined;
 const listeners=new Set<()=>void>();
 const update=(next:CollectionFavoriteState)=>{state=next;listeners.forEach(listener=>listener());};
 const accept=(next:CollectionFavoriteSnapshot)=>!state.snapshot||next.revision>=state.snapshot.revision?next:state.snapshot;
 async function refresh(){
  if(read)return read;
  update({...state,loading:true,error:''});
  read=(async()=>{try{const next=snapshot(await request(undefined,AbortSignal.timeout(20_000)),owner);update({...state,snapshot:accept(next),error:''});}
   catch(error){update({...state,error:error instanceof Error?error.message:'Could not load saved collections.'});}
   finally{read=undefined;update({...state,loading:false});}})();
  return read;
 }
 async function set(id:string,active:boolean){
  if(state.pending[id]?.busy||!state.snapshot||state.error)return;
  // An uncertain request is retried in its original direction, never toggled again.
  active=state.pending[id]?.active??active;
  update({...state,pending:{...state.pending,[id]:{active,busy:true}}});
  try{
   const value=await request({collection_id:id,active,expected_user_id:owner},AbortSignal.timeout(20_000));
   const next=snapshot(value,owner),ack=value as {collection_id:string;active:boolean};
   if(ack.collection_id!==id||ack.active!==active||next.items.some(item=>item.collection_id===id)!==active)throw Error('Could not confirm the collection change.');
   const pending={...state.pending};delete pending[id];update({...state,snapshot:accept(next),pending});changed();
  }catch(error){update({...state,pending:{...state.pending,[id]:{active,busy:false,error:error instanceof Error?error.message:'Could not save the collection.'}}});}
 }
 return {getSnapshot:()=>state,subscribe:(listener:()=>void)=>{listeners.add(listener);return()=>{listeners.delete(listener);};},refresh,set};
}
