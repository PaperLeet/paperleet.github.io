import type {Category,Paper,Problem} from './types';

type Index={schema_version?:number;problems:Problem[];papers:Paper[];categories?:Category[]};
const record=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const invalid=()=>new Error('Invalid catalog index');

/** Expand only the source transport; callers keep the exact established Problem model. */
export async function decodeCatalogIndex(value:unknown):Promise<Index>{
 if(Array.isArray(value))return {problems:value as Problem[],papers:[]};
 if(record(value)&&value.schema_version===4){
  const packed:unknown=JSON.parse(await decompress(value.payload));
  if(!record(packed)||!Array.isArray(packed.shapes)||packed.shapes.length>256)throw invalid();
  const shapes=packed.shapes;
  if(shapes.some(keys=>!Array.isArray(keys)||keys.length>256||keys.some(key=>typeof key!=='string')||new Set(keys).size!==keys.length))throw invalid();
  let nodes=0;
  function unpack(item:unknown,depth=0):unknown{
   if(depth>32||++nodes>100000)throw invalid();
   if(Array.isArray(item))return item.map(child=>unpack(child,depth+1));
   if(record(item)){
    const entries=Object.entries(item);if(entries.length!==1||!/^\d+$/.test(entries[0][0]))throw invalid();
    const [key,values]=entries[0],keys=shapes[Number(key)];
    if(!Array.isArray(keys)||!Array.isArray(values)||keys.length!==values.length)throw invalid();
    return Object.fromEntries(keys.map((field,index)=>[field,unpack(values[index],depth+1)]));
   }
   return item;
  }
  const index=unpack(packed.tree);
  if(!record(index)||index.schema_version!==2||!Array.isArray(index.problems)||!Array.isArray(index.papers))throw invalid();
  const ids=new Set<string>();
  for(const problem of index.problems){if(!record(problem)||typeof problem.id!=='string'||typeof problem.starter!=='string'||ids.has(problem.id))throw invalid();ids.add(problem.id);}
  return index as Index;
 }
 if(!record(value)||!Array.isArray(value.problems))throw invalid();
 if(value.schema_version===2||value.schema_version===undefined)return {...value,papers:Array.isArray(value.papers)?value.papers:[]} as Index;
 if(!Array.isArray(value.papers))throw invalid();
 if(value.schema_version!==3||!record(value.starters))throw invalid();
 const starters:unknown=JSON.parse(await decompress(value.starters));
 if(!record(starters)||Object.keys(starters).length!==value.problems.length)throw invalid();
 const seen=new Set<string>();
 const problems=value.problems.map((problem:unknown)=>{
  if(!record(problem)||typeof problem.id!=='string'||!Object.hasOwn(starters,problem.id)||typeof starters[problem.id]!=='string'||seen.has(problem.id)||Object.hasOwn(problem,'starter'))throw invalid();
  seen.add(problem.id);return {...problem,starter:starters[problem.id]} as Problem;
 });
 const {starters:_sources,...index}=value;
 return {...index,schema_version:2,problems} as Index;
}

async function decompress(source:unknown):Promise<string>{
 if(!record(source))throw invalid();
 if(source.encoding!=='gzip-base64'||!Number.isSafeInteger(source.bytes)||Number(source.bytes)<0||Number(source.bytes)>2*1024*1024||typeof source.data!=='string'||source.data.length>2*1024*1024)throw invalid();
 const bytes=Uint8Array.from(atob(source.data),char=>char.charCodeAt(0));
 const reader=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')).getReader();
 const decoder=new TextDecoder('utf-8',{fatal:true});
 let size=0,text='';
 try{
  for(;;){const chunk=await reader.read();if(chunk.done)break;size+=chunk.value.byteLength;if(size>Number(source.bytes))throw invalid();text+=decoder.decode(chunk.value,{stream:true});}
  text+=decoder.decode();
 }catch(error){await reader.cancel().catch(()=>{});throw error;}finally{reader.releaseLock();}
 if(size!==source.bytes)throw invalid();
 return text;
}

export async function loadCatalogIndex(base:string,fetcher:typeof fetch=fetch):Promise<Index>{
 const compact=typeof DecompressionStream==='function';
 let response=await fetcher(`${base}data/${compact?'index-v4':'index'}.json`,{cache:'no-cache'});
 if(compact&&(response.status===404||response.status===410))response=await fetcher(`${base}data/index.json`,{cache:'no-cache'});
 if(!response.ok)throw Error('Catalog unavailable');
 return decodeCatalogIndex(await response.json());
}
