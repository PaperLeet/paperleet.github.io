import type {Category,Paper,Problem} from './types';

type Index={schema_version?:number;problems:Problem[];papers:Paper[];categories?:Category[]};
const record=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const invalid=()=>new Error('Invalid catalog index');

/** Expand only the source transport; callers keep the exact established Problem model. */
export async function decodeCatalogIndex(value:unknown):Promise<Index>{
 if(Array.isArray(value))return {problems:value as Problem[],papers:[]};
 if(!record(value)||!Array.isArray(value.problems))throw invalid();
 if(value.schema_version===2||value.schema_version===undefined)return {...value,papers:Array.isArray(value.papers)?value.papers:[]} as Index;
 if(!Array.isArray(value.papers))throw invalid();
 if(value.schema_version!==3||!record(value.starters))throw invalid();
 const source=value.starters;
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
 const starters:unknown=JSON.parse(text);
 if(!record(starters)||Object.keys(starters).length!==value.problems.length)throw invalid();
 const seen=new Set<string>();
 const problems=value.problems.map((problem:unknown)=>{
  if(!record(problem)||typeof problem.id!=='string'||!Object.hasOwn(starters,problem.id)||typeof starters[problem.id]!=='string'||seen.has(problem.id)||Object.hasOwn(problem,'starter'))throw invalid();
  seen.add(problem.id);return {...problem,starter:starters[problem.id]} as Problem;
 });
 const {starters:_sources,...index}=value;
 return {...index,schema_version:2,problems} as Index;
}

export async function loadCatalogIndex(base:string,fetcher:typeof fetch=fetch):Promise<Index>{
 const compact=typeof DecompressionStream==='function';
 let response=await fetcher(`${base}data/${compact?'index-v3':'index'}.json`,{cache:'no-cache'});
 if(compact&&(response.status===404||response.status===410))response=await fetcher(`${base}data/index.json`,{cache:'no-cache'});
 if(!response.ok)throw Error('Catalog unavailable');
 return decodeCatalogIndex(await response.json());
}
