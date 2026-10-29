
import type {Index} from './catalogIndex';
const record=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const invalid=()=>new Error('Invalid catalog index');

/** Keep compressed-version negotiation and bounded reads in the deferred codec. */
export async function loadCatalogTransport(base:string,fetcher:typeof fetch):Promise<Index>{
 for(const name of ['index-v6','index-v5','index-v4','index']){
  const response=await fetcher(`${base}data/${name}.json`,{cache:'no-cache'});
  if(response.ok)return decodeCatalogTransport(await readCatalogResponse(response,name==='index-v6'?256*1024:2*1024*1024),base,fetcher);
  if(response.status!==404&&response.status!==410)throw Error('Catalog unavailable');
 }
 throw Error('Catalog unavailable');
}

/** Expand only the source transport; callers keep the exact established Problem model. */
export async function decodeCatalogTransport(value:unknown,base:string,fetcher:typeof fetch):Promise<Index>{
 if(record(value)&&value.schema_version===6)return hydrateStarters(value,base,fetcher);
 if(record(value)&&(value.schema_version===4||value.schema_version===5)){
  const dictionary=value.schema_version===5;
  const packed:unknown=JSON.parse(await decompress(value.payload));
  if(!record(packed)||!Array.isArray(packed.shapes)||packed.shapes.length>256)throw invalid();
  const shapes=packed.shapes;
  if(shapes.some(keys=>!Array.isArray(keys)||keys.length>256||keys.some(key=>typeof key!=='string')||new Set(keys).size!==keys.length))throw invalid();
  const table=dictionary?packed.strings:[];
  if(!Array.isArray(table)||table.length>100000||table.some(text=>typeof text!=='string'))throw invalid();
  const strings:string[]=table;
  let nodes=0,characters=0;
  const text=(value:string)=>{characters+=value.length;if(characters>2*1024*1024)throw invalid();return value;};
  function unpack(item:unknown,depth=0):unknown{
   if(depth>32||++nodes>100000)throw invalid();
   if(Array.isArray(item))return item.map(child=>unpack(child,depth+1));
   if(record(item)){
    const entries=Object.entries(item);if(entries.length!==1)throw invalid();
    const [key,values]=entries[0];
    if(dictionary&&key==='s'){
     if(!Number.isSafeInteger(values)||Number(values)<0||Number(values)>=strings.length)throw invalid();
     return text(strings[Number(values)]);
    }
    if(!/^\d+$/.test(key))throw invalid();
    const keys=shapes[Number(key)];
    if(!Array.isArray(keys)||!Array.isArray(values)||keys.length!==values.length)throw invalid();
    return Object.fromEntries(keys.map((field,index)=>[text(field),unpack(values[index],depth+1)]));
   }
   if(typeof item==='string'){if(dictionary)throw invalid();return text(item);}
   return item;
  }
  const index=unpack(packed.tree);
  if(!record(index)||index.schema_version!==2||!Array.isArray(index.problems)||!Array.isArray(index.papers))throw invalid();
  const ids=new Set<string>();
  for(const problem of index.problems){if(!record(problem)||typeof problem.id!=='string'||typeof problem.starter!=='string'||ids.has(problem.id))throw invalid();ids.add(problem.id);}
  return index as Index;
 }
 // Current catalogs need no legacy source-map decoder in their initial bundle.
 return (await import('./legacyCatalogIndex')).decodeLegacyCatalog(value,decompress);
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


/** Bound bytes before parsing, including streamed responses without Content-Length. */
async function responseBytes(response:Response,limit:number,expected?:number):Promise<Uint8Array<ArrayBuffer>>{
 if(!response.body)throw invalid();
 const reader=response.body.getReader(),chunks:Uint8Array[]=[];let size=0;
 try{
  for(;;){const part=await reader.read();if(part.done)break;size+=part.value.byteLength;if(size>limit||expected!==undefined&&size>expected)throw invalid();chunks.push(part.value);}
 }catch(error){await reader.cancel().catch(()=>{});throw error;}finally{reader.releaseLock();}
 if(expected!==undefined&&size!==expected)throw invalid();
 const bytes=new Uint8Array(size);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}return bytes;
}
const parseBytes=(bytes:Uint8Array)=>JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes)) as unknown;
export async function readCatalogResponse(response:Response,limit:number):Promise<unknown>{return parseBytes(await responseBytes(response,limit));}

async function hydrateStarters(value:Record<string,unknown>,base:string,fetcher:typeof fetch):Promise<Index>{
 const descriptor=value.starters;
 if(!record(descriptor)||Object.keys(descriptor).length!==3||typeof descriptor.sha256!=='string'||!/^[a-f0-9]{64}(?![\s\S])/.test(descriptor.sha256)||descriptor.url!==`starters/${descriptor.sha256}.json`||!Number.isSafeInteger(descriptor.bytes)||Number(descriptor.bytes)<1||Number(descriptor.bytes)>256*1024)throw invalid();
 const metadata=await decodeCatalogTransport({schema_version:5,payload:value.payload},base,fetcher);
 if(metadata.problems.some(problem=>problem.starter!==''))throw invalid();
 const response=await fetcher(`${base}data/${descriptor.url}`,{cache:'force-cache',credentials:'omit',redirect:'error'});
 if(!response.ok)throw Error('Catalog starters unavailable');
 const bytes=await responseBytes(response,256*1024,Number(descriptor.bytes));
 const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),byte=>byte.toString(16).padStart(2,'0')).join('');
 if(digest!==descriptor.sha256)throw invalid();
 const asset=parseBytes(bytes);
 if(!record(asset)||asset.schema_version!==1||Object.keys(asset).length!==2)throw invalid();
 const sources:unknown=JSON.parse(await decompress(asset.payload));
 if(!record(sources)||Object.keys(sources).length!==metadata.problems.length)throw invalid();
 const problems=metadata.problems.map(problem=>{
  if(!Object.hasOwn(sources,problem.id)||typeof sources[problem.id]!=='string')throw invalid();
  return {...problem,starter:sources[problem.id] as string};
 });
 const result={...metadata,problems};let characters=0,nodes=0;
 const bounded=(item:unknown,depth=0):void=>{
  if(depth>32||++nodes>100000)throw invalid();
  if(typeof item==='string')characters+=item.length;
  else if(Array.isArray(item))for(const child of item)bounded(child,depth+1);
  else if(record(item))for(const [key,child] of Object.entries(item)){characters+=key.length;bounded(child,depth+1);}
  if(characters>2*1024*1024)throw invalid();
 };
 bounded(result);return result;
}
