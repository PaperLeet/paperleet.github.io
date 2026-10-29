import type {Category,Paper,Problem} from './types';

export type Index={schema_version?:number;problems:Problem[];papers:Paper[];categories?:Category[]};
function plainIndex(value:unknown):Index{
 if(Array.isArray(value))return {problems:value as Problem[],papers:[]};
 if(value&&typeof value==='object'&&'problems' in value&&Array.isArray(value.problems)&&(!('schema_version' in value)||value.schema_version===2))return {...value,papers:'papers' in value&&Array.isArray(value.papers)?value.papers:[]} as Index;
 throw Error('Invalid catalog index');
}
/** Restore the established model before any editor or progress consumer sees it. */
export async function decodeCatalogIndex(value:unknown,base='',fetcher:typeof fetch=fetch):Promise<Index>{
 if(Array.isArray(value)||value&&typeof value==='object'&&(!('schema_version' in value)||value.schema_version===2))return plainIndex(value);
 return (await import('./catalogTransport')).decodeCatalogTransport(value,base,fetcher);
}
export async function loadCatalogIndex(base:string,fetcher:typeof fetch=fetch):Promise<Index>{
 let transport:typeof import('./catalogTransport')|undefined;
 if(typeof DecompressionStream==='function'){
  // Only a missing decoder chunk may use the plain legacy transport. Decode,
  // digest and source errors below remain visible and never silently downgrade.
  try{transport=await import('./catalogTransport');}catch{/* A retained plain index needs no decoder chunk. */}
 }
 if(transport)return transport.loadCatalogTransport(base,fetcher);
 const response=await fetcher(`${base}data/index.json`,{cache:'no-cache'});
 if(!response.ok)throw Error('Catalog unavailable');
 return plainIndex(await response.json());
}
