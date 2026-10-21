import type {Index} from './catalogIndex';
import type {Problem} from './types';

const record=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const invalid=()=>new Error('Invalid catalog index');

/** Retain older catalogs and decoder-free browsers without growing v4 startup. */
export async function decodeLegacyCatalog(value:unknown,decompress:(source:unknown)=>Promise<string>):Promise<Index>{
 if(Array.isArray(value))return {problems:value as Problem[],papers:[]};
 if(!record(value)||!Array.isArray(value.problems))throw invalid();
 if(value.schema_version===2||value.schema_version===undefined)return {...value,papers:Array.isArray(value.papers)?value.papers:[]} as Index;
 if(value.schema_version!==3||!Array.isArray(value.papers)||!record(value.starters))throw invalid();
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
