export type CustomCase = {function: string; args: unknown[]; kwargs: Record<string,unknown>; expected?: unknown};
export type CustomCaseResult = {ok: boolean; output?: unknown; matches?: boolean; stdout: string; error?: string};
export function parseCustomCase(name:string,args:string,kwargs:string,expected:string,allowed:string[]):CustomCase {
  if(!allowed.includes(name)||!/^\w+$/.test(name))throw Error('Choose a function from this module.');
  if(new TextEncoder().encode(args+kwargs+expected).length>32768)throw Error('Custom input must be at most 32 KB.');
  const positional:unknown=JSON.parse(args),keywords:unknown=JSON.parse(kwargs);
  if(!Array.isArray(positional))throw Error('Positional arguments must be a JSON array.');
  if(!keywords||typeof keywords!=='object'||Array.isArray(keywords))throw Error('Keyword arguments must be a JSON object.');
  return {function:name,args:positional,kwargs:keywords as Record<string,unknown>,...(expected.trim()?{expected:JSON.parse(expected)}:{})};
}
export function readCustomResult(value:unknown):CustomCaseResult|undefined {
  if(!value||typeof value!=='object')return;
  const row=value as CustomCaseResult;
  if(typeof row.ok!=='boolean'||typeof row.stdout!=='string'||row.stdout.length>33000||row.error!==undefined&&(typeof row.error!=='string'||row.error.length>8000)||row.matches!==undefined&&typeof row.matches!=='boolean')return;
  try{if(JSON.stringify(row).length>100000)return;}catch{return;}
  return {ok:row.ok,stdout:row.stdout,...('output' in row?{output:row.output}:{}),...(row.error!==undefined?{error:row.error}:{}),...(row.matches!==undefined?{matches:row.matches}:{})};
}
