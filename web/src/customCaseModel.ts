export type CustomCase = {function: string; args: unknown[]; kwargs: Record<string,unknown>; expected?: unknown};
export type CustomCaseResult = {ok: boolean; output?: unknown; matches?: boolean; stdout: string; error?: string};
export type CustomCaseField = 'function'|'args'|'kwargs'|'expected';
export class CustomCaseInputError extends Error {
  constructor(public field:CustomCaseField,public reason:'function'|'size'|'json'|'array'|'object') {super(reason);}
}
export function parseCustomCase(name:string,args:string,kwargs:string,expected:string,allowed:string[]):CustomCase {
  if(!allowed.includes(name)||!/^\w+$/.test(name))throw new CustomCaseInputError('function','function');
  if(new TextEncoder().encode(args+kwargs+expected).length>32768)throw new CustomCaseInputError('args','size');
  function parse(value:string,field:CustomCaseField):unknown {try{return JSON.parse(value);}catch{throw new CustomCaseInputError(field,'json');}}
  const positional=parse(args,'args');
  if(!Array.isArray(positional))throw new CustomCaseInputError('args','array');
  const keywords=parse(kwargs,'kwargs');
  if(!keywords||typeof keywords!=='object'||Array.isArray(keywords))throw new CustomCaseInputError('kwargs','object');
  return {function:name,args:positional,kwargs:keywords as Record<string,unknown>,...(expected.trim()?{expected:parse(expected,'expected')}:{})};
}
export function readCustomResult(value:unknown):CustomCaseResult|undefined {
  if(!value||typeof value!=='object')return;
  const row=value as CustomCaseResult;
  if(typeof row.ok!=='boolean'||typeof row.stdout!=='string'||row.stdout.length>33000||row.error!==undefined&&(typeof row.error!=='string'||row.error.length>8000)||row.matches!==undefined&&typeof row.matches!=='boolean')return;
  try{if(JSON.stringify(row).length>100000)return;}catch{return;}
  return {ok:row.ok,stdout:row.stdout,...('output' in row?{output:row.output}:{}),...(row.error!==undefined?{error:row.error}:{}),...(row.matches!==undefined?{matches:row.matches}:{})};
}
