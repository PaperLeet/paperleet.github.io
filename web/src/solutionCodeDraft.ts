export type SolutionCodeSource={owner:string;problemId:string;code:string;ready:boolean};
export type SolutionCodeEdit={before:string;after:string;code:string;start:number;end:number};
export const ownsSolutionCode=(source:SolutionCodeSource|undefined,owner:string,problemId:string|undefined):source is SolutionCodeSource=>!!owner&&!!source&&source.ready===true&&source.owner==='user:'+owner&&source.problemId===problemId&&typeof source.code==='string'&&!!source.code.trim();

/** Append the complete implementation as literal Python; existing prose is never replaced. */
export function insertSolutionCode(body:string,code:string,limit=20000):{ok:true;edit:SolutionCodeEdit}|{ok:false;reason:'empty'|'length'|'position'}{
 if(!code.trim())return {ok:false,reason:'empty'};
 if(!Number.isSafeInteger(limit)||limit<0||body.length>limit||code.length>limit)return {ok:false,reason:'length'};
 const eol=body.match(/\r?\n/)?.[0]||code.match(/\r?\n/)?.[0]||'\n';
 const fence='`'.repeat(Math.max(3,...(code.match(/`+/g)||[]).map(value=>value.length+1)));
 const space=!body?'':body.endsWith(eol+eol)?'':body.endsWith(eol)?eol:eol+eol;
 const prefix=body+space+fence+'python'+eol;
 const after=prefix+code+(code.endsWith('\n')?'':eol)+fence+eol;
 if(after.length>limit)return {ok:false,reason:'length'};
 // The inserted fence must become a top-level literal block, including when
 // the existing draft contains unfinished Markdown, HTML or display math.
 try{
  const node=(parsePostMarkdown(after).children||[]).find(node=>node.type==='code'&&node.position?.start.offset===body.length+space.length);
  if(node?.lang!=='python'||node.value?.replace(/\r\n?/g,'\n')!==code.replace(/\r\n?/g,'\n').replace(/\n$/,''))return {ok:false,reason:'position'};
 }catch{return {ok:false,reason:'position'};}
 return {ok:true,edit:{before:body,after,code,start:prefix.length,end:prefix.length+code.length}};
}
import {parsePostMarkdown} from './postExcerpt';

