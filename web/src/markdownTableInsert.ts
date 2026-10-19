import {parsePostMarkdown} from './postExcerpt';
import type {MarkdownEdit} from './markdownFormatting';
export type TableColumns=2|3|4;
export type TableInsertion={ok:true;edit:MarkdownEdit}|{ok:false;reason:'length'|'position'|'invalid'};

/** Map a native textarea's LF offsets back to the original source without normalizing it. */
export function tableSourceOffset(source:string,domOffset:number){
 const end=Number.isFinite(domOffset)?Math.max(0,Math.trunc(domOffset)):0;let dom=0,index=0;
 while(index<source.length&&dom<end){if(source[index]==='\r'&&source[index+1]==='\n')index++;index++;dom++;}
 return index;
}

/** Insert an ordinary GFM table; selection text and surrounding bytes are never replaced. */
export function insertMarkdownTable(body:string,columns:TableColumns,position:number,limit:number,english=false):TableInsertion{
 if(![2,3,4].includes(columns)||!Number.isFinite(position)||!Number.isFinite(limit)||limit<0)return {ok:false,reason:'invalid'};
 if(body.length>20000)return {ok:false,reason:'length'};
 let index=Math.max(0,Math.min(body.length,Math.trunc(position)));
 if(index>0&&index<body.length&&/[\uD800-\uDBFF]/.test(body[index-1])&&/[\uDC00-\uDFFF]/.test(body[index]))index++;
 const eol=body.match(/\r\n|\r|\n/)?.[0]||'\n',header=english?'Column 1':'列 1';
 const row=(values:string[])=>'| '+values.join(' | ')+' |';
 const block=[row(Array.from({length:columns},(_,n)=>english?`Column ${n+1}`:`列 ${n+1}`)),row(Array(columns).fill('---')),row(Array(columns).fill('')),row(Array(columns).fill(''))].join(eol);
 const prefix=body.slice(0,index),suffix=body.slice(index),left=prefix.replace(/\r\n?/g,'\n'),right=suffix.replace(/\r\n?/g,'\n');
 const before=!prefix||left.endsWith('\n\n')?'':left.endsWith('\n')?eol:eol+eol;
 const after=!suffix?eol:right.startsWith('\n\n')?'':right.startsWith('\n')?eol:eol+eol;
 const base=prefix.length+before.length,result=prefix+before+block+after+suffix;
 if(result.length>limit)return {ok:false,reason:'length'};
 try{
  const protectedTypes=new Set(['code','inlineCode','math','inlineMath','html','table','link','linkReference','image','imageReference','strong','emphasis','delete','heading','list','blockquote','definition']);
  type Node={type:string;position?:{start:{offset?:number};end?:{offset?:number}};children?:Node[]};
  const inside=(node:Node):boolean=>{
   const start=node.position?.start.offset,end=node.position?.end?.offset;
   return protectedTypes.has(node.type)&&start!==undefined&&end!==undefined&&index>start&&index<end||!!node.children?.some(inside);
  };
  if(inside(parsePostMarkdown(body)))return {ok:false,reason:'position'};
  const tree=parsePostMarkdown(result),table=tree.children?.find(node=>node.type==='table'&&node.position?.start.offset===base);
  const rows=table?.children;
  if(!table||table.type!=='table'||!rows||rows.length!==3||rows.some(row=>row.children?.length!==columns))return {ok:false,reason:'position'};
 }catch{return {ok:false,reason:'position'};}
 return {ok:true,edit:{before:body,after:result,start:base+2,end:base+2+header.length}};
}
