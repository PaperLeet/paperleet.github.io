import {parsePostMarkdown,postPreview} from './postExcerpt';
import {readInterviewQuestion,type InterviewQuestionScratch} from './interviewQuestionDraft';
import type {InterviewModule} from './InterviewFields';
export type QuestionRecordEdit={before:string;after:string;caret:number;scratch:InterviewQuestionScratch};
const inline=(value:string)=>value.replace(/([\\`*_{}\[\]()<>#+.!|$~-])/g,'\\$1');
/** Commit one complete, ordinary Markdown record; private scratch never travels in an API field. */
export function insertInterviewQuestion(body:string,scratch:InterviewQuestionScratch,questions:string[],problems:InterviewModule[],locale:'zh'|'en',position=body.length):{ok:true;edit:QuestionRecordEdit}|{ok:false;reason:'empty'|'length'|'module'|'position'}{
 const value=readInterviewQuestion(scratch);if(!value||!value.prompt.trim())return {ok:false,reason:'empty'};
 const module=value.module_id?problems.find(problem=>problem.id===value.module_id):undefined;
 if(value.module_id&&(!questions.includes(value.module_id)||!module))return {ok:false,reason:'module'};
 const zh=locale==='zh',eol=body.match(/\r?\n/)?.[0]||'\n',excerpt=postPreview(value.prompt,70,zh?'[公式]':'[formula]').text;
 const heading=inline(excerpt||(zh?'问题记录':'Interview question'));
 let block='### '+heading+eol+eol;const headings=[{offset:0,depth:3}];
 if(module){const name=locale==='en'?module.translations?.en?.title||module.title:module.title;block+=(zh?'关联知识点：':'Linked concept: ')+module.number+'. '+inline(name)+eol+eol;}
 for(const [label,text]of [[zh?'实际问题':'Question',value.prompt],[zh?'我的思路':'My approach',value.approach],[zh?'追问与复盘':'Follow-ups and reflection',value.followups]])if(text.trim()){
  headings.push({offset:block.length,depth:4});block+='#### '+label+eol+eol+text+eol+eol;
 }
 let index=Number.isSafeInteger(position)?Math.max(0,Math.min(body.length,position)):body.length;
 if(index>0&&index<body.length&&/[\uD800-\uDBFF]/.test(body[index-1])&&/[\uDC00-\uDFFF]/.test(body[index]))index++;
 const prefix=body.slice(0,index),suffix=body.slice(index),space=!prefix||prefix.endsWith(eol+eol)?'':prefix.endsWith(eol)?eol:eol+eol,base=prefix.length+space.length;
 const after=prefix+space+block+suffix;if(after.length>20000)return {ok:false,reason:'length'};
 try{
  const boundary='## paperleet-question-boundary'+eol+eol,probe=parsePostMarkdown(prefix+space+block+boundary+suffix),tree=parsePostMarkdown(after);
  const has=(root:typeof tree,offset:number,depth:number)=>(root.children||[]).some(node=>node.type==='heading'&&node.depth===depth&&node.position?.start.offset===offset);
  if(!has(probe,base+block.length,2)||headings.some(item=>!has(probe,base+item.offset,item.depth)||!has(tree,base+item.offset,item.depth)))return {ok:false,reason:'position'};
 }catch{return {ok:false,reason:'position'};}
 return {ok:true,edit:{before:body,after,caret:base+block.length,scratch:{...value}}};
}
