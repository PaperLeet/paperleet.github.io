export type WritingLocale='zh'|'en';
export type WritingOutlineId='technical'|'system'|'research'|'behavioral'|'process';
type Text={zh:string;en:string};
export type InterviewOutlineCatalog=Record<WritingOutlineId,{name:Text;sections:readonly [Text,Text,Text,Text]}>;
export const interviewOutlines:InterviewOutlineCatalog= {
 technical:{name:{zh:'技术与编程',en:'Technical & coding'},sections:[
  {zh:'面试流程',en:'Interview process'},
  {zh:'考察重点与我的思路',en:'Topics and my approach'},
  {zh:'验证与边界情况',en:'Checks and edge cases'},
  {zh:'复盘与准备建议',en:'Reflection and preparation tips'},
 ]},
 system:{name:{zh:'系统设计',en:'System design'},sections:[
  {zh:'问题与约束',en:'Problem and constraints'},
  {zh:'方案与关键取舍',en:'Design and trade-offs'},
  {zh:'追问与方案演进',en:'Follow-ups and iteration'},
  {zh:'复盘与准备建议',en:'Reflection and preparation tips'},
 ]},
 research:{name:{zh:'研究交流',en:'Research discussion'},sections:[
  {zh:'交流主题与背景',en:'Discussion topic and context'},
  {zh:'核心假设与方法',en:'Hypotheses and methods'},
  {zh:'实验、证据与局限',en:'Experiments, evidence and limitations'},
  {zh:'复盘与准备建议',en:'Reflection and preparation tips'},
 ]},
 behavioral:{name:{zh:'行为与项目经历',en:'Behavioral & project experience'},sections:[
  {zh:'情境与目标',en:'Situation and goal'},
  {zh:'我的行动与取舍',en:'My actions and decisions'},
  {zh:'结果与反馈',en:'Results and feedback'},
  {zh:'复盘与准备建议',en:'Reflection and preparation tips'},
 ]},
 process:{name:{zh:'流程与综合复盘',en:'Process & overall reflection'},sections:[
  {zh:'时间线与轮次',en:'Timeline and rounds'},
  {zh:'各轮考察与交流',en:'Topics across rounds'},
  {zh:'沟通与后续安排',en:'Communication and next steps'},
  {zh:'复盘与准备建议',en:'Reflection and preparation tips'},
 ]},
};
export function defaultInterviewOutline(stage:string):WritingOutlineId{
 return stage==='system-design'?'system':stage==='research'?'research':stage==='behavioral'?'behavioral':stage==='technical'||stage==='assessment'?'technical':'process';
}
export const interviewOutlineText=(id:WritingOutlineId,locale:WritingLocale)=>interviewOutlines[id].sections.map(section=>`## ${section[locale]}\n\n`).join('');
const headings=new Set(Object.values(interviewOutlines).flatMap(outline=>outline.sections.flatMap(section=>[section.zh,section.en])));
/** Recognize our blank scaffolds, including the older three-section outline, without grading free-form writing. */
export function hasInterviewDraftContent(body:string){
 return body.split(/\r?\n/).some(line=>{
  const text=line.trim();if(!text||/^[-*_]{3,}$/.test(text))return false;
  const heading=/^#{1,6}\s+(.+?)(?:\s+#+)?$/.exec(text);
  return !heading||!headings.has(heading[1]);
 });
}
export type InterviewOutlineEdit={before:string;after:string;caret:number};
export function insertInterviewOutline(body:string,id:WritingOutlineId,locale:WritingLocale,position=body.length):InterviewOutlineEdit|undefined{
 const outline=interviewOutlineText(id,locale),blank=!hasInterviewDraftContent(body);
 let index=Number.isSafeInteger(position)?Math.max(0,Math.min(body.length,position)):body.length;
 if(index>0&&index<body.length&&/[\uD800-\uDBFF]/.test(body[index-1])&&/[\uDC00-\uDFFF]/.test(body[index]))index++;
 const prefix=blank?'':body.slice(0,index),suffix=blank?'':body.slice(index);
 const separator=!prefix||prefix.endsWith('\n\n')?'':prefix.endsWith('\n')?'\n':'\n\n';
 const after=prefix+separator+outline+suffix;
 if(after.length>20000)return;
 return {before:body,after,caret:prefix.length+separator.length+outline.indexOf('\n\n')+2};
}
