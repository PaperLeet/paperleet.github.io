import {parsePostMarkdown} from './postExcerpt';

type Text={zh:string;en:string};
export type CommunityOutlineId='papers'|'implementation'|'questions'|'career'|'lounge';
export type CommunityOutline={name:Text;sections:readonly Text[]};
export const communityOutlines:Record<CommunityOutlineId,CommunityOutline>= {
 papers:{name:{zh:'论文讨论',en:'Paper discussion'},sections:[{zh:'我关注的问题',en:'The question I am exploring'},{zh:'论文中的依据',en:'Evidence in the paper'},{zh:'我的理解与验证',en:'My understanding and checks'},{zh:'想请大家讨论的地方',en:'What I would like to discuss'}]},
 implementation:{name:{zh:'实现与复现',en:'Implementation & reproduction'},sections:[{zh:'目标与参考方法',en:'Goal and reference method'},{zh:'最小实现与设置',en:'Minimal implementation and setup'},{zh:'观察结果与差异',en:'Observed results and differences'},{zh:'已尝试的排查',en:'Checks I have tried'}]},
 questions:{name:{zh:'问题求助',en:'Help with a question'},sections:[{zh:'问题背景与目标',en:'Context and goal'},{zh:'预期行为',en:'Expected behavior'},{zh:'实际结果与最小案例',en:'Actual result and minimal example'},{zh:'已尝试的方法',en:'What I have tried'}]},
 career:{name:{zh:'职业交流',en:'Career discussion'},sections:[{zh:'背景与目标',en:'Background and goal'},{zh:'可选方向与约束',en:'Options and constraints'},{zh:'我的考虑与依据',en:'My reasoning and evidence'},{zh:'希望获得的建议',en:'Advice I am looking for'}]},
 lounge:{name:{zh:'交流与分享',en:'Sharing and conversation'},sections:[{zh:'话题背景',en:'Topic background'},{zh:'想分享的内容',en:'What I would like to share'},{zh:'我的观察',en:'My observations'},{zh:'邀请大家交流',en:'An invitation to discuss'}]},
};
export const communityOutlineText=(id:CommunityOutlineId,en:boolean)=>communityOutlines[id].sections.map(section=>`## ${en?section.en:section.zh}\n\n`).join('');
const headings=new Set(Object.values(communityOutlines).flatMap(row=>row.sections.flatMap(section=>[section.zh,section.en])));
/** Recognize only our empty outlines; free-form writing and custom headings remain valid. */
export function hasCommunityDraftContent(body:string){
 let outlineHeading=false;
 for(const line of body.split(/\r?\n/)){
  const text=line.trim();if(!text||/^[-*_]{3,}$/.test(text))continue;
  const heading=/^#{1,6}\s+(.+?)(?:\s+#+)?$/.exec(text);
  if(!heading||!headings.has(heading[1]))return true;
  outlineHeading=true;
 }
 return !outlineHeading&&!!body.trim();
}
export type CommunityOutlineEdit={before:string;after:string;caret:number;id:CommunityOutlineId;en:boolean};
export function insertCommunityOutline(body:string,id:CommunityOutlineId,en:boolean,position=body.length):{ok:true;edit:CommunityOutlineEdit}|{ok:false;reason:'length'|'position'}{
 if(!Object.hasOwn(communityOutlines,id))return {ok:false,reason:'position'};
 let index=Number.isSafeInteger(position)?Math.max(0,Math.min(body.length,position)):body.length;
 if(typeof Intl.Segmenter==='function'){
  for(const part of new Intl.Segmenter(undefined,{granularity:'grapheme'}).segment(body)){const end=part.index+part.segment.length;if(part.index<index&&index<end){index=end;break;}}
 }else if(index>0&&index<body.length&&/[\uD800-\uDBFF]/.test(body[index-1])&&/[\uDC00-\uDFFF]/.test(body[index]))index++;
 const blank=!hasCommunityDraftContent(body),prefix=blank?'':body.slice(0,index),suffix=blank?'':body.slice(index);
 const separator=!prefix||prefix.endsWith('\n\n')?'':prefix.endsWith('\n')?'\n':'\n\n';
 const outline=communityOutlineText(id,en),start=prefix.length+separator.length,after=prefix+separator+outline+suffix;
 if(after.length>20000)return {ok:false,reason:'length'};
 // Ask the reader's parser to prove that inserted headings are real top-level
 // blocks, rather than text swallowed by an unfinished code/math/HTML block.
 try{
  const positions=new Set((parsePostMarkdown(after).children||[]).filter(node=>node.type==='heading').map(node=>node.position?.start.offset));
  let offset=start;
  for(const section of communityOutlines[id].sections){if(!positions.has(offset))return {ok:false,reason:'position'};offset+=`## ${en?section.en:section.zh}\n\n`.length;}
 }catch{return {ok:false,reason:'position'};}
 return {ok:true,edit:{before:body,after,caret:start+outline.indexOf('\n\n')+2,id,en}};
}
