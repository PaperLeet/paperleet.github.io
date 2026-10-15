import {useState} from 'react';
import {Plus,Undo2} from 'lucide-react';
import {useI18n} from './i18n';
import type {InterviewOutlineCatalog,WritingOutlineId} from './interviewWriting';

type Text={zh:string;en:string};
const prompts:Record<WritingOutlineId,readonly [Text,Text,Text,Text]>={
 technical:[
  {zh:'记录实际轮次、形式和大致时长；哪些信息是对方明确告知的？',en:'Record the actual rounds, format and approximate duration. What was explicitly communicated?'},
  {zh:'你如何澄清问题、推导方案并解释复杂度？只关联本轮实际考察的知识点。',en:'How did you clarify the problem, derive an approach and explain complexity? Link only concepts actually discussed.'},
  {zh:'你用了哪些例子、测试或边界情况？卡住时如何定位和调整？',en:'Which examples, tests or edge cases did you use? How did you diagnose a problem and adjust?'},
  {zh:'现在重做会改进什么？哪些准备真正有帮助，哪些只是个人感受？',en:'What would you improve next time? Which preparation helped, and which conclusions are personal impressions?'},
 ],
 system:[
  {zh:'需求、规模和约束中，哪些由面试官给出，哪些是你提出并确认的假设？',en:'Which requirements, scale and constraints came from the interviewer, and which assumptions did you propose and confirm?'},
  {zh:'说明数据流、关键组件与取舍。你为什么选择这个方案？',en:'Explain the data flow, key components and trade-offs. Why did you choose this design?'},
  {zh:'对方实际追问了哪些瓶颈、故障或扩展问题？你的方案如何演进？',en:'Which bottlenecks, failures or scaling concerns were actually raised? How did your design evolve?'},
  {zh:'哪些论证充分，哪些还缺证据？下次会如何组织讨论？',en:'Which arguments were well supported and which needed evidence? How would you structure the discussion next time?'},
 ],
 research:[
  {zh:'交流围绕哪项公开工作或研究问题？你的实际贡献是什么？',en:'Which public work or research question framed the discussion? What was your actual contribution?'},
  {zh:'你如何解释假设、方法和替代方案？哪些观点来自对方的实际追问？',en:'How did you explain the hypotheses, method and alternatives? Which points came from actual follow-up questions?'},
  {zh:'讨论了哪些实验、消融或失败案例？区分观察到的结果与尚未验证的推测。',en:'Which experiments, ablations or failures were discussed? Separate observed results from untested speculation.'},
  {zh:'哪些表达需要更清晰，哪些局限需要补充？只写可以公开的内容。',en:'What could you explain more clearly, and which limitations need work? Include only information you can make public.'},
 ],
 behavioral:[
  {zh:'当时的背景、目标与职责是什么？避免写出不必要的个人或团队识别信息。',en:'What were the context, goal and your responsibilities? Omit unnecessary personal or team-identifying details.'},
  {zh:'具体说明你做了什么、如何沟通和做取舍，区分个人行动与团队成果。',en:'Describe your actions, communication and decisions. Distinguish your contribution from the team result.'},
  {zh:'实际结果、收到的反馈与未解决的问题分别是什么？不确定的结果可以不透露。',en:'What were the actual outcome, feedback and unresolved issues? You can leave uncertain outcomes undisclosed.'},
  {zh:'你学到了什么，下次会怎么做？给读者一条可执行的复盘建议。',en:'What did you learn and what would you do differently? Offer one actionable reflection.'},
 ],
 process:[
  {zh:'按实际顺序记录联系、测评与面试轮次。日期和地点可以只保留大致范围。',en:'Record contacts, assessments and rounds in their actual order. Approximate dates and locations are enough.'},
  {zh:'分别概括每轮的形式与讨论方向，避免把准备题当作实际面试题。',en:'Summarize the format and topics of each round. Keep preparation exercises separate from what actually occurred.'},
  {zh:'记录招聘团队明确告知的后续步骤；等待中或不便透露的结果可以保持未公开。',en:'Record the next steps explicitly communicated by the recruiting team. Pending or private outcomes can remain undisclosed.'},
  {zh:'哪些准备最有帮助？哪些经验只适用于你这次的岗位、团队或时间点？',en:'Which preparation helped most? Which observations are specific to this role, team or point in time?'},
 ],
};
export type InterviewWritingGuideProps={stage:string;defaultId:WritingOutlineId;outlines:InterviewOutlineCatalog;hasBody:boolean;hasContent:boolean;busy:boolean;canUndo:boolean;editedAfterInsert:boolean;onInsert:(id:WritingOutlineId,opener:HTMLElement)=>void;onUndo:(opener:HTMLElement)=>void;onRestoreFocus:()=>void};
export default function InterviewWritingGuide({stage,defaultId,outlines,hasBody,hasContent,busy,canUndo,editedAfterInsert,onInsert,onUndo}:InterviewWritingGuideProps){
 const {t,locale}=useI18n(),language=locale==='en'?'en':'zh';
 const [choice,setChoice]=useState<{stage:string;id:WritingOutlineId}>();
 const id=choice?.stage===stage?choice.id:defaultId,outline=outlines[id];
 return <div className="interview-writing-panel">
  <p>{t('这些问题帮你整理真实经历。插入时只添加小标题，提示不会写进正文。','Use these questions to organize your real experience. Inserting adds section headings only; prompts stay outside your report.')}</p>
  <label><span>{t('选择写作提纲','Choose a writing outline')}</span><select value={id} disabled={busy} onChange={event=>setChoice({stage,id:event.target.value as WritingOutlineId})}>{(Object.keys(outlines) as WritingOutlineId[]).map(value=><option key={value} value={value}>{outlines[value].name[language]}</option>)}</select></label>
  <ol>{outline.sections.map((section,index)=><li key={section.en}><b>{section[language]}</b><p>{prompts[id][index][language]}</p></li>)}</ol>
  <div className="interview-writing-actions"><button className="button subtle" type="button" disabled={busy} onClick={event=>{event.currentTarget.focus({preventScroll:true});onInsert(id,event.currentTarget);}}><Plus size={14}/>{hasContent?t('在光标处插入提纲','Insert outline at cursor'):hasBody?t('更换空提纲','Replace empty outline'):t('插入写作提纲','Insert outline')}</button><button className="text-button" type="button" disabled={busy||!canUndo} onClick={event=>{event.currentTarget.focus({preventScroll:true});onUndo(event.currentTarget);}}><Undo2 size={13}/>{t('撤销本次提纲','Undo outline insertion')}</button></div>
  {editedAfterInsert&&<small role="status">{t('正文已有后续修改，撤销提纲暂不可用，当前文字会保留。','Your text changed after insertion. Outline undo is unavailable so your current writing stays intact.')}</small>}
 </div>;
}
