import {Plus,Undo2} from 'lucide-react';
import {useI18n} from './i18n';
import type {CommunityOutline,CommunityOutlineId} from './communityWriting';

type Text={zh:string;en:string};
const prompts:Record<CommunityOutlineId,readonly Text[]>={
 papers:[
  {zh:'指出你想理解的具体结论、公式或设计选择，让读者知道讨论的重点。',en:'Name the conclusion, equation or design choice you want to understand so readers can follow the question.'},
  {zh:'写出对应章节、公式编号或图表；区分作者的结论和你自己的推测。',en:'Cite the section, equation or figure. Separate the authors’ conclusions from your own interpretation.'},
  {zh:'描述你如何推导或检验这个观点。可以写出假设、简单例子和仍不确定的部分。',en:'Explain how you derived or checked the idea, including assumptions, a small example and what remains uncertain.'},
  {zh:'提出一个可以回答的问题，或给出你希望比较的另一种解释。',en:'Ask an answerable question or describe an alternative explanation you would like to compare.'},
 ],
 implementation:[
  {zh:'说明要复现的算法、公式和输入输出约定，附上公开的参考位置。',en:'Describe the algorithm, equation and input/output contract, with a public reference.'},
  {zh:'提供能说明问题的最小代码、输入形状、随机种子与必要的运行设置。',en:'Include minimal code, input shapes, random seeds and the runtime settings needed to understand the result.'},
  {zh:'列出实际测得的数值或现象，并说明它和参考结果在哪些条件下不同。',en:'Report measured values or behavior and the conditions under which they differ from the reference.'},
  {zh:'记录验证过的假设和排查结果，帮助读者接着分析，而不是从头猜测。',en:'Record the hypotheses and checks you have already tried so others can build on your investigation.'},
 ],
 questions:[
  {zh:'说明你正在做什么，以及问题出现在哪一步。有关联题目时可从题目讨论区发帖。',en:'Explain what you are working on and where you got stuck. Use a module’s discussion board when the question concerns that exercise.'},
  {zh:'写出你期待的输出、公式关系或程序行为，以及这个预期的依据。',en:'Describe the expected output, mathematical relation or behavior and why you expect it.'},
  {zh:'给出最小输入、实际输出和关键报错。只粘贴你希望公开的内容。',en:'Share a minimal input, actual output and the relevant error. Include only the material you intend to publish.'},
  {zh:'记录你已经尝试的方法、结果，以及目前最不确定的一点。',en:'Summarize your attempts, their results and the part you are still most unsure about.'},
 ],
 career:[
  {zh:'介绍与问题有关的经历、目标和时间范围，不必写出无关的个人信息。',en:'Share the experience, goal and timeframe relevant to the question, without unrelated personal details.'},
  {zh:'概括正在比较的方向，以及地点、时间或学习投入方面的约束。',en:'Describe the options you are comparing and any location, time or learning constraints.'},
  {zh:'区分亲身经历、公开信息和个人判断，说明每个选项的主要取舍。',en:'Separate first-hand experience, public information and personal judgment, and explain the main trade-offs.'},
  {zh:'告诉读者你最需要哪类建议，以及还有哪些信息没有确定。',en:'Tell readers which advice would help most and which details remain undecided.'},
 ],
 lounge:[
  {zh:'用几句话交代这次分享的起点，让第一次看到的人也能加入。',en:'Introduce what prompted the topic so someone reading it for the first time can join in.'},
  {zh:'分享一项具体经验、资源或想法，并附上方便读者继续了解的公开线索。',en:'Share a specific experience, resource or idea, with a public reference readers can explore.'},
  {zh:'说说它对你有什么帮助、有哪些局限，或与你之前的理解有什么不同。',en:'Explain how it helped, its limitations or how it differs from your earlier understanding.'},
  {zh:'留下一两个讨论方向，邀请大家补充经验或不同看法。',en:'Offer one or two discussion questions and invite others to contribute different experiences.'},
 ],
};
export type CommunityWritingGuideProps={category:CommunityOutlineId;outline:CommunityOutline;hasBody:boolean;hasContent:boolean;busy:boolean;canUndo:boolean;editedAfterInsert:boolean;onInsert:(id:CommunityOutlineId,opener:HTMLElement)=>void;onUndo:(opener:HTMLElement)=>void;onRestoreFocus:()=>void};
export default function CommunityWritingGuide({category,outline,hasBody,hasContent,busy,canUndo,editedAfterInsert,onInsert,onUndo}:CommunityWritingGuideProps){
 const {t,locale}=useI18n(),language=locale==='en'?'en':'zh';
 return <div className="community-writing-panel">
  <p>{t('这些提示帮助组织内容。只有标题会插入正文，下面的问题不会自动发布，也不会改写已有文字。','These prompts help organize your post. Only headings are inserted; the questions below are never posted automatically and your writing stays intact.')}</p>
  <ol>{outline.sections.map((section,index)=><li key={section.en}><b><span aria-hidden="true">{index+1}.</span> {section[language]}</b><p>{prompts[category][index][language]}</p></li>)}</ol>
  {category==='career'&&<p>{t('分享具体面试经历时，可以在面经区关联公司和题目。','For a specific interview experience, use the interview section to link companies and exercises.')} <a href="#/interviews">{t('前往面经区','Go to interviews')}</a></p>}
  <div className="community-writing-actions"><button type="button" className="button subtle" disabled={busy} onClick={event=>{event.currentTarget.focus({preventScroll:true});onInsert(category,event.currentTarget);}}><Plus size={14}/>{hasContent?t('在光标处插入提纲','Insert outline at cursor'):hasBody?t('更换空提纲','Replace empty outline'):t('插入写作提纲','Insert outline')}</button><button type="button" className="text-button" disabled={busy||!canUndo} onClick={event=>{event.currentTarget.focus({preventScroll:true});onUndo(event.currentTarget);}}><Undo2 size={14}/>{t('撤销提纲','Undo outline')}</button></div>
  {editedAfterInsert&&<small role="status">{t('正文已继续修改，撤销提纲暂不可用，以保留你的新内容。','You edited the post after inserting an outline. Undo is unavailable to keep your new writing.')}</small>}
 </div>;
}
