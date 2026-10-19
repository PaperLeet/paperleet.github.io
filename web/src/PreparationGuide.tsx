import AppLink from './AppLink';
import {useEffect,useId,useRef,useState} from 'react';
import {ArrowRight,BookOpen,ChevronDown,LockKeyhole,RefreshCw} from 'lucide-react';
import {useI18n} from './i18n';
import {hasPreparationGuide,loadPreparationGuide,type PreparationGuide as Guide} from './preparationGuides';
import {keepReadingTriggerVisible} from './readingViewport';
import './preparation-guides.css';

export const PREPARATION_GUIDE_TIMEOUT=12000;
type Props={problemId:string;practiceHref?:string;onNotes?:()=>void;notesUnlocked?:boolean;onLab?:()=>void};
export default function PreparationGuide(props:Props){
 return hasPreparationGuide(props.problemId)?<GuideReader {...props}/>:null;
}
function GuideReader({problemId,practiceHref,onNotes,notesUnlocked,onLab}:Props){
 const {t,locale}=useI18n(),id=useId();
 const root=useRef<HTMLDetailsElement>(null),summary=useRef<HTMLElement>(null),prompts=useRef<HTMLOListElement>(null);
 const [expanded,setExpanded]=useState(false),[attempt,setAttempt]=useState(0);
 const [resource,setResource]=useState<{id:string;attempt:number;phase:'loading'|'ready'|'failed';guide?:Guide}>({id:'',attempt:-1,phase:'loading'});
 const current=resource.id===problemId&&resource.attempt===attempt?resource:undefined;
 const settled=!!current&&current.phase!=='loading';
 const visit=useRef({problemId,attempt});visit.current={problemId,attempt};
 const here=()=>!!root.current?.isConnected&&root.current.open&&visit.current.problemId===problemId&&visit.current.attempt===attempt;
 function focusSummary(){if(root.current?.contains(document.activeElement)&&summary.current){summary.current.focus({preventScroll:true});keepReadingTriggerVisible(summary.current,100);}}
 function close(){if(!here())return;focusSummary();root.current!.open=false;setExpanded(false);}
 function collapseHints(){if(!here())return;prompts.current?.querySelectorAll<HTMLDetailsElement>('.preparation-hint[open]').forEach(hint=>{hint.open=false;});}
 useEffect(()=>{
  if(!expanded||settled)return;
  let active=true;const identity={id:problemId,attempt};
  setResource({...identity,phase:'loading'});
  const timer=setTimeout(()=>{if(active){active=false;setResource({...identity,phase:'failed'});}},PREPARATION_GUIDE_TIMEOUT);
  void loadPreparationGuide(problemId).then(guide=>{
   if(!active)return;active=false;clearTimeout(timer);setResource({...identity,phase:'ready',guide});
  }).catch(()=>{if(active){active=false;clearTimeout(timer);setResource({...identity,phase:'failed'});}});
  return()=>{active=false;clearTimeout(timer);};
 },[problemId,expanded,attempt,settled]);
 return <details id={id} className="preparation-guide" ref={root} onToggle={event=>{if(event.target===event.currentTarget)setExpanded(event.currentTarget.open);}}>
  <summary ref={summary}><span><BookOpen size={16}/><b>{t('面试自测提纲','Interview self-study prompts')}</b><small>{t('讲解 · 实现 · 验证','Explain · implement · verify')}</small></span><ChevronDown size={15}/></summary>
  {(expanded||settled)&&<div className="preparation-guide-body" aria-busy={!current||current.phase==='loading'}>
   <p className="preparation-guide-provenance">{t('根据站内题面整理的学习建议。先尝试自述，再展开提示；自测不计入成绩或公开面经统计。','Study prompts based on the exercise statement. Explain your answer before revealing hints. Self-study does not change grades or public report statistics.')}</p>
   {!current||current.phase==='loading'?<p role="status">{t('正在加载自测提纲…','Loading study prompts…')}</p>:current.phase==='failed'?<div className="preparation-guide-error" role="alert"><p>{t('自测提纲暂时无法加载，题面和代码仍然可用。','Study prompts could not load. The statement and your code remain available.')}</p><button type="button" className="text-button" onClick={()=>{if(!here())return;focusSummary();setAttempt(value=>value+1);}}><RefreshCw size={13}/>{t('重试自测提纲','Retry study prompts')}</button></div>:<ol id={id+'-prompts'} ref={prompts} className="preparation-prompt-list">{current.guide!.prompts.map(prompt=><li key={`${problemId}:${prompt.focus}`}><span className="preparation-prompt-focus">{prompt.focus==='explain'?t('讲清原理','EXPLAIN'):prompt.focus==='implement'?t('推到实现','IMPLEMENT'):t('检查边界','VERIFY')}</span><h4>{prompt.question[locale==='en'?'en':'zh']}</h4><details className="preparation-hint"><summary>{t('展开思路提示','Reveal a thinking hint')}<ChevronDown size={12}/></summary><p>{prompt.hint[locale==='en'?'en':'zh']}</p></details></li>)}</ol>}
   <div className="preparation-guide-actions">{practiceHref&&<AppLink className="button subtle" href={practiceHref}>{t('用代码验证','Validate through practice')}<ArrowRight size={14}/></AppLink>}{onLab&&<button type="button" className="text-button" onClick={()=>{if(here())onLab();}}>{t('在实验台观察','Explore in the lab')}</button>}{onNotes&&<button type="button" className="text-button" disabled={!notesUnlocked} onClick={()=>{if(here()&&notesUnlocked)onNotes();}}>{!notesUnlocked&&<LockKeyhole size={12}/>}<span>{notesUnlocked?t('复盘完整坑点','Review the full notes'):t('通过练习后解锁完整坑点','Pass the exercise to unlock full notes')}</span></button>}<span className="preparation-guide-reading-controls">{current?.phase==='ready'&&<button type="button" className="text-button" aria-controls={id+'-prompts'} onClick={collapseHints}>{t('收起所有提示','Collapse all hints')}</button>}<button type="button" className="text-button" aria-controls={id} onClick={close}>{t('收起提纲','Collapse study prompts')}<ChevronDown size={13} className="preparation-guide-close-icon" aria-hidden="true"/></button></span></div>
  </div>}
 </details>;
}
