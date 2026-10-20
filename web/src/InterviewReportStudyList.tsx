import {useId,useRef,useState} from 'react';
import {FolderPlus,ChevronDown} from 'lucide-react';
import {useAccount} from './social';
import {useI18n} from './i18n';
import {deferredFeature} from './deferredFeature';
import {activeSourcePreview} from './sourcePreviewKeyboard';
import {captureReportStudyList,reportStudyListSurface} from './reportStudyList';
import {appLocation} from './appLocation';
import {hostedAppAvailable} from './deployment';
import type {ProblemTopic} from './communityTopics';
import './interview-report-study-list.css';

function SaveUnavailable({canAct,retry}:{canAct?:()=>boolean;retry:()=>void}){
 const {t}=useI18n();return <div role="alert"><p>{t('题单表单暂时无法加载，面经和练习仍可使用。','The list form could not load. Reading and practice remain available.')}</p><button type="button" className="text-button" onClick={event=>{if(!canAct?.()||!activeSourcePreview(event.currentTarget))return;if(document.activeElement===event.currentTarget)event.currentTarget.closest('.interview-report-study-list')?.querySelector<HTMLButtonElement>('.interview-report-save-trigger')?.focus({preventScroll:true});retry();}}>{t('重试题单表单','Retry list form')}</button></div>;
}
const SaveList=hostedAppAvailable?deferredFeature(()=>import('./InterviewBatchSave'),['私人题单','private study list'],{failure:(props,retry)=><SaveUnavailable canAct={props.canAct} retry={retry}/>}):()=>null;
type Props={questions:string[];topics:ProblemTopic[];returnTo:string};
export default function InterviewReportStudyList({questions,topics,returnTo}:Props){
 const {user,loading,capabilities}=useAccount();
 if(!hostedAppAvailable||loading||!user||capabilities?.study_lists!==true||!questions.length||typeof location!=='undefined'&&!reportStudyListSurface(appLocation.hash,returnTo))return null;
 const capture=captureReportStudyList(questions,topics);
 const source=topics.filter(topic=>questions.includes(topic.id)).map(topic=>[topic.id,topic.number,topic.title,topic.title_en,topic.retired]);
 return <ReportStudyList key={JSON.stringify([user.id,returnTo,questions,source])} capture={capture} owner={user.id} returnTo={returnTo}/>;
}
function ReportStudyList({capture,owner,returnTo}:{capture:ReturnType<typeof captureReportStudyList>;owner:string;returnTo:string}){
 const {t}=useI18n(),{user,loading,capabilities}=useAccount(),id=useId(),root=useRef<HTMLDivElement>(null),trigger=useRef<HTMLButtonElement>(null),panel=useRef<HTMLElement>(null),composing=useRef(false);
 const [open,setOpen]=useState(false),[visited,setVisited]=useState(false);
 function here(){return activeSourcePreview(root.current)&&reportStudyListSurface(appLocation.hash,returnTo)&&!loading&&user?.id===owner&&capabilities?.study_lists===true;}
 function close(){if(!here())return;if(panel.current?.contains(document.activeElement))trigger.current?.focus({preventScroll:true});setOpen(false);}
 return <div ref={root} className="interview-report-study-list">
  <button ref={trigger} type="button" className="text-button interview-report-save-trigger" aria-expanded={open} aria-controls={id} aria-describedby={!capture?id+'-help':undefined} disabled={!capture} onClick={()=>{if(!here()||!capture)return;if(open)close();else{setVisited(true);setOpen(true);}}}><FolderPlus size={15} aria-hidden="true"/>{open?t('收起题单保存','Close list saving'):t('保存到私人题单','Save to a private list')}<ChevronDown size={14} aria-hidden="true"/></button>
  {!capture&&<p id={id+'-help'}>{t('部分配套练习尚不可用或信息不完整，暂不能保存完整题单。','Some linked exercises are unavailable or unconfirmed. The complete set cannot be saved yet.')}</p>}
  <section ref={panel} id={id} hidden={!open} className="interview-report-save-panel" onCompositionStart={()=>{composing.current=true;}} onCompositionEnd={()=>{composing.current=false;}} onKeyDown={event=>{
   if(!open||!here()||!activeSourcePreview(event.currentTarget)||event.key!=='Escape'||event.defaultPrevented||event.repeat||composing.current||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||(event.target as HTMLElement).closest('select'))return;
   if(!event.currentTarget.contains(document.activeElement))return;
   event.preventDefault();event.stopPropagation();close();
  }}>{visited&&capture&&<SaveList capture={capture} modules={capture.modules} initialTitle={t('面经配套练习','Experience practice')} canAct={()=>open&&here()}/>}</section>
 </div>;
}
