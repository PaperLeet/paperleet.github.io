import {useId,useRef} from 'react';
import {ChevronDown,ChevronUp} from 'lucide-react';
import AppLink from './AppLink';
import {appLocation} from './appLocation';
import {deferredFeature} from './deferredFeature';
import {useI18n} from './i18n';
import {libraryExercisePreviewScope,type LibraryExerciseSourceScope} from './interviewSourcePreviewModel';
import {keepReadingTriggerVisible} from './readingViewport';

type ReportsProps=Parameters<typeof import('./InterviewSourcePreview').InterviewSourceReports>[0];
function SourceFailure({retry,library,sourceHref}:Pick<ReportsProps,'library'|'sourceHref'>&{retry:()=>void}){
 const {t}=useI18n();
 return <p role="status">{t('来源预览暂时无法加载，仍可打开全部面经。','Source preview could not load. You can still open all experiences.')} <button type="button" className="text-button" onClick={event=>{
  const panel=event.currentTarget.closest<HTMLElement>('.library-exercise-source-preview');
  if(!panel?.isConnected||panel.closest('[hidden],[aria-hidden="true"]')||!library?.exercise||libraryExercisePreviewScope(appLocation.hash,sourceHref,library.exercise)!==library.href)return;
  panel.querySelector<HTMLElement>('h4')?.focus({preventScroll:true});retry();
 }}>{t('重试预览','Retry preview')}</button></p>;
}
const Sources=deferredFeature<ReportsProps>(()=>import('./InterviewSourcePreview').then(module=>({default:module.InterviewSourceReports})),['来源面经','source experiences'],{failure:(props,retry)=><SourceFailure {...props} retry={retry}/>});

type Props={cohort:LibraryExerciseSourceScope;sourceHref:string;returnTo:string;owner:string;name:string;label:string;reports:number;contributors:number;open:boolean;here:()=>boolean|undefined;onOpen:()=>void;onClose:()=>void};
export default function LibraryExerciseSourcePreview({cohort,sourceHref,returnTo,owner,name,label,reports,contributors,open,here,onOpen,onClose}:Props){
 const {t}=useI18n(),id=useId(),toggle=useRef<HTMLButtonElement>(null),panel=useRef<HTMLElement>(null);
 const current=()=>!!here()&&libraryExercisePreviewScope(appLocation.hash,sourceHref,cohort)===returnTo;
 const close=()=>{if(!open||!current())return;if(panel.current?.contains(document.activeElement)&&toggle.current){toggle.current.focus({preventScroll:true});keepReadingTriggerVisible(toggle.current,100);}onClose();};
 return <>
  <div className="library-exercise-source-actions"><button ref={toggle} type="button" className="library-exercise-source-toggle" aria-controls={id} aria-expanded={open} aria-label={open?t(`收起${name}的面经预览`,`Collapse source preview for ${name}`):t(`预览${name}的来源面经`,`Preview source experiences for ${name}`)} onClick={()=>{if(current())open?close():onOpen();}}>{open?<ChevronUp size={14} aria-hidden="true"/>:<ChevronDown size={14} aria-hidden="true"/>}{open?t('收起预览','Collapse preview'):t('预览来源','Preview sources')}</button><AppLink className="library-exercise-company-source" href={sourceHref} target="_blank" rel="noopener noreferrer" onClick={event=>{if(!current())event.preventDefault();}} aria-label={t(`在新标签页阅读${name}的全部来源面经`,`Read all source experiences for ${name} in a new tab`)}>{t('全部面经','All experiences')}</AppLink></div>
  <section id={id} ref={panel} hidden={!open} className="library-exercise-source-preview interview-source-preview" aria-label={t(`${name}的来源预览`,`Source preview for ${name}`)} onKeyDown={event=>{
   if(!open||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||(event.target as HTMLElement).closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])'))return;
   event.preventDefault();event.stopPropagation();close();
  }}>{open&&<><h4 tabIndex={-1}>{t(`${label} · 来源面经`,`${label} · Source experiences`)}</h4><Sources concept={cohort.concept} name={name} reports={reports} contributors={contributors} sourceHref={sourceHref} returnTo={sourceHref} owner={owner} library={{href:returnTo,exercise:cohort}}/><footer><button type="button" className="text-button" onClick={close}><ChevronUp size={14} aria-hidden="true"/>{t('收起面经预览','Close source preview')}</button></footer></>}</section>
 </>;
}
