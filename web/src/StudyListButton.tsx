import {useEffect,useId,useLayoutEffect,useRef,useState} from 'react';
import type {CSSProperties} from 'react';
import {FolderPlus,X} from 'lucide-react';
import {useAccount} from './social';
import {useI18n} from './i18n';
import {deferredFeature} from './deferredFeature';
import './study-lists.css';
const Picker=deferredFeature<Parameters<typeof import('./StudyLists').ProblemStudyListPicker>[0]>(()=>import('./StudyLists').then(module=>({default:module.ProblemStudyListPicker})),['私人题单','private study lists']);
export default function StudyListButton({problemId,canSave=true}:{problemId:string;canSave?:boolean}){
 const {user,loading,capabilities}=useAccount();
 if(loading||!user||capabilities?.study_lists!==true||!canSave)return null;
 return <Control key={user.id+':'+problemId} problemId={problemId}/>;
}
function Control({problemId}:{problemId:string}){
 const {t}=useI18n(),id=useId(),trigger=useRef<HTMLButtonElement>(null),root=useRef<HTMLDivElement>(null),panel=useRef<HTMLDivElement>(null),[open,setOpen]=useState(false),[visited,setVisited]=useState(false);
 const [placement,setPlacement]=useState<CSSProperties>({});
 useLayoutEffect(()=>{
  if(!open)return;
  const position=(event?:Event)=>{if(event?.type==='scroll'&&event.target instanceof Node&&panel.current?.contains(event.target))return;const rect=trigger.current?.getBoundingClientRect();if(!rect)return;const width=Math.min(380,Math.max(0,window.innerWidth-32)),below=window.innerHeight-rect.bottom-24,above=rect.top-24,useBelow=below>=Math.min(300,above);if(Math.max(below,above)<180){setPlacement({width,left:16,top:16,bottom:16,maxHeight:Math.max(0,window.innerHeight-32)});return;}setPlacement({width,left:Math.max(16,Math.min(rect.right-width,window.innerWidth-width-16)),...(useBelow?{top:rect.bottom+8,bottom:'auto',maxHeight:Math.max(100,below)}:{top:'auto',bottom:window.innerHeight-rect.top+8,maxHeight:Math.max(100,above)})});};
  position();window.addEventListener('resize',position);window.addEventListener('scroll',position,{capture:true,passive:true});return()=>{window.removeEventListener('resize',position);window.removeEventListener('scroll',position,true);};
 },[open]);
 function close(restore=false){if(restore&&panel.current?.contains(document.activeElement))trigger.current?.focus({preventScroll:true});setOpen(false);}
 useEffect(()=>{
  if(!open)return;
  const outside=(event:PointerEvent)=>{if(event.target instanceof Node&&!root.current?.contains(event.target))setOpen(false);},route=()=>close(true);
  const focus=(event:FocusEvent)=>{if(event.target instanceof Node&&!root.current?.contains(event.target))setOpen(false);};
  document.addEventListener('pointerdown',outside,true);document.addEventListener('focusin',focus);window.addEventListener('hashchange',route);window.addEventListener('popstate',route);
  return()=>{document.removeEventListener('pointerdown',outside,true);document.removeEventListener('focusin',focus);window.removeEventListener('hashchange',route);window.removeEventListener('popstate',route);};
 },[open]);
 return <div className="study-list-picker" ref={root}><button ref={trigger} className="problem-favorite-button" type="button" aria-label={t('加入私人题单','Add to a private study list')} title={t('加入私人题单','Add to a private study list')} aria-haspopup="dialog" aria-expanded={open} aria-controls={id} onClick={()=>{if(open)close(true);else{setVisited(true);setOpen(true);}}}><FolderPlus size={16}/></button><div id={id} ref={panel} style={placement} hidden={!open} role="dialog" aria-label={t('加入私人题单','Add to a private study list')} className="study-list-popover" onKeyDown={event=>{
  if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229)return;event.preventDefault();event.stopPropagation();close(true);
 }}>{visited&&<><button type="button" className="study-list-close icon-button" aria-label={t('收起题单选择','Close study-list picker')} onClick={()=>close(true)}><X size={16}/></button><Picker problemId={problemId}/></>}</div></div>;
}
