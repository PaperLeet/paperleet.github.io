import {useId,useLayoutEffect,useRef,useState,type RefObject} from 'react';
import {controlIsActive} from './activeControls';
import {revealReadingTarget} from './readingViewport';
import {useWorkspaceIntent} from './useWorkspaceIntent';

type Selection={start:number;end:number;direction:HTMLTextAreaElement['selectionDirection'];scroll:number;left:number};
type View={scope:string;value:string;field:HTMLTextAreaElement;selection:Selection};
type Request=View&{version:string|undefined;root:HTMLElement;claim:()=>boolean};
type Props={scope:string;version:string|undefined;value:string;preview:boolean;blocked:boolean;field:RefObject<HTMLTextAreaElement|null>;onPreviewChange:(next:boolean)=>void;canAct:()=>boolean;canNavigate?:()=>boolean};

/** Explicit version review returns to the same private source, yielding to newer intent. */
export function useEditReviewNavigation({scope,version,value,preview,blocked,field,onPreviewChange,canAct,canNavigate}:Props){
 const review=useRef<HTMLDetailsElement>(null),id=useId(),hintId=useId(),href=useRef(typeof location==='undefined'?'':location.href).current;
 const view=useRef<View>(undefined),pending=useRef<Request>(undefined),[tick,setTick]=useState(0);
 const intent=useWorkspaceIntent(JSON.stringify([scope,version]));
 const ownsView=()=>!blocked&&canAct()&&intent.isCurrent()&&document.visibilityState!=='hidden'&&(canNavigate?canNavigate():location.href===href);
 const root=()=>field.current?.closest<HTMLElement>('.community-composer');
 function active(opener:HTMLElement){const region=root();return ownsView()&&!!region&&controlIsActive(region)&&controlIsActive(opener)&&opener.closest('.community-composer')===region&&!!review.current&&controlIsActive(review.current);}
 function capture():View|undefined{
  const node=field.current;if(!node||node.value!==value.replace(/\r\n?/g,'\n'))return;
  return {scope,value,field:node,selection:{start:node.selectionStart,end:node.selectionEnd,direction:node.selectionDirection,scroll:node.scrollTop,left:node.scrollLeft}};
 }
 function changePreview(next:boolean){
  if(next&&!preview&&!blocked)view.current=capture();
  onPreviewChange(next);
 }
 function openReview(opener:HTMLButtonElement){
  if(!active(opener))return;
  if(!preview)view.current=capture();
  const details=review.current!,summary=details.querySelector('summary');if(!summary)return;
  details.open=true;summary.focus({preventScroll:true});revealReadingTarget(summary,'start');
 }
 function acceptReview(opener:HTMLButtonElement,onAccept:()=>void){
  if(pending.current||!active(opener))return;
  const current=!preview?capture():view.current?.scope===scope&&view.current.value===value&&view.current.field===field.current?view.current:capture();
  const region=root();if(!current||!region)return;
  opener.focus({preventScroll:true});pending.current={...current,version,root:region,claim:intent.create()};
  onAccept();onPreviewChange(false);setTick(n=>n+1);
 }
 useLayoutEffect(()=>{
  if(view.current?.scope!==scope)view.current=undefined;
  const request=pending.current;if(!request)return;
  if(request.scope!==scope||request.version!==version||request.value!==value||!ownsView()){
   pending.current=undefined;request.claim();return;
  }
  if(preview)return;
  pending.current=undefined;const allowed=request.claim(),node=field.current;
  if(!allowed||!node||node!==request.field||node.disabled||node.readOnly||!controlIsActive(node)||root()!==request.root||!controlIsActive(request.root)||node.value!==value.replace(/\r\n?/g,'\n'))return;
  node.focus({preventScroll:true});node.setSelectionRange(request.selection.start,request.selection.end,request.selection.direction);
  node.scrollTop=request.selection.scroll;node.scrollLeft=request.selection.left;view.current={scope,value,field:node,selection:request.selection};
  revealReadingTarget(node,'nearest');
 },[tick,preview,scope,version,value,blocked]);
 return {review,id,hintId,openReview,acceptReview,changePreview};
}
