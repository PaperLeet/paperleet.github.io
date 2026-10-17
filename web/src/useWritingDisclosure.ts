import {useId,useLayoutEffect,useRef,useState,type MouseEvent,type RefObject} from 'react';
import {revealReadingTarget} from './readingViewport';
import './writing-disclosure.css';

export type WritingPanelProps={id:string;ref:RefObject<HTMLElement|null>};
type Opening={scope:string;href:string;opener:HTMLElement;active:Element|null};

/** Only explicit opening/closing intent moves focus; retiring a view never does. */
export function useWritingDisclosure(scope:string){
 const id=useId(),panel=useRef<HTMLElement>(null),currentScope=useRef(scope);currentScope.current=scope;
 const [opening,setOpening]=useState<Opening>();
 const currentOpening=useRef(opening);currentOpening.current=opening;
 const href=globalThis.location?.href;
 const expanded=!!opening&&opening.scope===scope;
 useLayoutEffect(()=>{
  if(!opening)return;
  if(!expanded){setOpening(undefined);return;}
  if(opening.href!==globalThis.location?.href||document.activeElement!==opening.active)return;
  const title=panel.current?.querySelector<HTMLInputElement>('[data-writing-title]');
  if(!title||title.disabled)return;
  title.focus({preventScroll:true});revealReadingTarget(title,'center');
 },[opening,expanded]);
 function open(event:MouseEvent<HTMLElement>){
  if(!href||currentScope.current!==scope||globalThis.location?.href!==href||!event.currentTarget.isConnected)return;
  setOpening({scope,href,opener:event.currentTarget,active:document.activeElement});
 }
 function close(){
  if(currentOpening.current!==opening||currentScope.current!==scope||!expanded||href!==globalThis.location?.href)return;
  setOpening(undefined);
  if(!opening.opener.isConnected||opening.opener.matches(':disabled')||!panel.current?.contains(document.activeElement))return;
  opening.opener.focus({preventScroll:true});revealReadingTarget(opening.opener,'nearest');
 }
 return {expanded,close,triggerProps:{onClick:open,'aria-expanded':expanded,'aria-controls':expanded?id:undefined},panelProps:{id,ref:panel} satisfies WritingPanelProps};
}
