import {useLayoutEffect,useRef} from 'react';

/** A short-lived explicit navigation yields to the next user action or workspace. */
export function useWorkspaceIntent(scope:string) {
  const latest=useRef({scope,live:false,cancel:()=>{}});
  useLayoutEffect(()=>{
    latest.current.live=true;
    return()=>{latest.current.live=false;latest.current.cancel();};
  },[]);
  useLayoutEffect(()=>{latest.current.cancel();latest.current.scope=scope;},[scope]);
  function create() {
    if(!latest.current.live||latest.current.scope!==scope)return()=>false;
    latest.current.cancel();
    if(document.visibilityState==='hidden')return()=>false;
    const href=location.href,capturedScope=latest.current.scope;
    let opener=document.activeElement,retired=false;
    const events=['pointerdown','click','keydown','beforeinput','input','compositionstart','wheel','touchstart','touchmove'] as const;
    const stop=()=>{
      retired=true;opener=null;window.clearTimeout(deadline);
      if(latest.current.cancel===stop)latest.current.cancel=()=>{};
      for(const event of events)document.removeEventListener(event,stop,true);
      document.removeEventListener('focusin',focus,true);
      document.removeEventListener('visibilitychange',visibility);
      window.removeEventListener('hashchange',route);window.removeEventListener('popstate',route);window.removeEventListener('blur',stop);
    };
    // Hiding a mobile pane can yield body focus without a newer user action.
    const neutralBody=(target:EventTarget|null)=>target===document.body&&opener!==null&&(!opener.isConnected||opener.getClientRects().length===0);
    const focus=(event:FocusEvent)=>{if(event.target!==opener&&!neutralBody(event.target))stop();};
    const route=()=>{if(location.href!==href)stop();};
    const visibility=()=>{if(document.visibilityState==='hidden')stop();};
    const deadline=window.setTimeout(stop,8000);
    latest.current.cancel=stop;
    for(const event of events)document.addEventListener(event,stop,{capture:true,passive:true});
    document.addEventListener('focusin',focus,true);
    document.addEventListener('visibilitychange',visibility);
    window.addEventListener('hashchange',route);window.addEventListener('popstate',route);window.addEventListener('blur',stop);
    return()=>{
      const active=document.activeElement;
      const allowed=!retired&&latest.current.live&&latest.current.scope===capturedScope&&location.href===href&&document.visibilityState!=='hidden'&&
        (active===opener||neutralBody(active));
      stop();return allowed;
    };
  }
  return {create,cancel:()=>{if(latest.current.scope===scope)latest.current.cancel();},isCurrent:()=>latest.current.live&&latest.current.scope===scope};
}
