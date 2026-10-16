import {useEffect,useRef,useState} from 'react';

type CopyStatus='idle'|'copying'|'copied'|'failed';

/** Callers key their component by content/owner; locale changes keep its lifetime. */
export function useClipboardCopy() {
  const [status,setStatus]=useState<CopyStatus>('idle'),[value,setValue]=useState<string>();
  const live=useRef(false),busy=useRef(false),request=useRef(0),deadline=useRef<number|undefined>(undefined),surface=useRef('');
  function retire() {
    request.current++;busy.current=false;window.clearTimeout(deadline.current);deadline.current=undefined;
  }
  useEffect(()=>{
    live.current=true;surface.current=location.href;
    return()=>{live.current=false;retire();};
  },[]);
  useEffect(()=>{
    if(status==='idle')return;
    const leave=()=>{
      if(surface.current===location.href)return;
      surface.current=location.href;retire();setStatus('idle');setValue(undefined);
    };
    window.addEventListener('hashchange',leave);window.addEventListener('popstate',leave);
    return()=>{window.removeEventListener('hashchange',leave);window.removeEventListener('popstate',leave);};
  },[status]);
  function copy(prepare:()=>string) {
    if(busy.current)return;
    busy.current=true;setStatus('copying');setValue(undefined);
    const current=++request.current,href=location.href;surface.current=href;
    const settle=(next:CopyStatus)=>{
      if(!live.current||request.current!==current)return;
      retire();surface.current=location.href;
      if(location.href===href)setStatus(next);
      else {setStatus('idle');setValue(undefined);}
    };
    try {
      const prepared=prepare();setValue(prepared);
      deadline.current=window.setTimeout(()=>settle('failed'),8000);
      Promise.resolve(navigator.clipboard.writeText(prepared)).then(()=>settle('copied'),()=>settle('failed'));
    }catch {settle('failed');}
  }
  return {status,value,copy};
}
