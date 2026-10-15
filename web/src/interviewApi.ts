import {useEffect,useState} from 'react';
import {socialApi,errorMessage} from './social';
import type {InterviewModule} from './InterviewFields';
export function useInterviewResource<T>(path:string,enabled=true,retry=0,scope=''):{path:string;data?:T;error?:string;loading:boolean}{
  const [state,setState]=useState<{path:string;scope:string;data?:T;error?:string}>({path:'',scope:''});
  useEffect(()=>{
    if(!enabled)return;
    const controller=new AbortController();
    setState({path,scope});
    void socialApi<T>(path,undefined,controller.signal).then(data=>{if(!controller.signal.aborted)setState({path,scope,data});}).catch(error=>{if(!controller.signal.aborted)setState({path,scope,error:errorMessage(error)});});
    return()=>controller.abort();
  },[path,enabled,retry,scope]);
  return enabled&&state.path===path&&state.scope===scope?{...state,loading:!state.data&&!state.error}:{path,loading:enabled};
}
let modulePromise:Promise<InterviewModule[]>|undefined;
export function loadInterviewModules(){
  if(!modulePromise)modulePromise=fetch(`${import.meta.env.BASE_URL}data/index.json`,{cache:'no-cache'}).then(async response=>{
    if(!response.ok)throw Error('Catalog unavailable');
    const data=await response.json();if(!Array.isArray(data.problems))throw Error('Catalog unavailable');
    return data.problems.filter((p:InterviewModule)=>p&&typeof p.id==='string'&&typeof p.title==='string'&&Number.isSafeInteger(p.number));
  }).catch(error=>{modulePromise=undefined;throw error;});
  return modulePromise;
}
