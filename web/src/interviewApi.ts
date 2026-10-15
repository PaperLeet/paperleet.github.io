import {useEffect,useState} from 'react';
import {socialApi,errorMessage} from './social';
import {useI18n} from './i18n';
import type {InterviewModule} from './InterviewFields';
export const INTERVIEW_REQUEST_TIMEOUT=15000;
export function useInterviewResource<T>(path:string,enabled=true,retry=0,scope=''):{path:string;data?:T;error?:string;loading:boolean}{
  const {t}=useI18n();
  const [state,setState]=useState<{path:string;scope:string;attempt:number;phase:'loading'|'ready'|'failed';data?:T;error?:string;reason?:'timeout'|'invalid'}>({path:'',scope:'',attempt:-1,phase:'loading'});
  useEffect(()=>{
    if(!enabled)return;
    const controller=new AbortController();
    let settled=false;
    const identity={path,scope,attempt:retry};
    setState({...identity,phase:'loading'});
    const timer=setTimeout(()=>{
      if(settled||controller.signal.aborted)return;
      settled=true;
      setState({...identity,phase:'failed',reason:'timeout'});
      controller.abort();
    },INTERVIEW_REQUEST_TIMEOUT);
    void socialApi<T>(path,undefined,controller.signal).then(data=>{
      if(settled)return;
      settled=true;
      clearTimeout(timer);
      if(controller.signal.aborted)return;
      if(data===null||typeof data!=='object')setState({...identity,phase:'failed',reason:'invalid'});
      else setState({...identity,phase:'ready',data});
    }).catch(error=>{
      if(settled)return;
      settled=true;
      clearTimeout(timer);
      if(!controller.signal.aborted)setState({...identity,phase:'failed',error:errorMessage(error)});
    });
    return()=>{settled=true;clearTimeout(timer);controller.abort();};
  },[path,enabled,retry,scope]);
  if(!enabled||state.path!==path||state.scope!==scope||state.attempt!==retry)return {path,loading:enabled};
  const error=state.reason==='timeout'?t('加载超时，请重试。当前筛选会保留。','Loading timed out. Retry without losing your filters.'):state.reason==='invalid'?t('暂时无法读取返回的数据，请重试。','The response could not be read. Try again.'):state.error;
  return {path,data:state.data,error,loading:state.phase==='loading'};
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
