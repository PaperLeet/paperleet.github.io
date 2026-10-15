import { useEffect, useRef, useState } from 'react';
import { useAccount } from './social';
import type { Report } from './types';
import type { QueuedJudgeController, QueuedJudgeView } from './queuedJudgeController';
import { submissionModuleLoadError, pendingSubmissionModuleLoadError } from './judgeJobs';

const empty=(owner:string):QueuedJudgeView=>({owner,reconnecting:false,error:''});
function clientId(){
  try {
    const saved=sessionStorage.getItem('paperleet.judge.client.v1');
    if(saved&&/^[a-f0-9]{32}$/.test(saved))return saved;
    const id=crypto.randomUUID().replaceAll('-','');sessionStorage.setItem('paperleet.judge.client.v1',id);return id;
  } catch{return crypto.randomUUID().replaceAll('-','');}
}
type Context={owner:string;disposed:boolean;controller?:QueuedJudgeController;promise:Promise<QueuedJudgeController>};
export function useQueuedJudge(owner:string,ready:boolean,onReport:(report:Report,owner:string)=>void){
  const {health,user}=useAccount(),[client]=useState(clientId),[view,setView]=useState(()=>empty(owner));
  const callback=useRef(onReport);callback.current=onReport;
  const current=useRef({owner,ready});current.current={owner,ready};
  const context=useRef<Context|undefined>(undefined);
  function getController(){
    const previous=context.current;
    if(previous&&!previous.disposed&&previous.owner===owner)return previous.promise;
    const next:Context={owner,disposed:false,promise:undefined!};
    const alive=()=>!next.disposed&&context.current===next&&current.current.owner===owner&&current.current.ready;
    next.promise=import('./queuedJudgeController').catch(()=>{throw new Error(submissionModuleLoadError);}).then(module=>{
      if(!alive())throw new DOMException('Owner changed','AbortError');
      return next.controller=new module.QueuedJudgeController({owner,client,expectedUser:health?.mode==='hosted'?user?.id:undefined,
        alive,onView:setView,onReport:(report,runOwner)=>{if(alive())callback.current(report,runOwner);}});
    });
    context.current=next;return next.promise;
  }
  function fail(error:unknown){
    if(current.current.owner!==owner||!current.current.ready||error instanceof DOMException&&error.name==='AbortError')return;
    if(context.current?.owner===owner){context.current.disposed=true;context.current.controller?.dispose();context.current=undefined;}
    const reloadRequired=error instanceof Error&&error.message===submissionModuleLoadError;
    let pending=false;
    try{pending=!!sessionStorage.getItem(`paperleet.judge.pending.v1:${owner}`);}catch{/* A blocked receipt cannot identify an existing job. */}
    setView({...empty(owner),reloadRequired,error:reloadRequired?(pending?pendingSubmissionModuleLoadError:submissionModuleLoadError):'Server submission could not start. Your draft is preserved; try again.'});
  }
  useEffect(()=>{
    setView(empty(owner));
    if(ready){
      let pending=false;
      try{pending=!!sessionStorage.getItem(`paperleet.judge.pending.v1:${owner}`);}catch{/* No recovery without a receipt. */}
      if(pending)void getController().then(controller=>controller.resume()).catch(fail);
    }
    return()=>{const previous=context.current;if(previous){previous.disposed=true;previous.controller?.dispose();if(context.current===previous)context.current=undefined;}};
  },[owner,ready]);
  const visible=view.owner===owner?view:empty(owner);
  return {...visible,active:!!visible.pending,
    submit:async(problem:string,code:string)=>{if(ready&&current.current.owner===owner&&current.current.ready)try{await(await getController()).submit(problem,code);}catch(error){fail(error);}},
    cancel:()=>current.current.owner===owner?context.current?.controller?.cancel():undefined,
    stopWaiting:()=>current.current.owner===owner&&(context.current?.controller?.stopWaiting()||false)};
}
