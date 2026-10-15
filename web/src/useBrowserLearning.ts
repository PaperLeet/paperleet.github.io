import { useEffect, useRef, useState } from 'react';
import { learningKey, learningOwner, mergeBrowserLearning } from './browserLearning';
import type { createLearningController, LearningCommand, LearningView } from './browserLearningController';
import type { Problem, Report } from './types';

type Runtime={owner:string;pending:LearningCommand[];controller?:ReturnType<typeof createLearningController>};
/** Guest feedback is persisted with its draft by useGuestWorkspace. */
export const persistsBrowserLearning = (owner:string) => owner.startsWith('user:') && learningOwner(owner);
const empty=(owner:string):LearningView=>({owner,records:{},ready:!persistsBrowserLearning(owner),failed:false,saving:false});
export function useBrowserLearning(owner:string) {
  const [view,setView]=useState(()=>empty(owner));
  const latest=useRef(view),active=useRef<Runtime|undefined>(undefined),currentOwner=useRef(owner);currentOwner.current=owner;
  function current(runtime:Runtime){return active.current===runtime&&currentOwner.current===runtime.owner;}
  function change(runtime:Runtime,view:LearningView){if(current(runtime)){latest.current=view;setView(view);}}
  async function start(runtime:Runtime){
    try{
      const {createLearningController}=await import('./browserLearningController');
      if(!current(runtime))return;
      runtime.controller=createLearningController(runtime.owner,latest.current.records,next=>change(runtime,next));
      for(const command of runtime.pending.splice(0))runtime.controller.enqueue(command);
    }catch{change(runtime,{...latest.current,failed:true,ready:true});}
  }
  useEffect(()=>{
    latest.current=empty(owner);setView(latest.current);
    if(!persistsBrowserLearning(owner)){active.current=undefined;return;}
    const runtime:Runtime={owner,pending:[]};active.current=runtime;void start(runtime);
    return()=>{runtime.controller?.dispose();if(active.current===runtime)active.current=undefined;};
  },[owner]);
  function record(problem:Problem,report:Report,expectedOwner:string){
    const runtime=active.current;if(!runtime||!current(runtime)||runtime.owner!==expectedOwner)return;
    const command:LearningCommand={type:'report',problem,report,version:crypto.randomUUID()};
    if(runtime.controller)runtime.controller.enqueue(command);
    else{
      runtime.pending.push(command);
      const key=learningKey(problem.id,problem.browser_judge?.bundle_url || ''),row=mergeBrowserLearning(latest.current.records[key],runtime.owner,problem,report,command.version);
      change(runtime,{...latest.current,records:{...latest.current.records,[key]:row},ready:true});
    }
  }
  function dismiss(problem:Problem,expectedOwner:string){
    const suite=problem.browser_judge?.bundle_url || '',key=learningKey(problem.id,suite),runtime=active.current,row=latest.current.records[key];if(!runtime||!current(runtime)||runtime.owner!==expectedOwner||!row?.report)return;
    const command:LearningCommand={type:'dismiss',id:problem.id,suite,expected:row.version,version:crypto.randomUUID()};
    if(runtime.controller)runtime.controller.enqueue(command);
    else{runtime.pending.push(command);change(runtime,{...latest.current,records:{...latest.current.records,[key]:{...row,report:undefined,version:command.version}}});}
  }
  return {...(view.owner===owner?view:empty(owner)),record,dismiss,retry:()=>{const runtime=active.current;if(runtime&&current(runtime)){if(runtime.controller)runtime.controller.retry();else void start(runtime);}}};
}
