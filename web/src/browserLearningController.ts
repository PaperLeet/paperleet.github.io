import { browserNotifications } from './browserNotifications';
import { learningKey, mergeBrowserLearning, type BrowserLearning } from './browserLearning';
import { loadBrowserLearning, saveBrowserLearning, dismissBrowserLearning } from './browserLearningStore';
import type { Problem, Report } from './types';

export type LearningCommand = {type:'report';problem:Problem;report:Report;version:string} | {type:'dismiss';id:string;suite:string;expected:string;version:string};
export type LearningView = {owner:string;records:Record<string,BrowserLearning>;ready:boolean;failed:boolean;saving:boolean};
export function createLearningController(owner:string, records:Record<string,BrowserLearning>, changed:(view:LearningView)=>void) {
  let view:LearningView={owner,records,ready:false,failed:false,saving:false},revision=0,processing=false,failed=false,disposed=false;
  const pending:LearningCommand[]=[];
  function change(patch:Partial<LearningView>){view={...view,...patch};if(!disposed)changed(view);}
  async function refresh(){
    const generation=revision,records=await loadBrowserLearning(owner);
    if(!disposed&&generation===revision&&!pending.length)change({records,ready:true});
  }
  async function drain(){
    if(processing||failed)return;processing=true;change({saving:true});
    try{
      while(pending.length){
        const command=pending[0];
        if(command.type==='report')await saveBrowserLearning(owner,command.problem,command.report,command.version);
        else await dismissBrowserLearning(owner,command.id,command.suite,command.expected,command.version);
        pending.shift();revision++;notifications.notify();
      }
      change({failed:false,ready:true});
    }catch{failed=true;change({failed:true,ready:true});}
    finally{processing=false;change({saving:false});}
    if(!disposed&&!failed)try{await refresh();}catch{change({failed:true,ready:true});}
  }
  function enqueue(command:LearningCommand){
    revision++;pending.push(command);failed=false;
    const id=command.type==='report'?command.problem.id:command.id;
    const key=learningKey(id,command.type==='report'?command.problem.browser_judge?.bundle_url || '':command.suite),row=view.records[key];
    if(command.type==='report')change({records:{...view.records,[key]:mergeBrowserLearning(row,owner,command.problem,command.report,command.version)},ready:true});
    else if(row?.version===command.expected)change({records:{...view.records,[key]:{...row,report:undefined,version:command.version}}});
    void drain();
  }
  const reload=()=>{void refresh().catch(()=>change({failed:true,ready:true}));};
  const notifications=browserNotifications('paperleet.learning.v1',owner,reload);reload();
  return {enqueue,retry:()=>{failed=false;if(pending.length)void drain();else void refresh().then(()=>change({failed:false})).catch(()=>change({failed:true,ready:true}));},
    dispose:()=>{disposed=true;notifications.dispose();}};
}
