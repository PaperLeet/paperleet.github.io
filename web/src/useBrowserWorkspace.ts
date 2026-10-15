import { useEffect, useRef, useState } from 'react';
import { useI18n } from './i18n';
import type { BrowserBackup, BrowserOwner } from './browserWorkspaceStore';
import type { BrowserWorkspaceView, createBrowserWorkspace } from './browserWorkspaceController';
import type { DraftExpectation } from './practiceRecords';
import type { Problem, Report, Saved } from './types';

type Runtime = {owner: BrowserOwner; loading: boolean; controller?: ReturnType<typeof createBrowserWorkspace>};
const empty = (owner?: BrowserOwner): BrowserWorkspaceView => ({owner,saved:{},conflicts:{},reviewed:{},backups:[],ready:false,saving:false,failed:false});

/** Keep the library shell small; resolve the local cache before enabling edits. */
export function useBrowserWorkspace(owner?: BrowserOwner) {
  const {t} = useI18n();
  const [view,setView] = useState(() => empty(owner));
  const active = useRef<Runtime | undefined>(undefined), currentOwner = useRef(owner);
  const unsaved = useRef<() => boolean>(() => false);
  currentOwner.current = owner;
  function current(runtime?: Runtime): runtime is Runtime {
    return !!runtime && active.current === runtime && currentOwner.current === runtime.owner;
  }
  async function start(runtime: Runtime) {
    if (!current(runtime) || runtime.loading) return;
    runtime.loading = true;
    try {
      const module = await import('./browserWorkspaceController');
      unsaved.current = module.hasUnsavedDraftSafety;
      if (!current(runtime)) return;
      runtime.controller = module.createBrowserWorkspace(runtime.owner,
        next => { if (current(runtime)) setView(next); }, () => current(runtime));
    } catch {
      if (current(runtime)) setView({...empty(runtime.owner),failed:true});
    } finally {runtime.loading = false;}
  }
  useEffect(() => {
    setView(empty(owner));
    if (!owner) {active.current=undefined;return;}
    const runtime: Runtime = {owner,loading:false}; active.current=runtime;
    void start(runtime);
    return () => {runtime.controller?.dispose();if(active.current===runtime)active.current=undefined;};
  },[owner]);
  useEffect(() => {
    const leave = (event: BeforeUnloadEvent) => {if(unsaved.current()){event.preventDefault();event.returnValue='';}};
    window.addEventListener('beforeunload',leave);return()=>window.removeEventListener('beforeunload',leave);
  },[]);
  const runtime = active.current, visible = view.owner === owner ? view : empty(owner);
  const controller = () => current(runtime) ? runtime.controller : undefined;
  return {...visible,loading:!visible.ready && !visible.failed,pending:visible.saving,
    error:visible.failed ? t('无法保存本机工作区。请下载代码备份，再重试保存。','Could not save the browser workspace. Download a code backup, then retry saving.') : '',
    update:(id:string,patch:Partial<Saved>,expected?:DraftExpectation)=>controller()?.update(id,patch,expected) ?? false,
    applyReport:(problem:Problem,report:Report,expectedOwner:string)=>controller()?.applyReport(problem,report,expectedOwner),
    replaceCode:(problem:Problem,code:string,expected:DraftExpectation)=>controller()?.replaceCode(problem,code,expected) ?? false,
    resetToStarter:(problem:Problem,expected:DraftExpectation)=>controller()?.resetToStarter(problem,expected) ?? false,
    resolveConflict:(id:string,keepLocal:boolean)=>controller()?.resolveConflict(id,keepLocal,visible),
    restoreBackup:(backup:BrowserBackup)=>controller()?.restoreBackup(backup),
    discardBackup:async(backup:BrowserBackup)=>{await controller()?.discardBackup(backup);},
    retry:()=>{if(current(runtime)){if(runtime.controller)runtime.controller.retry();else void start(runtime);}},
  };
}
