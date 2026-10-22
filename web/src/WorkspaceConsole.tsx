import {useEffect,type ReactNode} from 'react';
import {ChevronDown,ChevronUp,SquareCheck} from 'lucide-react';
import {useI18n} from './i18n';
import {navigateTabs} from './tabNavigation';
import WorkspacePanelIcon from './WorkspacePanelIcon';

export type ConsoleTab='testcase'|'result';
type Dock={tab:ConsoleTab;open:boolean};
type Props={dock:Dock;onDock:(dock:Dock)=>void;resultState?:string;testcase:ReactNode;children:ReactNode};

/** LeetCode-style console under the editor. The parent owns the tab and open state so runs can reveal Test Result;
    `children` is the parent's results panel, which keeps its id, label and visibility. Collapsing leaves the tab band. */
export default function WorkspaceConsole({dock,onDock,resultState,testcase,children}:Props){
 const {t}=useI18n(),label=t('控制台','Console'),{tab,open}=dock;
 const onSelect=(next:ConsoleTab)=>onDock({tab:next,open:true}),onToggle=()=>onDock({tab,open:!open});
 // App reads this key at startup without importing the deferred console.
 useEffect(()=>{try{localStorage.setItem('paperleet.layout.console.open',String(open));}catch{/* The console still opens and closes without storage. */}},[open]);
 const panel=tab==='result'?'problem-results-panel':'problem-testcase-panel';
 return <section className="workspace-console" id="workspace-console" aria-label={label} data-open={open||undefined}>
  <div className="console-header" onDoubleClick={event=>{if(!(event.target as Element).closest('button'))onToggle();}}>
   <div className="pane-tabs console-tabs" role="tablist" aria-label={label} onKeyDown={navigateTabs}>
    <button id="console-tab-testcase" type="button" role="tab" tabIndex={tab==='testcase'?0:-1} aria-selected={tab==='testcase'} aria-controls="problem-testcase-panel" onClick={()=>onSelect('testcase')}><SquareCheck size={14} aria-hidden="true"/>{t('测试用例','Testcase')}</button>
    <button id="console-tab-result" type="button" role="tab" tabIndex={tab==='result'?0:-1} aria-selected={tab==='result'} aria-controls="problem-results-panel" data-state={resultState} onClick={()=>onSelect('result')}><WorkspacePanelIcon name="terminal" size={14}/>{t('测试结果','Test Result')}</button>
   </div>
   <button className="console-toggle" type="button" aria-expanded={open} aria-controls={panel} title={open?t('收起控制台','Collapse console'):t('展开控制台','Expand console')} onClick={onToggle}>
    {open?<ChevronDown size={15} aria-hidden="true"/>:<ChevronUp size={15} aria-hidden="true"/>}{label}
   </button>
  </div>
  <div className="console-panel" id="problem-testcase-panel" role="tabpanel" aria-labelledby="console-tab-testcase" tabIndex={0} hidden={!open||tab!=='testcase'}>{testcase}</div>
  {children}
 </section>;
}
