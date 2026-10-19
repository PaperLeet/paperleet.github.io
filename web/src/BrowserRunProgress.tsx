import {BrowserRunProgressFrame} from './BrowserRunProgressFrame';
export {BrowserRunProgressFrame} from './BrowserRunProgressFrame';
import {useI18n} from './i18n';
import {deferredFeature} from './deferredFeature';
import type {BrowserProgress} from './browserProgress';

const Details=deferredFeature(()=>import('./BrowserRunProgressDetails'),['',''],{
  loading:props=><ProgressFallback progress={props.progress}/>,
  failure:(props,retry)=><ProgressFallback progress={props.progress} retry={retry}/>,
});

export function BrowserRunProgressBody({progress}:{progress:BrowserProgress}){return <Details progress={progress}/>;}

/** The stop control stays mounted while optional detail rendering loads or retries. */
export function BrowserRunProgress({progress,onStop}:{progress:BrowserProgress;onStop:()=>void}){
  return <BrowserRunProgressFrame onStop={onStop}><BrowserRunProgressBody progress={progress}/></BrowserRunProgressFrame>;
}
function ProgressFallback({progress,retry}:{progress:BrowserProgress;retry?:()=>void}){
  const {t}=useI18n(),hasCount=progress.completed!==undefined&&progress.total!==undefined;
  const phase=progress.stage==='running'?hasCount?t('已完成测试','Tests completed'):t('正在执行测试…','Running tests…'):progress.stage==='calibrating'?progress.step==='mutants'?t('校验错误实现','Checking mutation cases'):progress.step==='reference'?t('校验参考测试','Checking reference tests'):t('正在校验判题…','Validating judging…'):t('正在准备运行环境…','Preparing the runtime…');
  return <div className="browser-progress-fallback">
    <div role="status" aria-live="polite" aria-atomic="true"><p>{phase}{hasCount&&` · ${progress.completed} / ${progress.total}`}</p><progress aria-label={t('当前阶段进度','Current stage progress')} value={hasCount?progress.completed:undefined} max={hasCount?progress.total:undefined}/></div>
    <p className="browser-progress-recovery" role={retry?'alert':undefined}>{retry?t('详细进度暂不可用，判题仍在继续。','Detailed progress is unavailable. Judging is still running.'):t('正在加载详细进度…','Loading detailed progress…')}</p>
    {retry&&<button type="button" className="browser-progress-retry" onClick={event=>{
      if(document.activeElement===event.currentTarget&&!event.currentTarget.closest('[hidden]'))event.currentTarget.closest('.browser-run-progress')?.querySelector<HTMLButtonElement>('.browser-progress-stop')?.focus({preventScroll:true});
      retry();
    }}>{t('重试详细进度','Retry detailed progress')}</button>}
  </div>;
}
