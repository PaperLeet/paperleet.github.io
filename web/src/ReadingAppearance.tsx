import {useId,useLayoutEffect,useRef,useSyncExternalStore,type RefObject} from 'react';
import {ChevronDown,Type} from 'lucide-react';
import {useI18n} from './i18n';
import {controlIsActive} from './activeControls';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import {restoreReadingAnchor} from './readingViewport';
import {readingPreferenceAnchor,type PreferenceAnchor} from './readingPreferenceAnchor';
import {readingPreferences,type ReadingPreferenceSnapshot} from './readingPreferences';
import './reading-appearance.css';

type Props={mode?:'inline'|'settings';target?:RefObject<HTMLElement|null>;scope?:string;controlsId?:string;canAct?:()=>boolean};
type Pending={point:PreferenceAnchor;root:HTMLElement;expected:ReadingPreferenceSnapshot;claim:()=>boolean};
export default function ReadingAppearance({mode='inline',target,scope='',controlsId,canAct}:Props){
 const {t,locale}=useI18n(),id=useId(),root=useRef<HTMLDivElement>(null),details=useRef<HTMLDetailsElement>(null),pending=useRef<Pending|undefined>(undefined);
 const preferences=useSyncExternalStore(readingPreferences.subscribe,readingPreferences.getSnapshot,readingPreferences.getServerSnapshot);
 const href=typeof window==='undefined'?'':window.location.href,intent=useWorkspaceIntent(JSON.stringify([scope,locale,href]));
 const current=()=>controlIsActive(root.current)&&location.href===href&&intent.isCurrent()&&canAct?.()!==false;
 const panelActive=()=>current()&&(mode==='settings'||details.current?.open===true);
 function change(action:()=>void){
  if(!panelActive())return;
  pending.current=undefined;intent.cancel();
  const reader=target?.current,point=reader&&controlIsActive(reader)?readingPreferenceAnchor(reader):undefined,before=readingPreferences.getSnapshot();
  if(reader&&point)pending.current={root:reader,point,expected:before,claim:intent.create()};
  action();
  const after=readingPreferences.getSnapshot();
  if(after===before){pending.current=undefined;intent.cancel();}
  else if(pending.current)pending.current.expected=after;
 }
 useLayoutEffect(()=>{
  const request=pending.current;if(!request)return;pending.current=undefined;
  if(request.expected===preferences&&panelActive()&&target?.current===request.root&&controlIsActive(request.root)&&request.root.contains(request.point.node)&&request.claim())restoreReadingAnchor(request.point.node,request.point.origin);
  else intent.cancel();
 },[preferences,scope,locale,href]);
 const controlled=controlsId||(mode==='settings'?`${id}-preview`:undefined);
 const defaults=preferences.size==='default'&&preferences.spacing==='default'&&preferences.saved;
 const fields=<div className="reading-appearance-fields">
  <fieldset><legend>{t('正文字号','Text size')}</legend><div role="group" aria-label={t('正文字号','Text size')}>{(['default','18','20'] as const).map(size=><button key={size} type="button" aria-controls={controlled} aria-pressed={preferences.size===size} onClick={()=>change(()=>readingPreferences.set({size}))}>{size==='default'?t('默认','Default'):`${size}px`}</button>)}</div></fieldset>
  <fieldset><legend>{t('行距','Line spacing')}</legend><div role="group" aria-label={t('行距','Line spacing')}>{(['default','relaxed'] as const).map(spacing=><button key={spacing} type="button" aria-controls={controlled} aria-pressed={preferences.spacing===spacing} onClick={()=>change(()=>readingPreferences.set({spacing}))}>{spacing==='default'?t('默认','Default'):t('舒展','Relaxed')}</button>)}</div></fieldset>
  {mode==='settings'&&<div id={`${id}-preview`} className="prose reading-preference-preview"><p>{t('把公式讲清楚，把实现推出来，再用边界情况检验理解。','Explain the idea, work through the implementation, then test your understanding with edge cases.')}</p></div>}
  <p className="reading-appearance-note" role={!preferences.saved?'status':undefined}>{preferences.saved?t('正文、公式与表格随阅读偏好显示；编辑器字号保持独立。偏好仅保存在当前浏览器。','Reading preferences apply to text, formulas and tables. Editor text size stays independent. Preferences save in this browser.'):t('外观已在当前页面应用；浏览器暂时无法保存偏好。','Applied in this page. Your browser could not save the preference.')}</p>
  <div className="reading-appearance-actions"><button type="button" aria-disabled={preferences.saved} tabIndex={preferences.saved?-1:0} onClick={()=>{if(!preferences.saved)change(readingPreferences.retry);}}>{preferences.saved?t('外观已保存','Appearance saved'):t('重试保存外观','Retry appearance saving')}</button><button type="button" aria-controls={controlled} aria-disabled={defaults} tabIndex={defaults?-1:0} onClick={()=>{if(!defaults)change(readingPreferences.reset);}}>{t('恢复默认','Restore defaults')}</button></div>
 </div>;
 return <div ref={root} className={`reading-appearance reading-appearance-${mode}`} onKeyDown={event=>{
  if(mode!=='inline'||!panelActive()||event.key!=='Escape'||event.defaultPrevented||event.repeat||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||event.nativeEvent.isComposing||event.keyCode===229)return;
  event.preventDefault();event.stopPropagation();details.current!.open=false;pending.current=undefined;intent.cancel();details.current?.querySelector('summary')?.focus({preventScroll:true});
 }}>
  {mode==='settings'?<><div className="reading-appearance-heading"><Type size={17} aria-hidden="true"/><span>{t('阅读外观','Reading appearance')}</span></div>{fields}</>:<details ref={details}><summary onClick={event=>{if(!current())event.preventDefault();}}><Type size={16} aria-hidden="true"/><span>{t('阅读外观','Reading appearance')}</span>{preferences.size!=='default'&&<small>{preferences.size}px</small>}<ChevronDown size={15} aria-hidden="true"/></summary>{fields}</details>}
 </div>;
}
