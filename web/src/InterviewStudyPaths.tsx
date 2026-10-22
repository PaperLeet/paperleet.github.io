import AppLink from './AppLink';
import {useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {Search,X,ChevronDown,ChevronUp} from 'lucide-react';
import {appLocation} from './appLocation';
import {readInterviewRoute} from './interviewNavigation';
import {controlIsActive} from './activeControls';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import {keepReadingTriggerVisible} from './readingViewport';
import {PATH_PREVIEW_SIZE,matchesPreparationPath,matchingPreparationModules,pathSearchDraft} from './preparationPathSearch';
import './preparation-path-browser.css';
import {ArrowRight,BookOpen,CheckCircle2,Clock3} from 'lucide-react';
import {useI18n} from './i18n';
import {interviewHref,type InterviewFilters} from './interviewNavigation';
import {preparationProgress,rolePreparationPaths} from './interviewPreparation';
import {withLibraryContext} from './libraryNavigation';
import {status,type Problem,type Saved} from './types';
import PracticeMetadata from './PracticeMetadata';
import PracticePathWorkload from './PracticePathWorkload';

type Props={filters:InterviewFilters;problems:Problem[];saved:Record<string,Saved>;ready:boolean;error?:string;onRetry?:()=>void;owner?:string;canAct?:()=>boolean};
export default function InterviewStudyPaths(props:Props){return <PathBrowser key={JSON.stringify([props.owner,props.filters.role,props.filters.track])} {...props}/>;}
function PathBrowser({filters,problems,saved,ready:loaded,error,onRetry,canAct}:Props){
 const {t,locale}=useI18n(),en=locale==='en',id=useId(),root=useRef<HTMLDivElement>(null),input=useRef<HTMLInputElement>(null),extra=useRef<HTMLDetailsElement>(null),summary=useRef<HTMLElement>(null),composing=useRef(false);
 const [search,setSearch]=useState({draft:'',applied:''}),[expanded,setExpanded]=useState({scope:'',open:false});
 const ready=loaded&&!error,available=rolePreparationPaths(filters.role),selected=preparationProgress(filters.track,problems)?.path;
 const paths=selected?[selected,...available.filter(path=>path.id!==selected.id)]:available;
 const modules=useMemo(()=>new Map(problems.map(problem=>[problem.id,problem])),[problems]);
 const matches=paths.filter(path=>matchesPreparationPath(path,modules,search.applied)),scope=JSON.stringify([search.applied,matches.map(path=>path.id)]),open=expanded.scope===scope&&expanded.open;
 const expected=interviewHref(filters),intent=useWorkspaceIntent(expected),pending=useRef<{target:'summary'|'selected';claim:()=>boolean}|undefined>(undefined);
 const current=()=>{const route=readInterviewRoute(appLocation.hash);return intent.isCurrent()&&controlIsActive(root.current)&&!root.current?.closest('details:not([open]),dialog:not([open])')&&!!route&&interviewHref(route)===expected&&canAct?.()!==false;};
 const active=(node:HTMLElement)=>current()&&controlIsActive(node)&&!node.closest('details:not([open]),dialog:not([open])');
 function change(value:string,transient=false){if(!current())return;const draft=transient?value:pathSearchDraft(value);setSearch({draft,applied:transient?search.applied:draft});}
 function clear(){if(!current())return;composing.current=false;setSearch({draft:'',applied:''});input.current?.focus({preventScroll:true});}
 function collapse(){if(!current()||!open)return;pending.current={target:'summary',claim:intent.create()};setExpanded({scope,open:false});}
 function revealSelected(){if(!current()||!selected)return;pending.current={target:'selected',claim:intent.create()};composing.current=false;setSearch({draft:'',applied:''});setExpanded({scope:'',open:false});}
 useLayoutEffect(()=>{
  const request=pending.current;if(!request)return;pending.current=undefined;
  const target=request.target==='summary'?summary.current:root.current?.querySelector<HTMLElement>('[data-selected] h3');
  const allowed=request.claim();if(!allowed||!current()||!controlIsActive(target))return;
  target?.focus({preventScroll:true});if(target)keepReadingTriggerVisible(target,100);
 },[open,scope]);
 const renderPath=(path:typeof paths[number])=><PathCard key={path.id} path={path} filters={filters} problems={problems} saved={saved} ready={ready} active={active} matches={matchingPreparationModules(path,modules,search.applied)}/>;
 return <div ref={root} className="interview-path-browser">
  <p className="interview-progress-context">{t('进度沿用当前学习工作区，通过测试后自动更新。用时按已知题目预计时长相加，仅供安排练习，实际用时因人而异。','Progress follows your current learning workspace and updates after passing tests. Time estimates add the known exercise estimates for planning; actual time varies.')}</p>
  {!ready&&<p className="interview-progress-status" role="status">{error?t('学习进度暂不可用，仍可打开题目练习。','Learning progress is unavailable. You can still open practice modules.'):t('正在读取学习进度…','Loading learning progress…')}{error&&onRetry&&<button type="button" className="text-button" onClick={()=>{if(current())onRetry?.();}}>{t('重试学习进度','Retry learning progress')}</button>}</p>}
  {(paths.length>PATH_PREVIEW_SIZE||!!search.draft)&&<div className="path-browser-search" role="search" aria-label={t('筛选准备路径','Filter preparation paths')}>
   <div><Search size={17} aria-hidden="true"/><input ref={input} type="search" maxLength={200} value={search.draft} aria-label={t('筛选准备路径','Filter preparation paths')} aria-controls={id+'-list'} aria-describedby={id+'-help '+id+'-count'} placeholder={t('路径名、知识点或练习…','Path, concept or exercise…')}
    onChange={event=>change(event.target.value,composing.current||(event.nativeEvent as InputEvent).isComposing===true)} onCompositionStart={()=>{if(current())composing.current=true;}} onCompositionEnd={event=>{composing.current=false;change(event.currentTarget.value);}}
    onKeyDown={event=>{if(!current()||event.defaultPrevented)return;const ime=composing.current||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229;if(event.key==='Enter'){event.stopPropagation();if(!ime)event.preventDefault();}else if(event.key==='Escape'&&!ime&&!event.repeat&&!event.altKey&&!event.ctrlKey&&!event.metaKey&&!event.shiftKey&&search.draft){event.preventDefault();event.stopPropagation();clear();}}}/>
    {!!search.draft&&<button type="button" aria-label={t('清空路径搜索','Clear path search')} onClick={clear}><X size={16} aria-hidden="true"/></button>}
   </div><p id={id+'-help'}>{t('仅筛选中英文路径名、简介与已加载练习；不会改变面经范围。最多100个字符。','Search bilingual path names, descriptions and loaded exercises without changing the experience scope. Up to 100 characters.')}</p>
  </div>}
  {selected&&!matches.some(path=>path.id===selected.id)&&<div className="path-browser-current" role="status"><span>{t('当前路径未出现在筛选结果中：','The current path is outside these results: ')}<b>{en?selected.name_en:selected.name}</b></span><button type="button" onClick={revealSelected}>{t('定位当前路径','Show current path')}</button></div>}
  <p id={id+'-count'} className="path-browser-count" role="status" aria-live="polite" aria-atomic="true">{t(`显示 ${open?matches.length:Math.min(PATH_PREVIEW_SIZE,matches.length)} / ${matches.length} 条路径${search.applied.trim()?` · 当前范围共 ${paths.length} 条`:''}`,`Showing ${open?matches.length:Math.min(PATH_PREVIEW_SIZE,matches.length)} of ${matches.length} paths${search.applied.trim()?` · ${paths.length} in this scope`:''}`)}</p>
  <div id={id+'-list'}>
   {!matches.length&&<div className="path-browser-empty"><p>{t('没有匹配的准备路径。可以换个关键词，或清空搜索。','No preparation paths match. Try another keyword or clear the search.')}</p><button type="button" onClick={clear}>{t('显示全部路径','Show all paths')}</button></div>}
   <div className="interview-path-grid">{matches.slice(0,PATH_PREVIEW_SIZE).map(renderPath)}</div>
   {matches.length>PATH_PREVIEW_SIZE&&<details ref={extra} className="path-browser-extra" open={open} onToggle={event=>{if(current()&&event.currentTarget.open!==open)setExpanded({scope,open:event.currentTarget.open});}} onKeyDown={event=>{
    const target=document.activeElement;if(!open||!current()||event.defaultPrevented||event.key!=='Escape'||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||!(target instanceof HTMLElement)||target.closest('details')!==extra.current||target.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])'))return;
    event.preventDefault();event.stopPropagation();collapse();
   }}><summary ref={summary} aria-controls={id+'-extra'} onClick={event=>{if(!current())event.preventDefault();}}><ChevronDown size={17} aria-hidden="true"/>{open?t('收起其余路径','Show fewer paths'):t(`展开其余 ${matches.length-PATH_PREVIEW_SIZE} 条路径`,`Explore ${matches.length-PATH_PREVIEW_SIZE} more paths`)}</summary>
    <div id={id+'-extra'} hidden={!open}>{open&&<><div className="interview-path-grid">{matches.slice(PATH_PREVIEW_SIZE).map(renderPath)}</div><button type="button" className="path-browser-collapse" onClick={collapse}><ChevronUp size={16} aria-hidden="true"/>{t('收起其余路径','Show fewer paths')}</button></>}</div>
   </details>}
  </div>
 </div>;
}

function PathCard({path,filters,problems,saved,ready,active,matches}:{path:ReturnType<typeof rolePreparationPaths>[number];filters:InterviewFilters;problems:Problem[];saved:Record<string,Saved>;ready:boolean;active:(node:HTMLElement)=>boolean;matches:Problem[]}){
 const {t,locale}=useI18n(),en=locale==='en';
 const currentSummary=(node:HTMLElement)=>{const parent=node.closest('details')?.parentElement;return !!parent&&active(parent);};
   const progress=preparationProgress(path.id,problems,saved)!;
   const selected=filters.track===path.id;
   const returnTo=interviewHref({...filters,track:path.id});
   const title=(problem:Problem)=>en?problem.translations?.en?.title||problem.title:problem.title;
   const links=progress.modules.map(problem=>{
    const state=ready?status(problem,saved[problem.id]):undefined;
    return <AppLink key={problem.id} onClick={event=>{if(!active(event.currentTarget))event.preventDefault();}} href={withLibraryContext(problem.number,returnTo)}><span className="interview-path-module-copy"><span className="interview-path-module-title">{title(problem)}</span><PracticeMetadata difficulty={problem.difficulty} estimatedMinutes={problem.estimated_minutes}/></span><span className={`interview-path-module-state ${state||''}`} aria-label={!ready?t('进度未就绪','Progress not ready'):state==='done'?t('已通过','Passed'):state==='started'?t('进行中','In progress'):t('未开始','Not started')}>{state==='done'?<CheckCircle2 size={13}/>:state==='started'?<Clock3 size={13}/>:<ArrowRight size={12}/>}</span></AppLink>;
   });
 return <article key={path.id} data-selected={selected||undefined} aria-labelledby={`interview-path-${path.id}`}>
    <span className="interview-path-icon"><BookOpen size={19}/></span>{selected&&<span className="interview-path-selected">{t('当前路径','Current path')}</span>}
    <h3 tabIndex={-1} id={`interview-path-${path.id}`}>{en?path.name_en:path.name}</h3><p>{en?path.description_en:path.description}</p>
    <PracticePathWorkload problems={progress.modules} total={progress.total} saved={saved} ready={ready}/>
    {matches.length>0&&<div className="path-browser-matches">
     <h4>{t('关键词命中练习','Exercises mentioning your keywords')}<span>{matches.length}</span></h4>
     <p>{t('标题或编号包含搜索词，可直接打开练习。','Titles or identifiers contain your search terms. Open an exercise directly.')}</p>
     <ul>{matches.map(problem=><li key={problem.id}><AppLink onClick={event=>{if(!active(event.currentTarget))event.preventDefault();}} href={withLibraryContext(problem.number,returnTo)}><span className="path-browser-match-number">{problem.number}</span><span><span className="path-browser-match-title">{title(problem)}</span><PracticeMetadata difficulty={problem.difficulty} estimatedMinutes={problem.estimated_minutes}/></span><ArrowRight size={14} aria-hidden="true"/></AppLink></li>)}</ul>
    </div>}
    <div className="interview-path-progress" aria-busy={!ready}>{ready?<><span>{t('已通过','Passed')} <b>{progress.completed} / {progress.total}</b></span><progress aria-label={t(`${path.name}进度`,`${path.name_en} progress`)} value={progress.completed} max={progress.total}/></>:<span>{t('等待学习进度','Waiting for progress')}</span>}</div>
    {ready&&progress.next?<AppLink onClick={event=>{if(!active(event.currentTarget))event.preventDefault();}} className="interview-path-start button subtle" href={withLibraryContext(progress.next.number,returnTo)}><span>{status(progress.next,saved[progress.next.id])==='started'?t('继续准备','Resume preparation'):t('开始下一题','Start next module')}<small>{title(progress.next)}</small><PracticeMetadata difficulty={progress.next.difficulty} estimatedMinutes={progress.next.estimated_minutes}/></span><ArrowRight size={15}/></AppLink>:ready&&progress.completed===progress.total?<AppLink onClick={event=>{if(!active(event.currentTarget))event.preventDefault();}} className="interview-path-start button subtle" href={interviewHref({...filters,track:path.id,view:'reports',page:1})}><CheckCircle2 size={15}/>{t('路径已完成 · 返回复盘','Path complete · review experiences')}</AppLink>:null}
    <div className="interview-path-modules">{links.slice(0,3)}{links.length>3&&<details className="interview-path-module-disclosure" open={selected}><summary onClick={event=>{if(!currentSummary(event.currentTarget))event.preventDefault();}}><span className="interview-path-module-expand-label">{t(`再看 ${links.length-3} 道练习`,`Show ${links.length-3} more modules`)}</span><span className="interview-path-module-collapse-label">{t('收起练习列表','Show fewer modules')}</span><span className="interview-path-module-chevron" aria-hidden="true"><ArrowRight size={13}/></span></summary><div className="interview-path-module-extra">{links.slice(3)}</div></details>}</div>
    {progress.missing>0&&<small role="status">{t(`${progress.missing} 道练习暂不可用；补全题库后再确认路径完成。`,`${progress.missing} ${progress.missing===1?'module is':'modules are'} unavailable. Completion waits for the full catalog.`)}</small>}
   </article>;
}
