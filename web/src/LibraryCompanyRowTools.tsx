import {appLocation} from './appLocation';
import AppLink from './AppLink';
import {useId,useRef,useState} from 'react';
import {ArrowUpRight,ChevronDown,ChevronUp,Users} from 'lucide-react';
import {useI18n} from './i18n';
import {useAccount} from './social';
import {deferredFeature} from './deferredFeature';
import {emptyInterviewFilters,interviewHref,type InterviewFilters} from './interviewNavigation';
import {libraryHref,readLibraryRoute} from './libraryNavigation';
import {libraryInterviewPreviewScope} from './interviewSourcePreviewModel';
import {keepReadingTriggerVisible} from './readingViewport';
import type {LibraryCompanyRowToolsProps} from './libraryCompanyScope';
import './library-company-rows.css';

function ToolsFailure({retry}:{retry:()=>void}){const {t}=useI18n();return <p className="library-company-tools-error" role="alert">{t('操作区暂时无法加载，仍可打开题目。','These actions could not load. You can still open the exercise.')} <button type="button" className="text-button" onClick={retry}>{t('重试加载','Retry loading')}</button></p>;}
const Favorite=deferredFeature<Parameters<typeof import('./ProblemFavorites').ProblemFavoriteButton>[0]>(()=>import('./ProblemFavorites').then(module=>({default:module.ProblemFavoriteButton})),['题目收藏','saved problems'],{loading:()=>null,failure:(_props,retry)=><ToolsFailure retry={retry}/>});
const Sources=deferredFeature<Parameters<typeof import('./InterviewSourcePreview').InterviewSourceReports>[0]>(()=>import('./InterviewSourcePreview').then(module=>({default:module.InterviewSourceReports})),['来源面经','source experiences'],{failure:(_props,retry)=><ToolsFailure retry={retry}/>});

export default function LibraryCompanyRowTools(props:LibraryCompanyRowToolsProps){
 const {user,loading,capabilities}=useAccount(),data=props.companyScope.data;
 if(loading||data?.scope!==props.companyScope.scope||data?.phase!=='ready')return null;
 return <RowTools key={JSON.stringify([user?.id,data.scope])} {...props} owner={user?.id||'guest'} favoriteAvailable={capabilities?.problem_bookmarks===true}/>;
}
function RowTools({problem,route,returnTo,companyScope,owner,favoriteAvailable}:LibraryCompanyRowToolsProps&{owner:string;favoriteAvailable:boolean}){
 const {t,locale}=useI18n(),id=useId(),toggle=useRef<HTMLButtonElement>(null),root=useRef<HTMLDivElement>(null),panel=useRef<HTMLElement>(null),[open,setOpen]=useState(false);
 const data=companyScope.data!,view=readLibraryRoute(route);
 const source=data.summary?.questions.find(row=>row.problem_id===problem.id)?.companies.find(row=>row.company===data.company);
 const sourceHref=interviewHref({...emptyInterviewFilters(),company:data.company,days:data.days as InterviewFilters['days'],concept:problem.id});
 const current=view?libraryHref(view.collectionId,view.filters):undefined;
 const here=()=>!!current&&!!root.current?.isConnected&&libraryInterviewPreviewScope(appLocation.hash,sourceHref)===current;
 const name=locale==='en'?problem.translations?.en?.title||problem.title:problem.title;
 function close(){if(!here())return;if(panel.current?.contains(document.activeElement)){toggle.current?.focus({preventScroll:true});if(toggle.current)keepReadingTriggerVisible(toggle.current,100);}setOpen(false);}
 if(!source||!data.ids.has(problem.id)||!current||view?.filters.company!==data.company||(view.filters.company_days||'365')!==data.days)return null;
 return <div ref={root} className="library-company-row-tools">
  <div className="library-company-row-toolbar">
   <span className="library-company-row-counts" title={t('按当前公司与面经发布时间汇总；提及不代表实际考频。','Snapshot for this company and report publication period; mentions are not actual interview frequency.')}><Users size={14} aria-hidden="true"/>{t(`${source.contributors} 位分享者 · ${source.reports} 篇面经`,`${source.contributors} ${source.contributors===1?'contributor':'contributors'} · ${source.reports} ${source.reports===1?'report':'reports'}`)}</span>
   <div className="library-company-row-actions"><button ref={toggle} type="button" className="text-button" aria-expanded={open} aria-controls={id} aria-label={open?t(`收起${name}的来源面经`,`Collapse sources for ${name}`):t(`预览${name}的来源面经`,`Preview sources for ${name}`)} onClick={()=>{if(here())open?close():setOpen(true);}}>{open?<ChevronUp size={14} aria-hidden="true"/>:<ChevronDown size={14} aria-hidden="true"/>}{open?t('收起来源','Collapse sources'):t('预览来源','Preview sources')}</button>{favoriteAvailable&&<Favorite problemId={problem.id} returnTo={returnTo}/>}</div>
  </div>
  <section id={id} ref={panel} hidden={!open} className="library-company-row-preview interview-source-preview" role="region" aria-label={t(`${name}的来源面经`,`Source experiences for ${name}`)} onKeyDown={event=>{if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey)return;event.preventDefault();event.stopPropagation();close();}}>
   {open&&<><Sources concept={problem.id} name={name} reports={source.reports} contributors={source.contributors} sourceHref={sourceHref} returnTo={sourceHref} owner={owner+':'+data.scope} library={{href:current}}/><footer><AppLink href={sourceHref} target="_blank" rel="noopener noreferrer">{t('新标签页查看全部面经','Read all sources in a new tab')}<ArrowUpRight size={14} aria-hidden="true"/></AppLink><button type="button" className="text-button" onClick={close}>{t('收起来源','Collapse sources')}</button></footer></>}
  </section>
 </div>;
}
