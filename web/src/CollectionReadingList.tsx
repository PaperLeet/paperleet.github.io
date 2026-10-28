import {useEffect,useId,useRef} from 'react';
import {ArrowLeft,ArrowRight,BookOpen,Code2,ExternalLink,Search,X} from 'lucide-react';
import AppLink from './AppLink';
import {appLocation} from './appLocation';
import {controlIsActive} from './activeControls';
import {useI18n} from './i18n';
import {paperCollections} from './paperCollections';
import {paperProgress} from './libraryPractice';
import {emptyLibraryFilters,libraryHref,readLibraryRoute,withLibraryContext} from './libraryNavigation';
import type {Paper,Problem,Saved} from './types';
import readingData from './data/collection-reading-lists.json';
import './collection-reading-list.css';

type ReadingPaper={id:string;short_title:string;title:string;year:number;url:string;summary:string;summary_en:string;priority:boolean;paper_id?:string};
type ReadingStage={id:string;title:string;title_en:string;papers:ReadingPaper[]};
type ReadingCollection={id:string;description:string;description_en:string;stages:ReadingStage[]};
export type Props={collectionId:string;papers:Paper[];problems:Problem[];saved:Record<string,Saved>;progressReady?:boolean;route?:string;onFilters?:(href:string,paging?:boolean)=>void};
const collections:ReadingCollection[]=readingData.collections;
const searchable=(value:string)=>value.normalize('NFKC').toLowerCase();

/** Full reading records stay separate from executable papers; this module and its data load only on reading views. */
export default function CollectionReadingList({collectionId,papers,problems,saved,progressReady=true,route='',onFilters}:Props){
 const {t}=useI18n(),root=useRef<HTMLElement>(null),search=useRef<HTMLInputElement>(null),listId=useId();
 const collection=paperCollections.find(item=>item.id===collectionId),reading=collections.find(item=>item.id===collectionId);
 const query=readLibraryRoute(route)?.filters.query||'',returnTo=libraryHref(collectionId,{...emptyLibraryFilters(),query});
 const practiceHref=libraryHref(collectionId,{...emptyLibraryFilters(),query,view:'modules'});
 const terms=searchable(query).trim().split(/\s+/).filter(Boolean);
 const matches=(paper:ReadingPaper,stage:ReadingStage)=>terms.every(term=>searchable([paper.id,paper.short_title,paper.title,paper.year,paper.url,paper.summary,paper.summary_en,stage.title,stage.title_en].join(' ')).includes(term));
 const stages=reading?.stages.map(stage=>({...stage,papers:stage.papers.filter(paper=>matches(paper,stage))})).filter(stage=>stage.papers.length)||[];
 const all=reading?.stages.flatMap(stage=>stage.papers)||[],shown=stages.reduce((count,stage)=>count+stage.papers.length,0);
 const practicePaper=(entry:ReadingPaper)=>entry.paper_id?papers.find(paper=>paper.id===entry.paper_id&&paper.module_ids.some(id=>problems.some(problem=>problem.id===id))):undefined;
 const available=all.filter(entry=>practicePaper(entry)).length;
 function currentView(){
  const current=readLibraryRoute(appLocation.hash);
  return controlIsActive(root.current)&&current?.collectionId===collectionId&&current.filters.view!=='modules'&&current.filters.query===query;
 }
 function changeQuery(value:string){
  if(!currentView())return;
  const href=libraryHref(collectionId,{...emptyLibraryFilters(),query:value});
  if(onFilters)onFilters(href);else appLocation.hash=href.slice(1);
 }
 function clear(){if(currentView()){search.current?.focus({preventScroll:true});changeQuery('');}}
 useEffect(()=>{
  const focus=(event:KeyboardEvent)=>{
   if(event.key!=='/'||event.defaultPrevented||event.repeat||event.isComposing||event.keyCode===229||event.ctrlKey||event.metaKey||event.altKey||!currentView())return;
   if(event.target instanceof Element&&event.target.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"]),[role="textbox"],[role="dialog"],dialog[open]')||document.querySelector('dialog[open]'))return;
   event.preventDefault();search.current?.focus({preventScroll:true});search.current?.scrollIntoView({block:'nearest'});
  };
  window.addEventListener('keydown',focus);return()=>window.removeEventListener('keydown',focus);
 },[returnTo]);
 if(!collection||!reading)return null;
 return <main className="home paper-library collection-page collection-reading-page" ref={root}>
  <header className="collection-page-header">
   <AppLink className="back-link" href="#"><ArrowLeft size={15} aria-hidden="true"/>{t('全部论文','All papers')}</AppLink>
   <div className="collection-page-title-row">
    <div className="collection-page-identity"><span className={`collection-page-mark collection-${collectionId}`} aria-hidden="true"><BookOpen size={25}/></span><div><h1>{t(collection.name,collection.name_en)}</h1><div className="collection-page-meta"><span>{all.length} {t('篇论文','papers')}</span><span>{available} {t('篇可练','with exercises')}</span></div></div></div>
    <AppLink className="button collection-practice" href={practiceHref}><Code2 size={16} aria-hidden="true"/>{t('查看练习','Browse exercises')}<ArrowRight size={15} aria-hidden="true"/></AppLink>
   </div>
   <p className="collection-reading-description">{t(reading.description,reading.description_en)}</p>
  </header>
  <section className="collection-reading-list" data-collection-id={collectionId} aria-label={t('合集阅读清单','Collection reading list')}>
   <nav className="collection-reading-views" aria-label={t('合集视图','Collection views')}>
    <AppLink href={returnTo} aria-current="page"><BookOpen size={17} aria-hidden="true"/>{t('阅读清单','Reading list')}<span>{all.length}</span></AppLink>
    <AppLink href={practiceHref}><Code2 size={17} aria-hidden="true"/>{t('可练习','Practice')}<span>{available}</span></AppLink>
   </nav>
   <div className="collection-reading-toolbar">
    <div className="search"><Search size={17} aria-hidden="true"/><input ref={search} type="search" maxLength={200} aria-label={t('搜索合集论文','Search collection papers')} aria-controls={listId} aria-keyshortcuts="/" placeholder={t('搜索标题、年份、方法或 arXiv…','Search title, year, method or arXiv…')} value={query} onChange={event=>changeQuery(event.target.value)} onKeyDown={event=>{if(event.key==='Escape'&&!event.defaultPrevented&&!event.repeat&&!event.nativeEvent.isComposing&&event.nativeEvent.keyCode!==229&&!event.ctrlKey&&!event.metaKey&&!event.altKey&&!event.shiftKey&&currentView()){event.preventDefault();event.stopPropagation();clear();}}}/>{query?<button type="button" className="search-clear" aria-label={t('清空搜索','Clear search')} onClick={clear}><X size={15}/></button>:<kbd aria-hidden="true">/</kbd>}</div>
    <p className="reading-result-count" role="status" aria-atomic="true">{query?t(`找到 ${shown} / ${all.length} 篇论文`,`${shown} of ${all.length} papers`):t(`${all.length} 篇论文 · ${reading.stages.length} 个学习阶段`,`${all.length} papers · ${reading.stages.length} learning stages`)}</p>
   </div>
   <div id={listId}>
    {stages.map(stage=><section className="collection-reading-stage" data-stage-id={stage.id} key={stage.id} aria-labelledby={listId+'-'+stage.id}>
     <h2 id={listId+'-'+stage.id}>{t(stage.title,stage.title_en)}<span>{stage.papers.length}</span></h2>
     <div className="collection-reading-papers">{stage.papers.map(entry=>{
      const paper=practicePaper(entry),progress=paper&&progressReady?paperProgress(paper,{problems,saved}):undefined;
      return <article className="collection-reading-paper" key={entry.id} data-reading-id={entry.id} data-practice-state={paper?'available':'missing'} data-progress={progress?.state}>
       <div className="reading-paper-heading"><h3>{entry.short_title}</h3><span className="reading-year">{entry.year}</span>{entry.priority&&<span className="reading-priority">{t('优先阅读','Read first')}</span>}</div>
       <p className="reading-paper-title">{entry.title}</p>
       <p className="reading-paper-summary">{t(entry.summary,entry.summary_en)}</p>
       <div className="reading-paper-actions">
        <AppLink className="reading-source-link" href={entry.url} target="_blank" rel="noreferrer">{t('阅读原文','Read paper')}<ExternalLink size={13} aria-hidden="true"/></AppLink>
        {paper?<AppLink className="reading-practice-link" data-paper-id={paper.id} href={withLibraryContext('papers/'+paper.id,returnTo)}><Code2 size={14} aria-hidden="true"/>{t('练习','Practice')}{progress&&<span>{progress.completed}/{progress.total} {t('已通过','solved')}</span>}<ArrowRight size={13} aria-hidden="true"/></AppLink>:<span className="reading-no-exercise">{t('暂无练习','No exercise yet')}</span>}
       </div>
      </article>;
     })}</div>
    </section>)}
    {!shown&&<div className="no-results" role="status"><h2>{t('没有匹配的论文','No matching papers')}</h2><p>{t('试试其他标题、方法或中英文关键词。','Try another title, method or keyword in Chinese or English.')}</p><button type="button" className="button subtle" onClick={clear}>{t('清空搜索','Clear search')}</button></div>}
   </div>
  </section>
 </main>;
}
