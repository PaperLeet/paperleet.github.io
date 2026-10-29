import {useEffect,useId,useRef,useState} from 'react';
import {BookOpen,ChevronDown} from 'lucide-react';
import AppLink from './AppLink';
import {appLocation} from './appLocation';
import {useI18n} from './i18n';
import {communityFeedHref,readCommunityFeed} from './communityNavigation';
import {communityTopicScopeHref,validCommunityTopic} from './communityTopicScope';
import {topicTitle,type ProblemTopic} from './communityTopics';
import {searchRelatedModules} from './relatedModuleMatching';
import RelatedModuleSearch from './RelatedModuleSearch';
import {keepReadingTriggerVisible} from './readingViewport';
import './community-topic-browser.css';

/** The parent keys this local draft by the canonical feed and account. */
export default function CommunityTopicBrowser({from,topics=[],loading=false,error,reload}:{from:string;topics?:ProblemTopic[];loading?:boolean;error?:string;reload?:()=>void}){
 const {t,locale}=useI18n(),selectId=useId(),hintId=useId();
 const root=useRef<HTMLDetailsElement>(null),summary=useRef<HTMLElement>(null),composing=useRef(false);
 const current=readCommunityFeed(from)?.topicId||'';
 const [query,setQuery]=useState(''),[selected,setSelected]=useState(current);
 const outside=useRef<(event:PointerEvent)=>void>(undefined);
 useEffect(()=>()=>{if(outside.current)document.removeEventListener('pointerdown',outside.current);},[]);
 /** As a dropdown (full community page), a press outside closes it like any menu; inline panels stay open. */
 function toggled(open:boolean){
  if(outside.current)document.removeEventListener('pointerdown',outside.current);outside.current=undefined;
  if(!open)return;
  outside.current=event=>dismiss(event.target as Node);
  document.addEventListener('pointerdown',outside.current);
 }
 /** Closes the dropdown (never the inline panel) when a press or keyboard focus lands outside it. */
 function dismiss(next:Node|null){const node=root.current,panel=node?.querySelector('.community-topic-panel');if(node?.open&&next&&panel&&!node.contains(next)&&getComputedStyle(panel).position==='absolute')node.open=false;}
 const available=topics.filter(topic=>!topic.retired&&validCommunityTopic(topic.id));
 const matches=searchRelatedModules(available,query),currentTopic=topics.find(topic=>topic.id===current),chosen=topics.find(topic=>topic.id===selected);
 const missingSelection=!!selected&&!matches.some(topic=>topic.id===selected);
 const unavailable=!!selected&&(loading||!!error||!available.some(topic=>topic.id===selected));
 const href=communityTopicScopeHref(from,selected),changed=selected!==current,canApply=changed&&!!href&&!unavailable;
 function here(){const feed=readCommunityFeed(appLocation.hash);return !!root.current?.isConnected&&!!feed&&communityFeedHref(feed.topicId,feed.state)===from;}
 function close(){if(!here()||!root.current)return;if(root.current.contains(document.activeElement)&&summary.current){summary.current.focus({preventScroll:true});keepReadingTriggerVisible(summary.current,100);}root.current.open=false;}
 if(!topics.length&&!current&&!loading&&!error)return null;
 const title=currentTopic?topicTitle(currentTopic,locale==='en'):current||t('全部题目','All modules');
 return <details className="community-topic-browser" ref={root} onToggle={event=>toggled(event.currentTarget.open)} onBlur={event=>dismiss(event.relatedTarget as Node|null)} onCompositionStart={()=>{composing.current=true;}} onCompositionEnd={()=>{composing.current=false;}} onKeyDown={event=>{
  if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey||composing.current||event.nativeEvent.isComposing||event.keyCode===229||(event.target as HTMLElement).tagName==='SELECT')return;
  event.preventDefault();event.stopPropagation();close();
 }}>
  <summary ref={summary} aria-label={t('题目范围：','Module scope: ')+title}><BookOpen size={16} aria-hidden="true"/><span title={title}>{current?title:t('题目范围','Module scope')}</span><ChevronDown size={15} aria-hidden="true"/></summary>
  <div className="community-topic-panel">
   <header><div><strong>{t('按题目发现题解与讨论','Find solutions and discussions by module')}</strong><p id={hintId}>{t('先搜索并选择题目，再查看结果；当前收藏、标签和排序等筛选会保留。','Search and select a module, then view results. Your saved, tag, order and other filters stay in place.')}</p></div><button type="button" className="text-button" onClick={close}>{t('收起','Collapse')}</button></header>
   <RelatedModuleSearch value={query} onChange={value=>{if(here())setQuery(value);}} label={t('查找要浏览的题目','Find a module to browse')}/>
   <label className="community-topic-selection" htmlFor={selectId}><span>{t('题目范围','Module scope')}</span><select id={selectId} value={selected} aria-describedby={hintId} onChange={event=>{if(here())setSelected(event.target.value);}}>
    <option value="">{t('全部题目','All modules')}</option>
    {missingSelection&&<optgroup label={t('保留的选择','Kept selection')}><option value={selected}>{chosen?topicTitle(chosen,locale==='en'):selected}</option></optgroup>}
    {matches.map(topic=><option key={topic.id} value={topic.id}>{topicTitle(topic,locale==='en')}</option>)}
   </select></label>
   <div className="community-topic-status" role="status" aria-live="polite" aria-atomic="true">
    {loading?t('正在读取题目目录；当前结果不受影响。','Loading modules; your current results are kept.'):error?t('题目目录暂时不可用。可以重试，或选择全部题目继续浏览。','The module catalog is unavailable. Retry, or browse all modules.'):unavailable?t('所选题目暂时不在可用目录中。请选择其他题目，或查看全部题目。','The selected module is unavailable. Choose another module, or browse all modules.'):!matches.length?t('没有匹配的题目。试试题号、其他名称或模块 ID；当前选择已保留。','No modules match. Try a number, another title or module ID. Your selection is kept.'):t(`匹配 ${matches.length} 道题目；搜索只筛选选项。`,`${matches.length} modules match; searching only filters the options.`)}
   </div>
   <footer>{error&&reload&&<button type="button" className="text-button" disabled={loading} onClick={()=>{if(here()&&!loading)reload();}}>{t('重试加载目录','Retry module catalog')}</button>}<AppLink className="button primary" href={canApply?href!:from} aria-disabled={!canApply} onClick={event=>{if(!here()||!canApply||composing.current)event.preventDefault();}}>{t('查看结果','View results')}</AppLink>{changed&&<span>{t('尚未应用题目范围','Module scope not applied')}</span>}</footer>
  </div>
 </details>;
}
