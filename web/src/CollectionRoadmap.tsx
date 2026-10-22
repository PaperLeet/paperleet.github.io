import {Fragment,useEffect,useId,useLayoutEffect,useRef,useState} from 'react';
import {ArrowRight} from 'lucide-react';
import AppLink from './AppLink';
import {useI18n} from './i18n';
import type {Paper,Problem,Saved} from './types';
import {paperProgress} from './libraryPractice';
import {withLibraryContext} from './libraryNavigation';
import {paperCollections} from './paperCollections';
import {publicSite,hostedAppAvailable,hostedAppHref} from './deployment';
import {parseArxivUrl} from './arxivLinks';
import PaperPreviewCard from './PaperPreviewCard';
import {roadmapFor,roadmapPapers} from './collectionRoadmaps';
import {ROADMAP,nodeBox,roadmapSize,startTag,edgePath,yearRows,matchPaper,neighbors} from './roadmapModel';
import './collection-roadmap.css';
export type Props={collectionId:string;papers:Paper[];problems:Problem[];saved:Record<string,Saved>;progressReady:boolean;returnTo:string};
type NodeState='done'|'started'|'todo'|'missing'|'pending';

/** Curated lineage graph: idle it shows only years, titles and the start tag; a selection lights its lineage (cyan in, pink out,
 *  with light flowing along the lit arrows) and opens one small panel. */
export default function CollectionRoadmap({collectionId,papers,problems,saved,progressReady,returnTo}:Props){
 const {t}=useI18n(),uid=useId(),arrow=uid+'-arrow',[selected,setSelected]=useState<string>(),buttons=useRef<Record<string,HTMLButtonElement|null>>({}),root=useRef<HTMLDivElement>(null),scroller=useRef<HTMLDivElement>(null);
 // Phones: a canvas wider than its scroller opens with the start node centered (instantly, clamped to the scroll range), on mount
 // and on every map change; selections never scroll it.
 useLayoutEffect(()=>{
  const element=scroller.current,start=roadmapFor(collectionId)?.nodes.find(node=>node.start),range=element?element.scrollWidth-element.clientWidth:0;
  if(!element||!start||range<=0)return;
  const box=nodeBox(start);element.scrollLeft=Math.min(range,Math.max(0,box.x+box.width/2-element.clientWidth/2));
 },[collectionId]);
 // Escape listens on the document because Safari and Firefox on macOS do not focus a clicked button, so keydown reaches the body.
 useEffect(()=>{
  if(!selected)return;
  const escape=(event:KeyboardEvent)=>{
   const target=event.target as Node;
   if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.isComposing||event.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||document.querySelector('dialog[open]')||target!==document.body&&!root.current?.contains(target))return;
   event.preventDefault();buttons.current[selected]?.focus();setSelected(undefined);
  };
  document.addEventListener('keydown',escape);
  return()=>document.removeEventListener('keydown',escape);
 },[selected]);
 const roadmap=roadmapFor(collectionId);
 if(!roadmap)return null;
 const size=roadmapSize(roadmap),tag=startTag(roadmap),lit=selected?neighbors(roadmap,selected):undefined,byId=(id:string)=>roadmap.nodes.find(node=>node.id===id)!;
 // Direction of each edge seen from the selection: 'from' ends at the selected node, 'to' leaves it; a neighbor takes its edge's side.
 const edges=roadmap.edges.map(([from,to])=>({from,to,d:edgePath(byId(from),byId(to)),dir:to===selected?'from':from===selected?'to':undefined})),side=(id:string)=>edges.find(edge=>edge.dir==='from'&&edge.from===id||edge.dir==='to'&&edge.to===id)?.dir,marker=(dir?:string)=>dir?arrow+'-'+dir:arrow;
 const nodes=roadmap.nodes.map(node=>{
  const paper=matchPaper(node,papers),progress=paper&&progressReady?paperProgress(paper,{problems,saved}):undefined;
  return {node,paper,progress,state:(!paper?'missing':progress?progress.state:'pending') as NodeState};
 });
 const stateLabel=(state:NodeState)=>({done:t('已完成','Solved'),started:t('进行中','In progress'),todo:t('未开始','Not started'),missing:t('暂无练习','No exercise yet'),pending:''})[state];
 const active=nodes.find(entry=>entry.node.id===selected);
 // Relation sentences of the selected node's incoming ('from') and outgoing ('to') edges, in edge order; a side without edges is left out.
 const relations=active?([['from',t('启发自','From'),roadmap.edges.filter(([,to])=>to===active.node.id)],['to',t('启发了','To'),roadmap.edges.filter(([from])=>from===active.node.id)]] as const).filter(([,,list])=>list.length>0):[];
 // Every node (all are arXiv papers) shows the community paper card with a bundled cover and its links under it, keyed by arXiv id
 // so no image or failed state outlives its paper.
 // A bare YYYY-MM-DD reads as UTC midnight (the day before west of UTC), so the date gets a local midnight and every time zone
 // shows the same calendar day.
 const arxiv=active?.node.arxiv?parseArxivUrl('https://arxiv.org/abs/'+active.node.arxiv):undefined,arxivMeta=arxiv&&roadmapPapers[arxiv.arxivId];
 const practiceHref=(paper:Paper)=>paperCollections.find(collection=>collection.id===collectionId)?.paperIds.includes(paper.id)?withLibraryContext('papers/'+paper.id,returnTo):'#/papers/'+paper.id;
 return <div className="collection-roadmap" ref={root}>
  <div className="roadmap-canvas-scroll" ref={scroller}>
   <div className="roadmap-canvas" role="group" aria-label={t('路线图','Roadmap')} style={{width:size.width,height:size.height}}>
    <svg aria-hidden="true" width={size.width} height={size.height}>
     <defs>{[undefined,'from','to'].map(dir=><marker key={marker(dir)} id={marker(dir)} className={dir?'roadmap-arrow '+dir:'roadmap-arrow'} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z"/></marker>)}</defs>
     {yearRows(roadmap).map(({year,row})=><text key={year} className="roadmap-year" x={8} y={ROADMAP.top+row*ROADMAP.rowHeight+ROADMAP.rowHeight/2} dominantBaseline="middle">{year}</text>)}
     {edges.map(({from,to,d,dir})=><path key={from+'>'+to} className="roadmap-edge" data-from={from} data-to={to} data-dir={dir} d={d} markerEnd={`url(#${marker(dir)})`} data-lit={dir?'':undefined} data-dimmed={selected&&!dir?'':undefined}/>)}
     {edges.filter(edge=>edge.dir).map(({from,to,d,dir})=><path key={from+'>'+to} className="roadmap-flow" data-dir={dir} data-from={from} data-to={to} d={d}/>)}
    </svg>
    {nodes.map(({node,progress,state})=>{const box=nodeBox(node),label=stateLabel(state),dimmed=!!lit&&!lit.has(node.id);return <Fragment key={node.id}>
     {node.start&&<span className="roadmap-start" data-side={tag.side} data-dimmed={dimmed?'':undefined} style={{left:tag.left,top:tag.top}}>{t('从这里开始','Start here')}</span>}
     <button type="button" ref={element=>{buttons.current[node.id]=element;}} className="roadmap-node" data-node={node.id} data-state={state} data-lineage={side(node.id)} data-dimmed={dimmed?'':undefined} tabIndex={dimmed?-1:undefined} aria-pressed={selected===node.id} aria-controls={selected===node.id?uid+'-panel':undefined} aria-describedby={selected===node.id?uid+'-sentence':undefined} aria-label={`${node.title}, ${node.year}${label?`, ${label}`:''}`} style={{left:box.x,top:box.y,width:box.width,height:box.height}} onClick={()=>setSelected(current=>current===node.id?undefined:node.id)}>
      <span className="roadmap-node-title">{node.title}</span>
      {progress&&<span className="roadmap-node-bar"><span style={{width:`${progress.total?progress.completed/progress.total*100:0}%`}}/></span>}
     </button>
    </Fragment>;})}
   </div>
  </div>
  {active&&<aside className="roadmap-panel" id={uid+'-panel'} aria-labelledby={uid+'-title'}>
   <h3 id={uid+'-title'}>{active.node.title}</h3>
   <p className="roadmap-meta">{active.node.venue.includes(String(active.node.year))?active.node.venue:`${active.node.year} · ${active.node.venue}`}</p>
   <p className="roadmap-sentence" id={uid+'-sentence'}>{t(active.node.sentence,active.node.sentence_en)}</p>
   {relations.map(([dir,label,list])=><div key={dir} className="roadmap-links" data-dir={dir}><p>{label}</p><ul>{list.map(([from,to,zh,en])=><li key={from+'>'+to}>{t(zh,en)}</li>)}</ul></div>)}
   {arxiv&&arxivMeta&&<PaperPreviewCard key={arxiv.arxivId} compact linksUnderCover url={arxiv.url} paper={{arxiv_id:arxiv.arxivId,url:arxiv.url,pdf_url:arxiv.pdfUrl,status:'ready',title:arxivMeta.title,authors:arxivMeta.authors,published:arxivMeta.published+'T00:00'}} thumbnail={`${import.meta.env.BASE_URL}roadmap-thumbnails/${arxiv.arxivId}.webp`}/>}
   {active.paper?<AppLink className="button roadmap-action" href={practiceHref(active.paper)}>{t('练习','Practice')}<ArrowRight size={16} aria-hidden="true"/></AppLink>:<>
    <p className="roadmap-missing">{t('暂无练习','No exercise yet')}</p>
    {(!publicSite||hostedAppAvailable)&&<AppLink className="button roadmap-action" href={hostedAppHref('#/submit')}>{t('投稿这篇论文','Submit this paper')}</AppLink>}
   </>}
  </aside>}
 </div>;
}
