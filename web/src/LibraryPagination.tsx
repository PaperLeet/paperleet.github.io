import {ChevronLeft,ChevronRight} from 'lucide-react';
import {useI18n} from './i18n';
import PageJump from './PageJump';
import type {LibraryPageSize} from './libraryNavigation';
import './library-pagination.css';

export type Props={page:number;size:LibraryPageSize;total:number;modules:boolean;listId:string;compact?:boolean;onPage:(page:number)=>void;onSize:(size:LibraryPageSize)=>void};

/** A bounded page window, even for a very large catalog. */
export function libraryPageNumbers(page:number,pages:number):(number|'gap')[]{
 if(!Number.isSafeInteger(pages)||pages<1||!Number.isSafeInteger(page))return [];
 const current=Math.max(1,Math.min(page,pages));
 const values=new Set([1,pages]);
 const start=current<=4?2:current>=pages-3?Math.max(2,pages-4):current-1;
 const end=current<=4?Math.min(pages-1,5):current>=pages-3?pages-1:current+1;
 for(let n=start;n<=end;n++)values.add(n);
 const result:(number|'gap')[]=[];
 for(const n of [...values].sort((a,b)=>a-b)){
  const previous=result.at(-1);
  if(typeof previous==='number'&&n-previous>1)result.push(n-previous===2?previous+1:'gap');
  result.push(n);
 }
 return result;
}

export default function LibraryPagination({page,size,total,modules,listId,compact=false,onPage,onSize}:Props){
 const {t}=useI18n(),pages=Math.max(1,Math.ceil(total/size)),valid=page>=1&&page<=pages,start=(page-1)*size+1,end=Math.min(page*size,total);
 if(!total)return null;
 const previous=<button type="button" className="library-page-arrow" aria-label={t('上一页','Previous page')} aria-controls={listId} disabled={!valid||page===1} onClick={()=>onPage(page-1)}><ChevronLeft size={17} aria-hidden="true"/></button>;
 const next=<button type="button" className="library-page-arrow" aria-label={t('下一页','Next page')} aria-controls={listId} disabled={!valid||page>=pages} onClick={()=>onPage(page+1)}><ChevronRight size={17} aria-hidden="true"/></button>;
 return <nav className={`library-pagination${compact?' library-pagination-compact':''}`} aria-label={compact?t('题库快捷翻页','Quick library pages'):t('题库分页','Library pages')}>
  {!compact&&<p className="library-page-range" role="status" aria-atomic="true">{valid?<>{t('显示','Showing')} <b>{start}–{end}</b> / {total} {modules?t('道题','modules'):t('篇论文','papers')}</>:t(`第 ${page} 页已超出范围 · 共 ${pages} 页`,`Page ${page} is unavailable · ${pages} pages`)}</p>}
  <div className="library-page-buttons">{previous}{compact?<span className="library-page-current" aria-current={valid?'page':undefined}>{valid?`${page} / ${pages}`:t('页码已失效','Page unavailable')}</span>:libraryPageNumbers(page,pages).map((n,index)=>n==='gap'?<span className="library-page-gap" key={'gap'+index} aria-hidden="true">…</span>:<button type="button" className="library-page-number" key={n} aria-label={t(`第 ${n} 页`,`Page ${n}`)} aria-controls={listId} aria-current={n===page?'page':undefined} onClick={()=>{if(n!==page)onPage(n);}}>{n}</button>)}{!compact&&<span className="library-page-mobile-current" aria-current={valid?'page':undefined}>{valid?t(`第 ${page} / ${pages} 页`,`Page ${page} of ${pages}`):t('选择有效页码','Choose an available page')}</span>}{next}</div>
  {!compact&&<div className="library-page-options"><label className="library-page-size"><span>{t('每页','Per page')}</span><select aria-label={t('每页显示数量','Results per page')} aria-controls={listId} value={size} onChange={event=>onSize(Number(event.target.value) as LibraryPageSize)}>{([20,50,100] as const).map(value=><option key={value} value={value}>{value}</option>)}</select></label>{pages>1&&<PageJump key={`${page}:${pages}`} page={page} maxPage={pages} onChange={onPage} label={t('跳转题库页码','Jump to library page')} controls={listId}/>}</div>}
 </nav>;
}
