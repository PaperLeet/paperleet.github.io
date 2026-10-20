import {useRef} from 'react';
import {controlIsActive} from './activeControls';
import {ChevronLeft,ChevronRight,ChevronsLeft,ChevronsRight} from 'lucide-react';
import {paginationWindow as libraryPageNumbers,focusPaginationButton} from './paginationNavigation';
export {paginationWindow as libraryPageNumbers} from './paginationNavigation';
import {useI18n} from './i18n';
import PageJump from './PageJump';
import type {LibraryPageSize} from './libraryNavigation';
import './library-pagination.css';

export type Props={page:number;size:LibraryPageSize;total:number;modules:boolean;listId:string;compact?:boolean;labels?:{nav:[string,string];jump:[string,string];count:[string,string]};onPage:(page:number)=>void;onSize:(size:LibraryPageSize)=>void;canAct?:()=>boolean};

export default function LibraryPagination({page,size,total,modules,listId,compact=false,labels,onPage,onSize,canAct}:Props){
 const root=useRef<HTMLElement>(null);
 const active=()=>controlIsActive(root.current)&&canAct?.()!==false;
 const choose=(next:number)=>{if(active())onPage(next);};
 const {t}=useI18n(),pages=Math.max(1,Math.ceil(total/size)),valid=page>=1&&page<=pages,start=(page-1)*size+1,end=Math.min(page*size,total);
 const singlePage=valid&&pages===1;
 if(!total)return null;
 const numbers=libraryPageNumbers(page,pages);
 const count=labels?t(...labels.count):modules?t('道题','modules'):t('篇论文','papers');
 const pageHint=(target:number)=>target>=1&&target<=pages?t(`第 ${target} / ${pages} 页 · ${((target-1)*size)+1}–${Math.min(target*size,total)} / ${total} ${count}`,`Page ${target} of ${pages} · ${((target-1)*size)+1}–${Math.min(target*size,total)} of ${total} ${count}`):undefined;
 const boundary=(last:boolean)=>{const target=last?pages:1,name=last?t('末页','Last page'):t('首页','First page'),Icon=last?ChevronsRight:ChevronsLeft;return <button type="button" data-pagination-control="" className="library-page-boundary" aria-label={name} aria-controls={listId} title={pageHint(target)} disabled={!valid||page===target} onClick={()=>choose(target)}><Icon size={17} aria-hidden="true"/></button>;};
 const pageSize=<label className="library-page-size"><span>{t('每页','Per page')}</span><select aria-label={t('每页显示数量','Results per page')} aria-controls={listId} value={size} onChange={event=>{if(active())onSize(Number(event.target.value) as LibraryPageSize);}}>{([20,50,100] as const).map(value=><option key={value} value={value}>{value}</option>)}</select></label>;
 const previous=<button type="button" data-pagination-control="" className="library-page-arrow" aria-label={t('上一页','Previous page')} aria-controls={listId} title={!valid?t('请选择有效页码','Choose an available page'):pageHint(page-1)||t('已在首页','Already on the first page')} disabled={!valid||page===1} onClick={()=>choose(page-1)}><ChevronLeft size={17} aria-hidden="true"/>{!compact&&<span className="library-page-arrow-label">{t('上一页','Previous')}</span>}</button>;
 const next=<button type="button" data-pagination-control="" className="library-page-arrow" aria-label={t('下一页','Next page')} aria-controls={listId} title={!valid?t('请选择有效页码','Choose an available page'):pageHint(page+1)||t('已在末页','Already on the last page')} disabled={!valid||page>=pages} onClick={()=>choose(page+1)}>{!compact&&<span className="library-page-arrow-label">{t('下一页','Next')}</span>}<ChevronRight size={17} aria-hidden="true"/></button>;
 return <nav ref={root} className={`library-pagination${compact?' library-pagination-compact':''}`} data-single-page={singlePage||undefined} aria-label={labels?t(...labels.nav):compact?t('题库快捷翻页','Quick library pages'):t('题库分页','Library pages')}>
  {!compact&&<p className="library-page-range" role="status" aria-atomic="true">{valid?<>{t('显示','Showing')} <b>{start}–{end}</b> / {total} {count}<span className="library-page-total">{t(`第 ${page} / ${pages} 页`,`Page ${page} of ${pages}`)}</span></>:t(`第 ${page} 页已超出范围 · 共 ${pages} 页`,`Page ${page} is unavailable · ${pages} pages`)}</p>}
  {compact&&pageSize}
  {!singlePage&&<div className="library-page-buttons" data-wide={String(pages).length>5||undefined} onKeyDown={focusPaginationButton}>{!compact&&boundary(false)}{previous}{compact?(pages>1?<PageJump key={page} page={page} maxPage={pages} onChange={choose} canAct={active} label={labels?t(...labels.jump):t('快捷跳转题库页码','Quick jump to library page')} controls={listId} className="library-quick-jump"/>:<span className="library-page-current" aria-current={valid?'page':undefined}>{valid?`${page} / ${pages}`:t('页码已失效','Page unavailable')}</span>):numbers.map((n,index)=>{
   if(n==='gap'){
    const before=numbers[index-1] as number,after=numbers[index+1] as number;
    const target=Math.max(before+1,Math.min(after-1,page+(after<=page?-5:5)));
    const name=t(`跳至第 ${target} 页`,`Jump to page ${target}`);
    return <button type="button" data-pagination-control="" className="library-page-gap" key={'gap'+index} aria-label={name} title={pageHint(target)} aria-controls={listId} onClick={()=>choose(target)}>…</button>;
   }
   return <button type="button" data-pagination-control="" className="library-page-number" key={n} aria-label={t(`第 ${n} 页`,`Page ${n}`)} title={pageHint(n)} aria-controls={listId} aria-current={n===page?'page':undefined} onClick={()=>{if(n!==page)choose(n);}}>{n}</button>;
  })}{!compact&&<span className="library-page-mobile-current" aria-current={valid?'page':undefined}>{valid?t(`第 ${page} / ${pages} 页`,`Page ${page} of ${pages}`):t('选择有效页码','Choose an available page')}</span>}{next}{!compact&&boundary(true)}</div>}
  {!compact&&<div className="library-page-options">{pageSize}{pages>1&&<PageJump key={page} page={page} maxPage={pages} onChange={choose} canAct={active} label={labels?t(...labels.jump):t('跳转题库页码','Jump to library page')} controls={listId}/>}</div>}
 </nav>;
}
