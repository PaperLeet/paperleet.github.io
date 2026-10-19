import {ChevronLeft,ChevronRight} from 'lucide-react';
import type {KeyboardEvent} from 'react';
import {useI18n} from './i18n';
import PageJump from './PageJump';
import type {LibraryPageSize} from './libraryNavigation';
import './library-pagination.css';

export type Props={page:number;size:LibraryPageSize;total:number;modules:boolean;listId:string;compact?:boolean;labels?:{nav:[string,string];jump:[string,string];count:[string,string]};onPage:(page:number)=>void;onSize:(size:LibraryPageSize)=>void};

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

/** Move focus within the page strip; Enter/Space still activate the chosen page. */
function focusPageButton(event:KeyboardEvent<HTMLDivElement>){
 if(event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
 if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
 const strip=event.currentTarget,target=event.target,view=strip.ownerDocument.defaultView;
 if(!view||!(target instanceof view.HTMLButtonElement)||target!==strip.ownerDocument.activeElement||!target.matches('.library-page-arrow,.library-page-number,.library-page-gap')||!strip.isConnected||strip.closest('[hidden],[inert],[aria-hidden="true"]'))return;
 const buttons=[...strip.querySelectorAll<HTMLButtonElement>('.library-page-arrow,.library-page-number,.library-page-gap')].filter(button=>!button.disabled&&view.getComputedStyle(button).display!=='none');
 const index=buttons.indexOf(target);
 if(index<0)return;
 const next=event.key==='Home'?0:event.key==='End'?buttons.length-1:Math.max(0,Math.min(buttons.length-1,index+(event.key==='ArrowRight'?1:-1)));
 event.preventDefault();
 buttons[next]?.focus({preventScroll:true});
}

export default function LibraryPagination({page,size,total,modules,listId,compact=false,labels,onPage,onSize}:Props){
 const {t}=useI18n(),pages=Math.max(1,Math.ceil(total/size)),valid=page>=1&&page<=pages,start=(page-1)*size+1,end=Math.min(page*size,total);
 if(!total)return null;
 const numbers=libraryPageNumbers(page,pages);
 const previous=<button type="button" className="library-page-arrow" aria-label={t('上一页','Previous page')} aria-controls={listId} title={t('上一页','Previous page')} disabled={!valid||page===1} onClick={()=>onPage(page-1)}><ChevronLeft size={17} aria-hidden="true"/>{!compact&&<span className="library-page-arrow-label">{t('上一页','Previous')}</span>}</button>;
 const next=<button type="button" className="library-page-arrow" aria-label={t('下一页','Next page')} aria-controls={listId} title={t('下一页','Next page')} disabled={!valid||page>=pages} onClick={()=>onPage(page+1)}>{!compact&&<span className="library-page-arrow-label">{t('下一页','Next')}</span>}<ChevronRight size={17} aria-hidden="true"/></button>;
 return <nav className={`library-pagination${compact?' library-pagination-compact':''}`} aria-label={labels?t(...labels.nav):compact?t('题库快捷翻页','Quick library pages'):t('题库分页','Library pages')}>
  {!compact&&<p className="library-page-range" role="status" aria-atomic="true">{valid?<>{t('显示','Showing')} <b>{start}–{end}</b> / {total} {labels?t(...labels.count):modules?t('道题','modules'):t('篇论文','papers')}</>:t(`第 ${page} 页已超出范围 · 共 ${pages} 页`,`Page ${page} is unavailable · ${pages} pages`)}</p>}
  <div className="library-page-buttons" onKeyDown={focusPageButton}>{previous}{compact?(pages>1?<PageJump key={page} page={page} maxPage={pages} onChange={onPage} label={labels?t(...labels.jump):t('快捷跳转题库页码','Quick jump to library page')} controls={listId} className="library-quick-jump"/>:<span className="library-page-current" aria-current={valid?'page':undefined}>{valid?`${page} / ${pages}`:t('页码已失效','Page unavailable')}</span>):numbers.map((n,index)=>{
   if(n==='gap'){
    const before=numbers[index-1] as number,after=numbers[index+1] as number;
    const target=Math.max(before+1,Math.min(after-1,page+(after<=page?-5:5)));
    const name=t(`跳至第 ${target} 页`,`Jump to page ${target}`);
    return <button type="button" className="library-page-gap" key={'gap'+index} aria-label={name} title={name} aria-controls={listId} onClick={()=>onPage(target)}>…</button>;
   }
   return <button type="button" className="library-page-number" key={n} aria-label={t(`第 ${n} 页`,`Page ${n}`)} aria-controls={listId} aria-current={n===page?'page':undefined} onClick={()=>{if(n!==page)onPage(n);}}>{n}</button>;
  })}{!compact&&<span className="library-page-mobile-current" aria-current={valid?'page':undefined}>{valid?t(`第 ${page} / ${pages} 页`,`Page ${page} of ${pages}`):t('选择有效页码','Choose an available page')}</span>}{next}</div>
  {!compact&&<div className="library-page-options"><label className="library-page-size"><span>{t('每页','Per page')}</span><select aria-label={t('每页显示数量','Results per page')} aria-controls={listId} value={size} onChange={event=>onSize(Number(event.target.value) as LibraryPageSize)}>{([20,50,100] as const).map(value=><option key={value} value={value}>{value}</option>)}</select></label>{pages>1&&<PageJump key={page} page={page} maxPage={pages} onChange={onPage} label={labels?t(...labels.jump):t('跳转题库页码','Jump to library page')} controls={listId}/>}</div>}
 </nav>;
}
