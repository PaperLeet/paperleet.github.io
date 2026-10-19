import {ChevronLeft,ChevronRight} from 'lucide-react';
import {useRef} from 'react';
import {useI18n} from './i18n';
import {COMMUNITY_PAGE_SIZE as PAGE_SIZE,MAX_COMMUNITY_PAGE} from './communityNavigation';
import PageJump from './PageJump';
import NumberedPages from './NumberedPages';
import {focusPaginationButton,isPaginationActive} from './paginationNavigation';
import './community-pagination.css';

export default function CommunityPagination({offset,total,count,onChange,label,controls,canAct}:{offset:number;total:number;count:number;onChange:(offset:number)=>void;label:string;controls?:string;canAct:()=>boolean}){
 const {t}=useI18n(),root=useRef<HTMLElement>(null);
 const page=offset/PAGE_SIZE+1,pages=Math.max(1,Math.ceil(total/PAGE_SIZE)),available=Math.min(pages,MAX_COMMUNITY_PAGE),expired=page>1&&!count,valid=page>=1&&page<=available;
 if(total<=PAGE_SIZE&&!expired)return null;
 function go(next:number){
  if(!isPaginationActive(root.current)||!canAct()||!Number.isSafeInteger(next)||next<0||next%PAGE_SIZE||next/PAGE_SIZE+1>available||next===offset)return;
  onChange(next);
 }
 return <nav ref={root} className="community-pagination" aria-label={label}>
  <span className="community-page-status"><span className="sr-only">{t(`第 ${page} 页，共 ${pages} 页；`,`Page ${page} of ${pages}; `)}</span><span className="community-pagination-range">{count?`${offset+1}–${Math.min(total,offset+count)} / ${total}`:`0 / ${total}`}</span></span>
  <div className="community-page-controls" onKeyDown={focusPaginationButton}>
   <button type="button" data-pagination-control="" disabled={!valid||page===1} aria-label={t('上一页','Previous page')} aria-controls={controls} onClick={()=>go(offset-PAGE_SIZE)}><ChevronLeft size={17} aria-hidden="true"/></button>
   <NumberedPages page={page} pages={available} totalPages={pages} controls={controls} onChange={next=>go((next-1)*PAGE_SIZE)}/>
   <button type="button" data-pagination-control="" disabled={!valid||page>=available} aria-label={t('下一页','Next page')} aria-controls={controls} onClick={()=>go(offset+PAGE_SIZE)}><ChevronRight size={17} aria-hidden="true"/></button>
  </div>
  {pages>=4&&<PageJump canAct={()=>isPaginationActive(root.current)&&canAct()} page={page} maxPage={available} controls={controls} onChange={next=>go((next-1)*PAGE_SIZE)} label={t('跳转列表页码','Jump to list page')}/>}
  {pages>MAX_COMMUNITY_PAGE&&<p className="community-page-limit">{t('结果较多，请使用搜索或筛选缩小范围。','There are too many results to browse at once. Narrow the search or filters.')}</p>}
  {expired&&<button type="button" className="community-page-recovery" aria-controls={controls} onClick={()=>go(0)}>{t('返回第一页','Return to first page')}</button>}
 </nav>;
}
