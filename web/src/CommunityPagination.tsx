import {ChevronLeft,ChevronRight} from 'lucide-react';
import {useI18n} from './i18n';
import {COMMUNITY_PAGE_SIZE as PAGE_SIZE,MAX_COMMUNITY_PAGE} from './communityNavigation';
import PageJump from './PageJump';
import './community-pagination.css';

export default function CommunityPagination({offset,total,count,onChange,label}:{offset:number;total:number;count:number;onChange:(offset:number)=>void;label:string}){
 const {t}=useI18n();
 const page=offset/PAGE_SIZE+1,pages=Math.max(1,Math.ceil(total/PAGE_SIZE)),available=Math.min(pages,MAX_COMMUNITY_PAGE),expired=page>1&&!count;
 if(total<=PAGE_SIZE&&!expired)return null;
 function go(next:number){
  if(!Number.isSafeInteger(next)||next<0||next%PAGE_SIZE||next/PAGE_SIZE+1>MAX_COMMUNITY_PAGE||next===offset)return;
  onChange(next);
 }
 return <nav className="community-pagination" aria-label={label}>
  <span aria-current="page"><span className="sr-only">{t(`第 ${page} 页，共 ${pages} 页；`,`Page ${page} of ${pages}; `)}</span><span className="community-pagination-range">{count?`${offset+1}–${Math.min(total,offset+count)} / ${total}`:`0 / ${total}`}</span></span>
  <button type="button" disabled={!offset} aria-label={t('上一页','Previous page')} onClick={()=>go(offset-PAGE_SIZE)}><ChevronLeft size={17}/></button>
  <button type="button" disabled={page>=available} aria-label={t('下一页','Next page')} onClick={()=>go(offset+PAGE_SIZE)}><ChevronRight size={17}/></button>
  {pages>=4&&<PageJump page={page} maxPage={available} onChange={next=>go((next-1)*PAGE_SIZE)} label={t('跳转列表页码','Jump to list page')}/>}
  {pages>MAX_COMMUNITY_PAGE&&<p className="community-page-limit">{t('结果较多，请使用搜索或筛选缩小范围。','There are too many results to browse at once. Narrow the search or filters.')}</p>}
  {expired&&<button type="button" className="community-page-recovery" onClick={()=>go(0)}>{t('返回第一页','Return to first page')}</button>}
 </nav>;
}
