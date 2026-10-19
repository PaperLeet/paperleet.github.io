import {ChevronLeft,ChevronRight} from 'lucide-react';
import {useRef} from 'react';
import {useI18n} from './i18n';
import {MAX_INTERVIEW_PAGE} from './interviewNavigation';
import PageJump from './PageJump';
import NumberedPages from './NumberedPages';
import {focusPaginationButton,isPaginationActive} from './paginationNavigation';
import './interview-pagination.css';

export default function InterviewPagination({page,total,onChange,controls,canAct}:{page:number;total:number;onChange:(page:number)=>void;controls?:string;canAct:()=>boolean}){
 const {t}=useI18n(),root=useRef<HTMLElement>(null);
 const pages=Math.max(1,Math.ceil(total/20)),available=Math.min(pages,MAX_INTERVIEW_PAGE),valid=page>=1&&page<=available;
 function go(next:number){if(isPaginationActive(root.current)&&canAct()&&Number.isSafeInteger(next)&&next>=1&&next<=available&&next!==page)onChange(next);}
 return <nav ref={root} className="interview-pagination" aria-label={t('面经分页','Experience pages')}>
  <span className="interview-page-status">{page>pages?t(`请求第 ${page} 页 · 当前共 ${pages} 页`,`Requested page ${page} · ${pages} pages available`):t(`第 ${page} 页 · 共 ${pages} 页`,`Page ${page} of ${pages}`)}</span>
  <div className="interview-page-controls" onKeyDown={focusPaginationButton}>
   <button type="button" data-pagination-control="" disabled={!valid||page===1} aria-label={t('上一页','Previous page')} aria-controls={controls} onClick={()=>go(page-1)}><ChevronLeft size={15} aria-hidden="true"/><span className="pagination-arrow-label">{t('上一页','Previous')}</span></button>
   <NumberedPages page={page} pages={available} totalPages={pages} controls={controls} onChange={go}/>
   <button type="button" data-pagination-control="" disabled={!valid||page>=available} aria-label={t('下一页','Next page')} aria-controls={controls} onClick={()=>go(page+1)}><span className="pagination-arrow-label">{t('下一页','Next')}</span><ChevronRight size={15} aria-hidden="true"/></button>
  </div>
  {pages>=4&&<PageJump className="interview-page-jump" page={page} maxPage={available} controls={controls} onChange={go} label={t('跳转面经页码','Jump to experience page')}/>}
  {pages>MAX_INTERVIEW_PAGE&&<p className="interview-page-limit">{t(`最多可浏览前 ${MAX_INTERVIEW_PAGE} 页，请按公司、岗位或时间缩小范围。`,`The first ${MAX_INTERVIEW_PAGE} pages are available. Narrow the company, role or time range to find more relevant experiences.`)}</p>}
 </nav>;
}
