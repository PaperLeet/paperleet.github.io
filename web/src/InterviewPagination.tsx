import {ChevronLeft,ChevronRight} from 'lucide-react';
import {useI18n} from './i18n';
import {MAX_INTERVIEW_PAGE} from './interviewNavigation';
import PageJump from './PageJump';
import './interview-pagination.css';

export default function InterviewPagination({page,total,onChange}:{page:number;total:number;onChange:(page:number)=>void}){
 const {t}=useI18n();
 const pages=Math.ceil(total/20),available=Math.min(pages,MAX_INTERVIEW_PAGE);
 return <nav className="interview-pagination" aria-label={t('面经分页','Experience pages')}>
  <button type="button" disabled={page===1} onClick={()=>onChange(page-1)}><ChevronLeft size={15}/>{t('上一页','Previous')}</button>
  <span aria-current="page">{page>pages?t(`请求第 ${page} 页 · 当前共 ${pages} 页`,`Requested page ${page} · ${pages} pages available`):t(`第 ${page} 页 · 共 ${pages} 页`,`Page ${page} of ${pages}`)}</span>
  <button type="button" disabled={page>=available} onClick={()=>onChange(page+1)}>{t('下一页','Next')}<ChevronRight size={15}/></button>
  {pages>=4&&<PageJump className="interview-page-jump" page={page} maxPage={available} onChange={onChange} label={t('跳转面经页码','Jump to experience page')}/>}
  {pages>MAX_INTERVIEW_PAGE&&<p className="interview-page-limit">{t(`最多可浏览前 ${MAX_INTERVIEW_PAGE} 页，请按公司、岗位或时间缩小范围。`,`The first ${MAX_INTERVIEW_PAGE} pages are available. Narrow the company, role or time range to find more relevant experiences.`)}</p>}
 </nav>;
}
