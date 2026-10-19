import {useI18n} from './i18n';
import {paginationWindow} from './paginationNavigation';
import './numbered-pages.css';

export default function NumberedPages({page,pages,totalPages=pages,onChange,controls}:{page:number;pages:number;totalPages?:number;onChange:(page:number)=>void;controls?:string}){
 const {t}=useI18n(),numbers=paginationWindow(page,pages),valid=page>=1&&page<=pages;
 return <div className="numbered-pages" data-wide={String(pages).length>5||undefined} role="group" aria-label={t('页码选择','Page numbers')}>
  {numbers.map((number,index)=>{
   if(number==='gap'){
    const before=numbers[index-1] as number,after=numbers[index+1] as number;
    const target=Math.max(before+1,Math.min(after-1,page+(after<=page?-5:5)));
    const label=t(`跳至第 ${target} 页`,`Jump to page ${target}`);
    return <button type="button" data-pagination-control="" className="numbered-page-gap" key={'gap'+index} aria-label={label} title={label} aria-controls={controls} onClick={()=>onChange(target)}>…</button>;
   }
   return <button type="button" data-pagination-control="" key={number} aria-label={t(`第 ${number} 页`,`Page ${number}`)} title={t(`第 ${number} 页`,`Page ${number}`)} aria-controls={controls} aria-current={number===page?'page':undefined} onClick={()=>{if(number!==page)onChange(number);}}>{number}</button>;
  })}
  <span className="numbered-pages-current" aria-current={valid?'page':undefined}>{valid?t(`第 ${page} / ${totalPages} 页`,`Page ${page} of ${totalPages}`):t('选择有效页码','Choose an available page')}</span>
 </div>;
}
