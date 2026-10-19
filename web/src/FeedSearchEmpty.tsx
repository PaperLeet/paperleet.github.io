import {Search} from 'lucide-react';
import {useI18n} from './i18n';
import './feed-search-empty.css';

type Kind='discussions'|'solutions'|'experiences';

/** A successful search with zero matches, distinct from access and request failures. */
export default function FeedSearchEmpty({kind,saved=false,onClear,onReview}:{kind:Kind;saved?:boolean;onClear:()=>void;onReview:()=>void}){
 const {t}=useI18n();
 const title=kind==='experiences'
  ? saved?t('没有匹配的收藏面经','No saved experiences match this search'):t('没有找到相关面经','No matching experiences')
  : kind==='solutions'
   ? saved?t('没有匹配的收藏题解','No saved solutions match this search'):t('没有找到相关题解','No matching solutions')
   : saved?t('没有匹配的收藏讨论','No saved discussions match this search'):t('没有找到相关讨论','No matching discussions');
 return <div className="feed-search-empty">
  <span className="feed-search-empty-icon" aria-hidden="true"><Search size={25}/></span>
  <h3>{title}</h3>
  <p>{t('清空关键词后，继续查看当前筛选范围内的内容；也可以调整其他筛选条件。','Clear the keywords to browse within your current filters, or adjust the other filters.')}</p>
  <div className="feed-search-empty-actions"><button type="button" className="button" onClick={onClear}>{t('清空搜索','Clear search')}</button><button type="button" className="button subtle" onClick={onReview}>{t('调整筛选','Adjust filters')}</button></div>
 </div>;
}
