import {useId,type RefObject} from 'react';
import {ArrowDown} from 'lucide-react';
import CommunitySearch from './CommunitySearch';
import {useI18n} from './i18n';
import type {useDiscussionReplies} from './useDiscussionReplies';

type Props={title:string;list:ReturnType<typeof useDiscussionReplies>;total:number;controls:string;canAct:()=>boolean;inputRef:RefObject<HTMLInputElement|null>};
export default function CardCommentSearch({title,list,total,controls,canAct,inputRef}:Props){
 const {t}=useI18n(),help=useId(),permitted=()=>!list.disabled&&canAct();
 return <div className="card-comment-search">
  <CommunitySearch applied={list.query} readingRevision={list.searchReadingRevision} onApply={list.chooseQuery} disabled={list.disabled} canAct={permitted} controls={controls} inputRef={inputRef} describedBy={help} label={t(`搜索本帖评论：${title}`,`Search comments in “${title}”`)} placeholder={t('内容、代码、作者或机构…','Content, code, author or affiliation…')}/>
  <p id={help}>{t('搜索全部公开评论，多个关键词用空格分隔；不搜索草稿。','Search all public comments. Separate terms with spaces; drafts are excluded.')}</p>
  {!!list.query&&<div className="card-comment-search-summary"><span role="status">{t(`${total} 条评论中匹配 ${list.total} 条`,`${list.total} matching ${list.total===1?'comment':'comments'} out of ${total}`)}</span>{list.firstMatch&&<button type="button" disabled={list.disabled} onClick={()=>{if(permitted()&&list.firstMatch)list.seekSearchReply(list.firstMatch);}}><ArrowDown size={14} aria-hidden="true"/>{t('定位首条匹配','Go to first match')}</button>}</div>}
 </div>;
}
