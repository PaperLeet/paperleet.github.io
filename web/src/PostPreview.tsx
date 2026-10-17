import {memo,useMemo} from 'react';
import {Code2,Sigma,Table2} from 'lucide-react';
import {useI18n} from './i18n';
import {postPreview,type PostContentKind,type PostPreviewSummary} from './postExcerpt';
import PostSearchMatch from './PostSearchMatch';
import './post-preview.css';

const icons={code:Code2,math:Sigma,table:Table2};
const PostPreview=memo(function PostPreview({body,limit=210,query}:{body:string;limit?:number;query?:string}){
 const {locale}=useI18n();
 const summary=useMemo(()=>postPreview(body,limit,locale==='en'?'[formula]':'[公式]',query),[body,limit,locale,query]);
 return <PostPreviewContent summary={summary} query={query}/>;
});
export function PostPreviewContent({summary,reply=false,inlinePost=false,query}:{summary:PostPreviewSummary;reply?:boolean;inlinePost?:boolean;query?:string}){
 const {t}=useI18n();
 const label=(kind:PostContentKind)=>({code:t('代码片段','Code'),math:t('公式','Math'),table:t('表格','Table')})[kind];
 const fallback=reply?t('展开回复，查看完整内容。','Expand to read the full reply.'):inlinePost?t('展开全文，查看这份分享的完整内容。','Expand to read the full post.'):t('打开阅读全文，查看这份分享的完整内容。','Open the discussion to read the full post.');
 const unavailable=reply?t('摘要暂不可用，展开查看回复。','The preview is unavailable. Expand to read the reply.'):inlinePost?t('摘要暂不可用，展开查看正文。','The preview is unavailable. Expand to keep reading.'):t('摘要暂不可用，打开查看正文。','The preview is unavailable. Open the post to keep reading.');
 const contextual=!!summary.search&&summary.search!=='source'&&summary.search!=='outside';
 const origin=summary.search==='code'?t('代码片段','Code excerpt'):summary.search==='math'?t('公式源码','Math source'):summary.search==='table'?t('表格内容','Table excerpt'):undefined;
 const kinds=summary.kinds.filter(kind=>kind!==summary.search);
 return <>
  <p className={'post-excerpt'+(contextual?' has-search-match':'')} data-search-context={summary.search}>{origin&&<span className="post-search-origin">{origin}</span>}{contextual?<PostSearchMatch text={summary.text} query={query} range={summary.searchRange}/>:summary.text||(summary.unavailable?unavailable:fallback)}</p>
  {summary.search==='source'&&<p className="post-search-note">{t('关键词出现在链接或格式源码中。','The keyword appears in a link or formatting source.')}</p>}
  {summary.search==='outside'&&<p className="post-search-note">{t('关键词位于正文其他位置，可展开继续阅读。','The keyword is elsewhere in this post. Expand to keep reading.')}</p>}
  {kinds.length>0&&<ul className="post-content-signals" aria-label={t('内容包含','Post contains')}>{kinds.map(kind=>{const Icon=icons[kind];return <li key={kind} data-content-kind={kind}><Icon size={12} aria-hidden="true"/>{label(kind)}</li>;})}</ul>}
 </>;
}
export default PostPreview;
