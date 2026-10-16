import {memo,useMemo} from 'react';
import {Code2,Sigma,Table2} from 'lucide-react';
import {useI18n} from './i18n';
import {postPreview,type PostContentKind,type PostPreviewSummary} from './postExcerpt';
import './post-preview.css';

const icons={code:Code2,math:Sigma,table:Table2};
const PostPreview=memo(function PostPreview({body,limit=210}:{body:string;limit?:number}){
 const {locale}=useI18n();
 const summary=useMemo(()=>postPreview(body,limit,locale==='en'?'[formula]':'[公式]'),[body,limit,locale]);
 return <PostPreviewContent summary={summary}/>;
});
export function PostPreviewContent({summary,reply=false}:{summary:PostPreviewSummary;reply?:boolean}){
 const {t}=useI18n();
 const label=(kind:PostContentKind)=>({code:t('代码片段','Code'),math:t('公式','Math'),table:t('表格','Table')})[kind];
 const fallback=reply?t('展开回复，查看完整内容。','Expand to read the full reply.'):t('打开阅读全文，查看这份分享的完整内容。','Open the discussion to read the full post.');
 const unavailable=reply?t('摘要暂不可用，展开查看回复。','The preview is unavailable. Expand to read the reply.'):t('摘要暂不可用，打开查看正文。','The preview is unavailable. Open the post to keep reading.');
 return <>
  <p className="post-excerpt">{summary.text||(summary.unavailable?unavailable:fallback)}</p>
  {summary.kinds.length>0&&<ul className="post-content-signals" aria-label={t('内容包含','Post contains')}>{summary.kinds.map(kind=>{const Icon=icons[kind];return <li key={kind} data-content-kind={kind}><Icon size={12} aria-hidden="true"/>{label(kind)}</li>;})}</ul>}
 </>;
}
export default PostPreview;
