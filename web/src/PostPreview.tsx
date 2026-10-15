import {memo,useMemo} from 'react';
import {Code2,Sigma,Table2} from 'lucide-react';
import {useI18n} from './i18n';
import {postPreview,type PostContentKind} from './postExcerpt';
import './post-preview.css';

const icons={code:Code2,math:Sigma,table:Table2};
const PostPreview=memo(function PostPreview({body,limit=210}:{body:string;limit?:number}){
 const {t,locale}=useI18n();
 const summary=useMemo(()=>postPreview(body,limit,locale==='en'?'[formula]':'[公式]'),[body,limit,locale]);
 const label=(kind:PostContentKind)=>({code:t('代码片段','Code'),math:t('公式','Math'),table:t('表格','Table')})[kind];
 return <>
  <p className="post-excerpt">{summary.text||(summary.unavailable?t('摘要暂不可用，打开查看正文。','The preview is unavailable. Open the post to keep reading.'):t('打开阅读全文，查看这份分享的完整内容。','Open the discussion to read the full post.'))}</p>
  {summary.kinds.length>0&&<ul className="post-content-signals" aria-label={t('内容包含','Post contains')}>{summary.kinds.map(kind=>{const Icon=icons[kind];return <li key={kind} data-content-kind={kind}><Icon size={12} aria-hidden="true"/>{label(kind)}</li>;})}</ul>}
 </>;
});
export default PostPreview;
