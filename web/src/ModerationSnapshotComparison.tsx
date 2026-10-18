import {useMemo} from 'react';
import {useI18n} from './i18n';
import {communityCategories} from './communityCategories';
import type {ReviewPost} from './communityReports';
import {diffLineContent,moderationDiffRows,moderationTextDiff,MAX_DIFF_ROWS} from './moderationTextDiff';
import './moderation-snapshot-comparison.css';

export default function ModerationSnapshotComparison({before,after}:{before:ReviewPost;after:ReviewPost}){
 const {t}=useI18n(),diff=useMemo(()=>moderationTextDiff(before.body,after.body),[before.body,after.body]);
 const display=useMemo(()=>diff?moderationDiffRows(diff):undefined,[diff]);
 const category=(id:string)=>{const value=communityCategories.find(row=>row.id===id);return value?t(value.zh,value.en):id;};
 const sameTitle=before.title===after.title,sameCategory=before.category===after.category,sameBody=before.body===after.body;
 return <div className="moderation-comparison">
  <p className="moderation-comparison-scope">{t('对比举报时快照与当前已加载版本的标题、正文和分类。','Compare the reported snapshot with the currently loaded title, body and category.')}</p>
  <section aria-label={t('标题与分类变更','Title and category changes')} className="moderation-comparison-fields">
   <div><h4>{t('标题','Title')}</h4>{sameTitle?<p>{t('未改变','Unchanged')} · {after.title}</p>:<dl><div><dt>{t('举报时','Reported')}</dt><dd><del>{before.title}</del></dd></div><div><dt>{t('当前','Current')}</dt><dd><ins>{after.title}</ins></dd></div></dl>}</div>
   <div><h4>{t('分类','Category')}</h4>{sameCategory?<p>{t('未改变','Unchanged')} · {category(after.category)}</p>:<dl><div><dt>{t('举报时','Reported')}</dt><dd>{category(before.category)}</dd></div><div><dt>{t('当前','Current')}</dt><dd>{category(after.category)}</dd></div></dl>}</div>
  </section>
  <section aria-label={t('正文变更','Body changes')}><div className="moderation-comparison-body-heading"><h4>{t('正文','Body')}</h4>{diff&&<span>{t(`新增 ${diff.added} 行 · 删除 ${diff.removed} 行`,`${diff.added} added · ${diff.removed} removed`)}</span>}</div>
   {sameBody?<p>{t('正文未改变。','The body is unchanged.')}</p>:!diff?<p role="status">{t('内容超出逐行对比范围，请查看下方两份完整原文。','This content exceeds the line comparison limit. Read both complete sources below.')}</p>:<>
    {diff.mode==='block'&&<p role="status">{t('变更区块较大，完整标出该区块的旧行与新行；其中未逐行匹配的内容也会计入新增和删除。','The changed block is large. Its old and new lines are shown as a complete replacement; unmatched content within the block is included in the counts.')}</p>}
    <div className="moderation-comparison-scroll" tabIndex={0} role="region" aria-label={t('正文逐行变更，可滚动阅读','Scrollable body line changes')}><table><thead><tr><th scope="col">{t('原行','Old')}</th><th scope="col">{t('新行','New')}</th><th scope="col">{t('内容变更','Content changes')}</th></tr></thead><tbody>{display?.rows.map((line,index)=>line.kind==='gap'?<tr className="moderation-diff-gap" key={'gap:'+index}><td colSpan={3}>{t(`省略 ${line.count} 行未改变的内容`,`${line.count} unchanged lines omitted`)}</td></tr>:<tr className={'moderation-diff-'+line.kind} key={line.kind+':'+line.before+':'+line.after}><td>{line.before??'—'}</td><td>{line.after??'—'}</td><td><span className="moderation-diff-mark" aria-label={line.kind==='added'?t('新增','Added'):line.kind==='removed'?t('删除','Removed'):t('未改变','Unchanged')}>{line.kind==='added'?'+':line.kind==='removed'?'−':' '}</span><code>{diffLineContent(line.text).text||'\u00a0'}</code><small title={t('行结束符；EOF 表示末尾无换行','Line ending; EOF means no final newline')}>{diffLineContent(line.text).ending}</small></td></tr>)}</tbody></table></div>
    {display?.truncated&&<p role="status">{t(`逐行预览显示前 ${MAX_DIFF_ROWS} 行。请展开完整原文查看其余内容。`,`The line preview shows the first ${MAX_DIFF_ROWS} rows. Expand the complete sources to read the rest.`)}</p>}
   </>}
  </section>
  {sameTitle&&sameCategory&&sameBody&&<p role="status">{t('标题、正文和分类一致，但版本仍有变化。请打开原帖检查面经、论文链接等其他信息。','The title, body and category match, but the version changed. Open the original post to check other information such as interview details or paper links.')}</p>}
  <details className="moderation-comparison-sources"><summary>{t('查看两份完整原文','Read both complete sources')}</summary><div><section aria-label={t('举报时完整正文','Complete reported body')}><h5>{t('举报时正文','Reported body')}</h5><pre tabIndex={0}>{before.body}</pre></section><section aria-label={t('当前完整正文','Complete current body')}><h5>{t('当前已加载正文','Currently loaded body')}</h5><pre tabIndex={0}>{after.body}</pre></section></div></details>
 </div>;
}
