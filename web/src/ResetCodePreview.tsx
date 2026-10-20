import {useMemo,useState} from 'react';
import {useI18n} from './i18n';
import {diffLineContent,moderationDiffRows,moderationTextDiff,MAX_DIFF_ROWS} from './moderationTextDiff';

type Props={source:string;starter:string;canAct:()=>boolean};
export default function ResetCodePreview({source,starter,canAct}:Props){
 const {t}=useI18n(),[visited,setVisited]=useState(false);
 const guard=(event:import('react').MouseEvent<HTMLElement>)=>{if(!canAct())event.preventDefault();};
 return <div className="reset-code-preview">
  <details className="reset-code-diff" onToggle={event=>{if(event.currentTarget.open&&canAct())setVisited(true);}}>
   <summary onClick={guard}>{t('对比这次恢复的代码变化','Compare reset changes')}</summary>
   {visited&&<ResetDiff source={source} starter={starter}/>}
  </details>
  <div className="reset-code-full-sources">
   <details><summary onClick={guard}>{t('查看将被替换的代码','Review code to be replaced')}</summary><pre tabIndex={0} aria-label={t('核对中的完整代码','Complete reviewed source')}>{source||t('空文件（0 字节）','Empty file (0 bytes)')}</pre></details>
   <details><summary onClick={guard}>{t('查看恢复后的初始代码','Preview replacement starter')}</summary><pre tabIndex={0} aria-label={t('恢复后的完整初始代码','Complete replacement starter')}>{starter||t('空文件（0 字节）','Empty file (0 bytes)')}</pre></details>
  </div>
 </div>;
}
function ResetDiff({source,starter}:Pick<Props,'source'|'starter'>){
 const {t}=useI18n(),same=source===starter,diff=useMemo(()=>same?undefined:moderationTextDiff(source,starter),[source,starter,same]),display=useMemo(()=>diff?moderationDiffRows(diff):undefined,[diff]);
 if(same)return <p role="status">{t('核对的代码已与初始代码一致，无需恢复。','The reviewed source already matches the starter. No reset is needed.')}</p>;
 if(!diff)return <p role="status">{t('代码较长，未生成逐行预览。可展开下方两份完整代码核对，或先下载核对中的代码。','This source exceeds the line-preview limit. Review both complete sources below or download the reviewed source first.')}</p>;
 return <>
  <p className="reset-diff-count">{t(`将移除 ${diff.removed} 行 · 换入 ${diff.added} 行`,`${diff.removed} lines removed · ${diff.added} lines restored`)}</p>
  {diff.mode==='block'&&<p role="status">{t('变更区块较大，以下按整块替换展示；行数不代表最小编辑量。','The changed block is large and is shown as a complete replacement. Counts are not a minimal edit estimate.')}</p>}
  <div className="reset-diff-scroll" tabIndex={0} role="region" aria-label={t('恢复代码的逐行对比，可滚动阅读','Scrollable reset line comparison')}><table><thead><tr><th scope="col">{t('当前行','Current')}</th><th scope="col">{t('初始行','Starter')}</th><th scope="col">{t('代码变化','Code changes')}</th></tr></thead><tbody>{display?.rows.map((line,index)=>{
   if(line.kind==='gap')return <tr className="reset-diff-gap" key={'gap:'+index}><td colSpan={3}>{t(`省略 ${line.count} 行相同代码`,`${line.count} unchanged lines omitted`)}</td></tr>;
   const value=diffLineContent(line.text);
   return <tr className={'reset-diff-'+line.kind} key={line.kind+':'+line.before+':'+line.after}><td>{line.before??'—'}</td><td>{line.after??'—'}</td><td><span className="reset-diff-mark" aria-label={line.kind==='removed'?t('将移除','Removed'):line.kind==='added'?t('换入初始代码','Restored from starter'):t('相同代码','Unchanged')}>{line.kind==='removed'?'−':line.kind==='added'?'+':' '}</span><code>{value.text||'\u00a0'}</code><small title={t('换行符；EOF 表示末尾没有换行','Line ending; EOF means no final newline')}>{value.ending}</small></td></tr>;
  })}</tbody></table></div>
  {display?.truncated&&<p role="status">{t(`预览最多显示 ${MAX_DIFF_ROWS} 行；其余内容请查看完整代码。`,`The preview shows up to ${MAX_DIFF_ROWS} rows. Review the complete source for the rest.`)}</p>}
 </>;
}
