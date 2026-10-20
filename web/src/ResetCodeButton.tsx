import {useEffect,useId,useRef,useState} from 'react';
import {Download,RotateCcw} from 'lucide-react';
import {useI18n} from './i18n';
import {draftSnapshotMatches,type DraftExpectation} from './practiceRecords';
import type {Problem,Saved} from './types';
import {controlIsActive} from './activeControls';
import ResetCodePreview from './ResetCodePreview';
import './reset-code-review.css';

type Review={expected:DraftExpectation;source:string;starter:string;problem:string;href:string;sequence:number};
type Props={owner:string;problem:Problem;saved:Saved;ready:boolean;busy:boolean;conflict:boolean;failed:boolean;signedIn:boolean;canReview?:()=>boolean;onReset:(expected:DraftExpectation)=>boolean};
export function ResetCodeButton(props:Props){return <ResetControl key={JSON.stringify([props.owner,props.problem.id])} {...props}/>;}
function ResetControl({owner,problem,saved,ready,busy,conflict,failed,signedIn,canReview,onReset}:Props){
 const {t}=useI18n(),[review,setReview]=useState<Review>(),[error,setError]=useState(false),[download,setDownload]=useState<'idle'|'started'|'failed'>('idle');
 const dialog=useRef<HTMLDialogElement>(null),root=useRef<HTMLButtonElement>(null),cancel=useRef<HTMLButtonElement>(null),titleId=useId(),sequence=useRef(0);
 const renderHref=typeof window==='undefined'?'':window.location.href;
 const source=saved.code??problem.starter,unavailable=!ready||busy||conflict||failed;
 const matches=!!review&&review.expected.owner===owner&&review.problem===problem.id&&review.starter===problem.starter&&draftSnapshotMatches(saved,review.expected);
 const same=review?.source===review?.starter;
 const inScope=()=>!!review&&review.expected.owner===owner&&review.problem===problem.id&&review.href===location.href&&canReview?.()!==false&&controlIsActive(dialog.current)&&!dialog.current?.closest('details:not([open]),dialog:not([open])')&&dialog.current?.open===true;
 function capture(){
  if(unavailable||!controlIsActive(root.current)||root.current?.closest('details:not([open]),dialog:not([open])')||canReview?.()===false)return;
  setError(false);setDownload('idle');setReview({expected:{owner,code:saved.code,updated:saved.updated},source,starter:problem.starter,problem:problem.id,href:location.href,sequence:++sequence.current});
 }
 useEffect(()=>{
  const node=dialog.current;
  if(review&&review.href===location.href&&node){if(!node.open){node.showModal();cancel.current?.focus();}}
  else if(node?.open)node.close();
  if(review&&review.href!==location.href)setReview(undefined);
 },[review,renderHref]);
 useEffect(()=>{
  if(!review)return;
  const leave=()=>{if(location.href!==review.href)setReview(undefined);};
  window.addEventListener('hashchange',leave);window.addEventListener('popstate',leave);
  return()=>{window.removeEventListener('hashchange',leave);window.removeEventListener('popstate',leave);};
 },[review]);
 function reset(){
  if(!inScope()||unavailable||!matches||same)return;
  try{if(onReset(review!.expected))setReview(undefined);else setError(true);}catch{setError(true);}
 }
 function downloadReviewed(){
  if(!inScope())return;
  let url:string|undefined;
  try{
   url=URL.createObjectURL(new Blob([review!.source],{type:'text/x-python;charset=utf-8'}));
   const link=document.createElement('a');link.href=url;link.download=`paperleet-${problem.number}-before-reset.py`;link.click();setDownload('started');
  }catch{setDownload('failed');}
  finally{if(url){const created=url;window.setTimeout(()=>URL.revokeObjectURL(created),0);}}
 }
 return <>
  <button ref={root} className="icon-button" type="button" disabled={unavailable||source===problem.starter} title={t('核对并恢复初始代码','Review and restore starter')} aria-label={t('恢复初始代码','Restore starter code')} onClick={capture}><RotateCcw size={14}/></button>
  <dialog ref={dialog} className="reset-code-dialog" aria-labelledby={titleId} aria-describedby={titleId+'-description'} onCancel={()=>setReview(undefined)}>
   {review&&review.expected.owner===owner&&review.problem===problem.id&&review.href===location.href&&<>
    <h2 id={titleId}>{t('恢复本题初始代码？','Restore this module’s starter?')}</h2>
    <p id={titleId+'-description'}>{t('确认后将替换编辑器中的代码。已提交的解答和成绩会保留。','This replaces the editor code. Submitted solutions and results are kept.')}</p>
    <p>{signedIn?t('账号草稿会同步更新。之前提交的代码可在「我的提交」中查看。','Your account draft will sync. Previous submitted code is available in My submissions.'):t('这会替换当前浏览器中的草稿。如需保留当前代码，请先复制或下载。','This replaces the draft in this browser. Copy or download your current code first if you want to keep it.')}</p>
    <div className="reset-code-backup"><button className="button subtle" type="button" onClick={downloadReviewed}><Download size={15} aria-hidden="true"/>{t('下载核对中的代码','Download reviewed source')}</button><span>{t('可先保留这份代码，再决定是否恢复。','Keep this source before deciding to reset.')}</span></div>
    {download!=='idle'&&<p className="reset-code-download-status" role={download==='failed'?'alert':'status'}>{download==='started'?t('下载已交给浏览器，请确认文件已保存。恢复仍需单独确认。','Download started. Check that the file saved. Reset still needs separate confirmation.'):t('暂时无法启动下载。可展开完整代码，手动选择并复制后再操作。','Download could not start. Expand the complete source and copy it manually before continuing.')}</p>}
    <ResetCodePreview key={review.sequence} source={review.source} starter={review.starter} canAct={inScope}/>
    {!matches&&<p role="alert">{t('代码或工作区已变化，请重新核对后再恢复。预览与下载仍对应上次核对的代码。','The code or workspace changed. Review the latest source before restoring. Preview and download still use the reviewed snapshot.')}</p>}
    {unavailable&&<p role="status">{t('请先等待保存完成，或解决工作区的保存错误和草稿冲突。','Wait for saving to finish, or resolve workspace save errors and draft conflicts first.')}</p>}
    {error&&<p role="alert">{t('未能完成恢复。请检查当前草稿后重试。','Restore could not finish. Check the current draft and retry.')}</p>}
    <div className="reset-code-actions"><button ref={cancel} className="button subtle" type="button" onClick={()=>setReview(undefined)}>{t('保留当前代码','Keep current code')}</button>{!matches&&<button className="button subtle" type="button" disabled={unavailable} onClick={capture}>{t('核对最新代码','Review latest source')}</button>}<button className="button" type="button" disabled={unavailable||!matches||same} onClick={reset}>{t('恢复初始代码','Restore starter')}</button></div>
   </>}
  </dialog>
 </>;
}
