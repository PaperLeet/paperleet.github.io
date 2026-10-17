import {useEffect,useId,useLayoutEffect,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import {Flag,X} from 'lucide-react';
import {useI18n} from './i18n';
import {errorMessage,SocialError,socialApi,useAccount} from './social';
import {signInHref} from './accountNavigation';
import {confirmedReport,reportReasons,reportRequestId,type OwnReport,type ReportCommand,type ReportReason} from './communityReports';
import './community-reports.css';

export default function PostReport({postId,returnTo,own=false,disabled=false}:{postId:string;returnTo:string;own?:boolean;disabled?:boolean}){
 const {user,loading,capabilities}=useAccount(),{t}=useI18n();
 if(loading||capabilities?.community_reports!==true||own)return null;
 if(!user)return <a className="post-report-trigger" href={signInHref(returnTo)}><Flag size={14} aria-hidden="true"/>{t('举报','Report')}</a>;
 return <ReportForm key={`${user.id}:${postId}`} owner={user.id} postId={postId} disabled={disabled}/>;
}
function ReportForm({owner,postId,disabled}:{owner:string;postId:string;disabled:boolean}){
 const {t}=useI18n(),{refresh}=useAccount(),id=useId(),trigger=useRef<HTMLButtonElement>(null),dialog=useRef<HTMLDialogElement>(null),mounted=useRef(true),sequence=useRef(0),generation=useRef(0),working=useRef(false);
 const [open,setOpen]=useState(false),[known,setKnown]=useState<OwnReport>(),[loading,setLoading]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState(''),[rejected,setRejected]=useState(false);
 const [reason,setReason]=useState<ReportReason>('spam'),[details,setDetails]=useState(''),[command,setCommand]=useState<ReportCommand>();
 const origin=useRef(''),confirmed=useRef(false),path=`community/posts/${encodeURIComponent(postId)}/report`;
 useEffect(()=>{mounted.current=true;const leave=()=>{sequence.current++;generation.current++;confirmed.current=false;setOpen(false);setKnown(undefined);setCommand(undefined);setError('');setDetails('');setReason('spam');setLoading(false);};window.addEventListener('hashchange',leave);window.addEventListener('popstate',leave);return()=>{mounted.current=false;sequence.current++;generation.current++;window.removeEventListener('hashchange',leave);window.removeEventListener('popstate',leave);};},[]);
 useLayoutEffect(()=>{if(open&&dialog.current){if(typeof dialog.current.showModal==='function')dialog.current.showModal();else dialog.current.setAttribute('open','');}},[open]);
 function close(){const active=dialog.current?.contains(document.activeElement);if(typeof dialog.current?.close==='function')dialog.current.close();else dialog.current?.removeAttribute('open');setOpen(false);if(active&&origin.current===location.href)trigger.current?.focus({preventScroll:true});}
 async function load(){
  const request=++sequence.current,href=location.href;setLoading(true);if(!command)setError('');
  try{const value=confirmedReport(await socialApi<OwnReport>(path,undefined,AbortSignal.timeout(20_000)),owner,postId);if(mounted.current&&request===sequence.current&&href===location.href){setKnown(previous=>previous?.reported?previous:value);if(value.reported){confirmed.current=true;setCommand(undefined);setRejected(false);setError('');}}}
  catch(e){if(mounted.current&&request===sequence.current)setError(errorMessage(e));}
  finally{if(mounted.current&&request===sequence.current)setLoading(false);}
 }
 async function submit(previous?:ReportCommand){
  if(working.current||disabled||loading||!known||known.reported||!previous&&(!!command||reason==='other'&&!details.trim()))return;
  const sent=previous||{expected_user_id:owner,request_id:reportRequestId(),reason,details:details.trim()},request=generation.current,href=location.href;
  working.current=true;setBusy(true);setCommand(sent);setError('');setRejected(false);
  try{const value=confirmedReport(await socialApi<OwnReport>(path,sent,AbortSignal.timeout(20_000)),owner,postId,sent.request_id);if(mounted.current&&request===generation.current&&href===location.href){confirmed.current=true;setKnown(value);setCommand(undefined);}}
  catch(e){if(mounted.current&&request===generation.current&&!confirmed.current){setError(errorMessage(e));setRejected(e instanceof SocialError&&[400,403,404,409].includes(e.status)&&e.code!=='account_changed');if(e instanceof SocialError&&(e.status===401||e.code==='account_changed'))void refresh().catch(()=>{});}}
  finally{working.current=false;if(mounted.current){setBusy(false);}}
 }
 return <><button ref={trigger} type="button" className="post-report-trigger" disabled={disabled} aria-haspopup="dialog" aria-expanded={open} onClick={()=>{origin.current=location.href;setOpen(true);void load();}}><Flag size={14} aria-hidden="true"/>{known?.reported?t('已举报','Reported'):t('举报','Report')}</button>
 {open&&createPortal(<dialog ref={dialog} className="post-report-dialog" aria-labelledby={id} onCancel={event=>{event.preventDefault();close();}} onKeyDown={event=>{if(event.key==='Escape'&&(event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey))event.preventDefault();}}>
  <header><h2 id={id}>{t('举报帖子','Report post')}</h2><button type="button" className="icon-button" aria-label={t('关闭举报','Close report')} onClick={close}><X size={18}/></button></header>
  <p>{t('举报仅对审核人员可见。同一账号对同一帖子只计一次，由管理员人工审核。','Reports are private to reviewers. Each account counts once per post. An administrator reviews the content.')}</p>
  {known?.reported?<p role="status" className="report-confirmed">{t('举报已记录，感谢帮助维护社区。','Your report is recorded. Thank you for helping the community.')}</p>:<form onSubmit={event=>{event.preventDefault();void submit();}}>
   <label>{t('举报原因','Reason')}<select value={reason} onChange={event=>setReason(event.target.value as ReportReason)}>{reportReasons.map(([value,zh,en])=><option key={value} value={value}>{t(zh,en)}</option>)}</select></label>
   <label>{t('补充说明','Details')}<textarea rows={4} maxLength={1000} value={details} required={reason==='other'} onChange={event=>setDetails(event.target.value)} placeholder={t('说明具体问题，便于审核（其他原因必填）','Describe the issue for reviewers (required for Other)')}/></label>
   <button className="button" type="submit" disabled={disabled||loading||busy||!known||!!command||reason==='other'&&!details.trim()}>{busy?t('正在确认…','Confirming…'):t('提交举报','Submit report')}</button>
  </form>}
  {loading&&<p role="status">{t('正在读取举报状态…','Checking report status…')}</p>}
  {error&&<div role="status"><p>{error}</p>{command?<><p>{t('重试会确认原来的举报，不会新增一票。','Retry confirms the original report; it does not add another vote.')}</p><button type="button" className="text-button" disabled={busy||loading} onClick={()=>{if(rejected){setCommand(undefined);setError('');setRejected(false);}else void submit(command);}}>{rejected?t('返回修改','Return to editing'):t('重试确认','Retry confirmation')}</button></>:<button type="button" className="text-button" disabled={loading||busy} onClick={()=>void load()}>{t('重试加载','Retry loading')}</button>}</div>}
 </dialog>,document.body)}</>;
}
