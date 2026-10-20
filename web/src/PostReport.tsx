import AppLink from './AppLink';
import {useEffect,useId,useLayoutEffect,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import {CheckCircle2,Flag,X} from 'lucide-react';
import {useI18n} from './i18n';
import {errorMessage,SocialError,socialApi,useAccount} from './social';
import {signInHref} from './accountNavigation';
import {confirmedReport,reportReasons,reportRequestId,type OwnReport,type ReportCommand,type ReportReason} from './communityReports';
import {reportHistoryHref} from './reportHistoryNavigation';
import {controlIsActive} from './activeControls';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import './community-reports.css';

type Props={postId:string;returnTo:string;postTitle?:string;own?:boolean;disabled?:boolean;canAct?:()=>boolean};
const reasonHints:Record<ReportReason,readonly [string,string]>={
 spam:['重复推广、垃圾链接或与讨论无关的广告。','Repeated promotions, spam links or unrelated advertising.'],
 harassment:['针对个人或群体的骚扰、侮辱或威胁。','Harassment, insults or threats targeting a person or group.'],
 privacy:['未经许可公开他人的联系方式或其他私人信息。','Someone’s contact details or other private information shared without permission.'],
 misinformation:['描述具体的不实表述，并尽可能提供核对线索。','Describe the misleading claim and, if possible, provide a way to check it.'],
 copyright:['说明涉及的原创内容或权利来源，便于审核。','Identify the original work or rights involved so a reviewer can check.'],
 other:['请补充具体问题，帮助审核人员理解举报原因。','Add details so a reviewer can understand the issue.'],
};
export default function PostReport({postId,returnTo,postTitle,own=false,disabled=false,canAct}:Props){
 const {user,loading,capabilities}=useAccount(),{t}=useI18n();
 if(loading||capabilities?.community_reports!==true||own)return null;
 if(!user)return <AppLink className="post-report-trigger" href={signInHref(returnTo)} aria-disabled={disabled||undefined} onClick={event=>{
  if(disabled||canAct?.()===false||!controlIsActive(event.currentTarget)||event.currentTarget.closest('details:not([open]),dialog:not([open])'))event.preventDefault();
 }}><Flag size={14} aria-hidden="true"/>{t('举报','Report')}</AppLink>;
 return <ReportForm key={`${user.id}:${postId}`} owner={user.id} postId={postId} postTitle={postTitle} disabled={disabled} canAct={canAct}/>;
}
function ReportForm({owner,postId,postTitle,disabled,canAct}:{owner:string;postId:string;postTitle?:string;disabled:boolean;canAct?:()=>boolean}){
 const {t}=useI18n(),{refresh,capabilities}=useAccount(),id=useId(),trigger=useRef<HTMLButtonElement>(null),dialog=useRef<HTMLDialogElement>(null),mounted=useRef(true),sequence=useRef(0),generation=useRef(0),working=useRef(false);
 const [open,setOpen]=useState(false),[known,setKnown]=useState<OwnReport>(),[loading,setLoading]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState(''),[rejected,setRejected]=useState(false);
 const [reason,setReason]=useState<ReportReason>('spam'),[details,setDetails]=useState(''),[command,setCommand]=useState<ReportCommand>();
 const origin=useRef(''),confirmed=useRef(false),path=`community/posts/${encodeURIComponent(postId)}/report`;
 const retry=useRef<HTMLButtonElement>(null),success=useRef<HTMLDivElement>(null),intent=useWorkspaceIntent(JSON.stringify(['report-feedback',owner,postId]));
 const feedback=useRef<{href:string;claim:()=>boolean;mode?:'edit'|'read'}|undefined>(undefined);
 const reasonField=useRef<HTMLSelectElement>(null),detailsField=useRef<HTMLTextAreaElement>(null);
 const canStart=()=>mounted.current&&!disabled&&controlIsActive(trigger.current)&&!trigger.current?.closest('details:not([open]),dialog:not([open])')&&canAct?.()!==false;
 const current=()=>canStart()&&open&&origin.current===location.href&&controlIsActive(dialog.current)&&dialog.current?.open===true;
 useEffect(()=>{mounted.current=true;const leave=()=>{sequence.current++;generation.current++;confirmed.current=false;intent.cancel();feedback.current=undefined;setOpen(false);setKnown(undefined);setCommand(undefined);setError('');setDetails('');setReason('spam');setLoading(false);};window.addEventListener('hashchange',leave);window.addEventListener('popstate',leave);return()=>{mounted.current=false;sequence.current++;generation.current++;window.removeEventListener('hashchange',leave);window.removeEventListener('popstate',leave);};},[]);
 useLayoutEffect(()=>{if(open&&dialog.current&&!dialog.current.open){if(typeof dialog.current.showModal==='function')dialog.current.showModal();else dialog.current.setAttribute('open','');}},[open]);
 useLayoutEffect(()=>{
  const pending=feedback.current;if(busy||loading||!pending)return;feedback.current=undefined;
  const allowed=pending.claim(),target=known?.reported?success.current:error?retry.current:pending.mode==='edit'||pending.mode==='read'&&!!details?detailsField.current:pending.mode==='read'?reasonField.current:null;
  if(!allowed||pending.href!==location.href||!current()||!controlIsActive(target))return;
  target?.focus({preventScroll:true});
 },[busy,loading,error,known,command]);
 function close(){
  const active=dialog.current?.contains(document.activeElement);intent.cancel();feedback.current=undefined;
  if(typeof dialog.current?.close==='function')dialog.current.close();else dialog.current?.removeAttribute('open');setOpen(false);
  if(active&&origin.current===location.href&&canStart())trigger.current?.focus({preventScroll:true});
 }
 async function load(){
  if(!canStart())return;
  const request=++sequence.current,href=location.href;setLoading(true);if(!command)setError('');
  try{const value=confirmedReport(await socialApi<OwnReport>(path,undefined,AbortSignal.timeout(20_000)),owner,postId);if(mounted.current&&request===sequence.current&&href===location.href){setKnown(previous=>previous?.reported?previous:value);if(value.reported){confirmed.current=true;setCommand(undefined);setRejected(false);setError('');}}}
  catch(e){if(mounted.current&&request===sequence.current&&href===location.href)setError(errorMessage(e));}
  finally{if(mounted.current&&request===sequence.current)setLoading(false);}
 }
 async function submit(previous?:ReportCommand){
  if(!current()||working.current||loading||!known||known.reported||!previous&&(!!command||reason==='other'&&!details.trim()))return;
  const sent=previous||{expected_user_id:owner,request_id:reportRequestId(),reason,details:details.trim()},request=generation.current,href=location.href;
  feedback.current={href,claim:intent.create()};working.current=true;setBusy(true);setCommand(sent);setError('');setRejected(false);
  try{const value=confirmedReport(await socialApi<OwnReport>(path,sent,AbortSignal.timeout(20_000)),owner,postId,sent.request_id);if(mounted.current&&request===generation.current&&href===location.href){confirmed.current=true;setKnown(value);setCommand(undefined);}}
  catch(e){if(mounted.current&&request===generation.current&&!confirmed.current&&href===location.href){setError(errorMessage(e));setRejected(e instanceof SocialError&&[400,403,404,409].includes(e.status)&&e.code!=='account_changed');if(e instanceof SocialError&&(e.status===401||e.code==='account_changed'))void refresh().catch(()=>{});}}
  finally{working.current=false;if(mounted.current)setBusy(false);}
 }
 const changed=!!command&&(command.reason!==reason||command.details!==details.trim());
 return <><button ref={trigger} type="button" className="post-report-trigger" disabled={disabled} aria-haspopup="dialog" aria-expanded={open} onClick={()=>{
  if(open||!canStart())return;origin.current=location.href;setOpen(true);void load();
 }}><Flag size={14} aria-hidden="true"/>{known?.reported?t('已举报','Reported'):t('举报','Report')}</button>
 {open&&createPortal(<dialog ref={dialog} className="post-report-dialog" aria-labelledby={id} aria-describedby={id+'-privacy'} onCancel={event=>{event.preventDefault();close();}} onKeyDown={event=>{
  if(event.key==='Escape'&&(event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey)){event.preventDefault();event.stopPropagation();}
 }}>
  <header><h2 id={id}>{t('举报帖子','Report post')}</h2><button type="button" className="icon-button" aria-label={t('关闭举报','Close report')} onClick={close}><X size={18} aria-hidden="true"/></button></header>
  {postTitle&&<div className="report-post-context"><span>{t('举报对象','Reporting post')}</span><strong>{postTitle}</strong></div>}
  <p id={id+'-privacy'}>{t('举报仅对审核人员可见。同一账号对同一帖子只计一次，由管理员人工审核。','Reports are private to reviewers. Each account counts once per post. An administrator reviews the content.')}</p>
  {known?.reported?<div ref={success} tabIndex={-1} className="report-success"><CheckCircle2 size={24} aria-hidden="true"/><p role="status" className="report-confirmed">{t('举报已记录，感谢帮助维护社区。','Your report is recorded. Thank you for helping the community.')}</p><p>{t('管理员会人工核对内容。已记录不代表帖子已违规或已被移除。','An administrator will review the content. Recording a report does not mean the post violates a rule or has been removed.')}</p>{capabilities?.community_report_history===true&&<AppLink className="button subtle" href={reportHistoryHref()}>{t('查看我的举报','View my reports')}</AppLink>}</div>:<form aria-busy={loading||busy} onSubmit={event=>{const handled=event.defaultPrevented;event.preventDefault();if(!handled)void submit();}}>
   <label htmlFor={id+'-reason'}>{t('举报原因','Reason')}<select ref={reasonField} id={id+'-reason'} value={reason} aria-describedby={id+'-reason-hint'} onChange={event=>{const choice=reportReasons.find(row=>row[0]===event.target.value);if(current()&&choice)setReason(choice[0]);}}>{reportReasons.map(([value,zh,en])=><option key={value} value={value}>{t(zh,en)}</option>)}</select></label>
   <p className="report-field-hint" id={id+'-reason-hint'}>{t(...reasonHints[reason])}</p>
   <label htmlFor={id+'-details'}><span className="report-details-label">{t('补充说明','Details')}<small>{reason==='other'?t('必填','Required'):t('选填','Optional')}</small></span><textarea ref={detailsField} id={id+'-details'} rows={4} maxLength={1000} value={details} required={reason==='other'} aria-describedby={`${id}-details-hint ${id}-details-count`} onChange={event=>{if(current())setDetails(event.target.value);}} placeholder={t('描述具体问题或提供核对线索','Describe the issue or provide a way to check it')}/></label>
   <div className="report-details-help"><span id={id+'-details-hint'}>{reason==='other'&&!details.trim()?t('选择“其他问题”时，请填写具体说明后提交。','For Other, add specific details before submitting.'):t('请勿填写与举报无关的私人信息。','Avoid including private information unrelated to this report.')}</span><span id={id+'-details-count'}>{details.length} / 1000</span></div>
   <button className="button" type="submit" disabled={disabled||loading||busy||!known||!!command||reason==='other'&&!details.trim()}>{busy?t('正在确认…','Confirming…'):t('提交举报','Submit report')}</button>
  </form>}
  {command&&!known?.reported&&<section className="report-pending" aria-labelledby={id+'-pending'}>
   <h3 id={id+'-pending'}>{busy?t('正在确认这份举报','Confirming this report'):rejected?t('这份举报未被接受','This report was not accepted'):t('这份举报尚未确认','This report is not confirmed')}</h3>
   <p>{t('以下是本次实际提交的内容。重试只确认这一份举报，不会新增一票。','This is the content submitted in this attempt. Retry confirms this report without adding another vote.')}</p>
   <details><summary>{t('查看本次提交内容','Review submitted report')}</summary><dl><dt>{t('举报原因','Reason')}</dt><dd>{t(...reportReasons.find(row=>row[0]===command.reason)!.slice(1) as [string,string])}</dd><dt>{t('补充说明','Details')}</dt><dd className="report-submitted-details" tabIndex={command.details?0:undefined} aria-label={t('本次提交的完整说明','Complete submitted details')}>{command.details||t('未填写补充说明','No additional details')}</dd></dl></details>
   {changed&&<p className="report-newer-draft" role="status">{t('表单中的新修改尚未提交；重试仍使用上方这份内容。','Your newer form edits have not been submitted. Retry still uses the content shown above.')}</p>}
  </section>}
  {loading&&<p role="status">{t('正在读取举报状态…','Checking report status…')}</p>}
  {error&&<div className="report-recovery" role="alert"><p>{error}</p>{command?<><p>{rejected?t('当前输入已保留，可以返回表单修改。','Your current input is kept. Return to the form to make changes.'):t('网络错误不代表举报失败。关闭后，可在当前页面重新打开核对状态。','A network error does not prove the report failed. Close and reopen it on this page to check its status.')}</p><button ref={retry} type="button" className="text-button" disabled={busy||loading} onClick={()=>{
   if(!current())return;if(rejected){feedback.current={href:location.href,claim:intent.create(),mode:'edit'};setCommand(undefined);setError('');setRejected(false);}else void submit(command);
  }}>{rejected?t('返回修改','Return to editing'):t('重试确认','Retry confirmation')}</button></>:<button ref={retry} type="button" className="text-button" disabled={loading||busy} onClick={()=>{if(current()){feedback.current={href:location.href,claim:intent.create(),mode:'read'};void load();}}}>{t('重试加载','Retry loading')}</button>}</div>}
  <footer className="report-dialog-footer"><button type="button" className="button subtle" onClick={close}>{t('关闭','Close')}</button></footer>
 </dialog>,document.body)}</>;
}
