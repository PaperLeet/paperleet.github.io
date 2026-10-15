import {useEffect,useRef,useState} from 'react';
import {Check,ChevronRight,Plus,X} from 'lucide-react';
import {useI18n} from './i18n';
import {socialApi,errorMessage,useAccount} from './social';
import CommunityMarkdown,{CommunityFormatHelp} from './CommunityMarkdown';
import InterviewFields,{type InterviewModule} from './InterviewFields';
import {readInterviewDraft,writeInterviewDraft,clearInterviewDraft,discardInterviewDraft,interviewPublication,type InterviewDraft} from './interviewDrafts';
import {tabDraftStorage} from './communityDrafts';
import {clearPublication,preparePublication,readPublication,samePublication,type PublicationReceipt} from './publicationRequests';
import PublicationRecovery from './PublicationRecovery';
import {PublishingRules} from './PublishingRules';
import type {InterviewPost} from './interviewModels';
import {communityPostHref} from './communityNavigation';
import {emptyInterviewFilters,interviewHref} from './interviewNavigation';

export default function InterviewComposer({owner,company,problems,onClose,onPublished}:{owner:string;company:string;problems:InterviewModule[];onClose:()=>void;onPublished:(post:InterviewPost)=>void}){
 const {t}=useI18n(),{user}=useAccount();
 const [draft,setDraft]=useState<InterviewDraft>(()=>readInterviewDraft(tabDraftStorage(),owner)||{title:'',body:'',interview:{company:company||'other',role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:[]}});
 const latest=useRef(draft),active=useRef(true),inflight=useRef(false),currentOwner=useRef(user?.id);currentOwner.current=user?.id;
 const [pending,setPending]=useState(()=>readPublication(tabDraftStorage(),owner,'interview','general'));
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[preview,setPreview]=useState(false),[durable,setDurable]=useState(true),[discard,setDiscard]=useState(false);
 const [confirmedPost,setConfirmedPost]=useState<InterviewPost>();
 useEffect(()=>{active.current=true;return()=>{active.current=false;};},[]);
 function edit(change:Partial<InterviewDraft>){const next={...latest.current,...change};latest.current=next;setDraft(next);setDurable(writeInterviewDraft(tabDraftStorage(),owner,next));}
 async function publish(previous?:PublicationReceipt){
   if(inflight.current||!user||user.id!==owner)return;
   const submitted=previous?{title:previous.payload.title as string,body:previous.payload.body as string,interview:JSON.parse(previous.payload.interview as string)} as InterviewDraft:latest.current;
   if(!submitted.title.trim()||!submitted.body.trim())return;
   inflight.current=true;setBusy(true);setError('');
   try{
     const request=preparePublication(tabDraftStorage(),owner,'interview','general',interviewPublication(submitted),previous);setPending(request);
     const {interview,...content}=request.receipt.payload;
     const confirmed=await socialApi<InterviewPost>('community/posts',{...content,interview:JSON.parse(interview as string),request_id:request.receipt.requestId,expected_user_id:owner});
     if(!confirmed?.interview||confirmed.author?.id!==owner||!/^[a-f0-9]{32}$/.test(confirmed.id))throw Error(t('发布结果暂时无法确认，请保留草稿并重试。','Publication could not be confirmed. Keep your draft and retry.'));
     if(!active.current||currentOwner.current!==owner)return;
     const removed=clearInterviewDraft(tabDraftStorage(),owner,submitted);
     const same=samePublication(request.receipt,interviewPublication(latest.current));
     if(!removed&&same){setConfirmedPost(confirmed);setError(t('面经已发布，但暂时无法清理草稿。重试会继续确认同一篇面经。','Your report was published, but its draft could not be cleared. Retrying confirms the same report.'));return;}
     clearPublication(tabDraftStorage(),request.receipt);setPending(undefined);
     if(removed||same)onPublished(confirmed);else setConfirmedPost(confirmed);
   }catch(error){if(active.current&&currentOwner.current===owner)setError(errorMessage(error));}
   finally{inflight.current=false;if(active.current){setBusy(false);}}
 }
 return <section className="interview-composer social-panel" aria-labelledby="interview-compose-heading">
  <div className="interview-compose-heading"><div><span className="social-eyebrow">{t('把经验留给下一位','HELP THE NEXT PERSON')}</span><h2 id="interview-compose-heading">{t('分享面经','Share an interview experience')}</h2></div><button className="icon-button" type="button" disabled={busy} aria-label={t('关闭，保留草稿','Close and keep draft')} onClick={onClose}><X size={18}/></button></div>
  <p className="interview-composer-caption">{t('公司、岗位与知识点会公开展示；结果和地点可以不透露。以你的公开昵称发布。','Company, role and linked concepts are public. Outcome and location are optional. Your public display name appears as the author.')}</p>
  {!durable&&<p role="alert" className="interview-notice">{t('草稿暂时无法保存到此标签页，请保持页面打开或复制内容。','This tab cannot save the draft. Keep this page open or copy your text.')}</p>}
  {error&&<p role="alert" className="interview-error">{error}</p>}
  {confirmedPost&&<p role="status" className="interview-notice">{t('面经已发布；当前草稿仍保留。','Your report was published; the current draft is preserved.')} <a href={communityPostHref(confirmedPost.id,interviewHref({...emptyInterviewFilters(),company:confirmedPost.interview.company}))}>{t('查看已发布面经','View published report')}</a></p>}
  {!confirmedPost&&<PublicationRecovery pending={pending} payload={interviewPublication(draft)} busy={busy} onRetry={()=>void publish(pending?.receipt)}/>}
  <form className="social-form" onSubmit={event=>{event.preventDefault();void publish();}}>
   <InterviewFields value={draft.interview} onChange={interview=>edit({interview})} problems={problems} disabled={busy}/>
   <label>{t('面经标题','Title')}<input autoFocus required maxLength={180} disabled={busy} value={draft.title} placeholder={t('例如：机器学习工程师技术面试复盘','e.g. ML engineering technical interview recap')} onChange={event=>edit({title:event.target.value})}/></label>
   <div className="editor-toggle" role="group" aria-label={t('面经编辑模式','Report editing mode')}><button type="button" aria-pressed={!preview} onClick={()=>setPreview(false)}>{t('编写','Write')}</button><button type="button" aria-pressed={preview} onClick={()=>setPreview(true)}>{t('预览','Preview')}</button>{!draft.body&&<button type="button" disabled={busy} className="text-button" onClick={()=>edit({body:t('## 面试流程\n\n## 考察重点与我的思路\n\n## 复盘与准备建议\n','## Interview process\n\n## Topics and my approach\n\n## Reflection and preparation tips\n')})}><Plus size={12}/>{t('使用写作提纲','Use outline')}</button>}</div>
   <CommunityFormatHelp/>
   {preview?<div className="compose-preview"><CommunityMarkdown body={draft.body||t('写下内容后在这里预览。','Write your experience to preview it here.')}/></div>:<label>{t('流程、思路与复盘','Process, approach and reflection')}<textarea required rows={12} maxLength={20000} disabled={busy} value={draft.body} placeholder={t('记录轮次、考察方向、你的解题思路和准备建议。只分享可以公开的信息。','Describe the rounds, concepts, your approach and preparation tips. Share information you can make public.')} onChange={event=>edit({body:event.target.value})}/></label>}
   <PublishingRules action="post"/>
   <div className="form-footer"><button type="button" className="text-button" disabled={busy} onClick={()=>setDiscard(true)}>{t('丢弃草稿','Discard draft')}</button><small><Check size={12}/>{durable?t('草稿仅保存在本标签页','Draft stays in this tab'):t('草稿暂存在当前页面','Draft is held in this page')}</small><button className="button" disabled={busy||!draft.title.trim()||!draft.body.trim()}>{busy?<span className="spinner"/>:<ChevronRight size={15}/>} {pending?samePublication(pending.receipt,interviewPublication(draft))?t('重试发布','Retry publication'):t('作为新面经发布','Publish as a new report'):t('发布面经','Publish report')}</button></div>
  </form>
  {discard&&<div className="draft-discard-confirm" role="alert"><p>{t('丢弃当前草稿？已发送但待确认的发布记录会保留，避免误发重复内容。','Discard this draft? Unconfirmed publication receipts are retained so they can be retried safely.')}</p><button type="button" disabled={busy} onClick={()=>{if(discardInterviewDraft(tabDraftStorage(),owner))onClose();else setError(t('暂时无法清除草稿，请重试。','The draft could not be cleared. Try again.'));}}>{t('确认丢弃','Discard draft')}</button><button type="button" onClick={()=>setDiscard(false)}>{t('继续编辑','Keep editing')}</button></div>}
 </section>;
}
