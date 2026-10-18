import CommunityTagPicker,{TaxonomyStatus} from './CommunityTagPicker';
import {useCommunityTaxonomy} from './useCommunityTaxonomy';
import {useEffect,useId,useRef,useState} from 'react';
import {BookOpen,Check,ChevronDown,ChevronRight,X} from 'lucide-react';
import {useI18n} from './i18n';
import {socialApi,errorMessage,useAccount} from './social';
import MarkdownEditor from './MarkdownEditor';
import InterviewFields,{type InterviewModule} from './InterviewFields';
import {readInterviewDraft,writeInterviewDraft,isInterviewDraftDurable,clearInterviewDraft,discardInterviewDraft,interviewPublication,type InterviewDraft} from './interviewDrafts';
import {tabDraftStorage} from './communityDrafts';
import {clearPublication,preparePublication,readPublication,samePublication,type PublicationReceipt} from './publicationRequests';
import PublicationRecovery from './PublicationRecovery';
import {PublishingRules} from './PublishingRules';
import {affiliationRequired,canPublishCommunity} from './communityPublishing';
import PublishingAffiliationNotice from './PublishingAffiliationNotice';
import type {InterviewPost} from './interviewModels';
import {communityPostHref} from './communityNavigation';
import {emptyInterviewFilters,interviewHref} from './interviewNavigation';
import {deferredFeature} from './deferredFeature';
import {defaultInterviewOutline,hasInterviewDraftContent,insertInterviewOutline,interviewOutlines,type InterviewOutlineEdit,type WritingOutlineId} from './interviewWriting';
import './interview-writing.css';
import PublishingIdentity from './PublishingIdentity';
import {ownsCommunityPost} from './communityIdentity';
import type {WritingPanelProps} from './useWritingDisclosure';

const WritingGuide=deferredFeature(()=>import('./InterviewWritingGuide'),['写作提纲','writing prompts'],{
 failure:(props,retry)=><WritingGuideFailure onRetry={()=>{props.onRestoreFocus();retry();}}/>,
});
function WritingGuideFailure({onRetry}:{onRetry:()=>void}){
 const {t}=useI18n();
 return <div className="interview-writing-failure"><p role="alert">{t('写作提示暂时无法加载，正文仍可继续编辑。','Writing prompts could not load. You can keep editing your report.')}</p><button type="button" onClick={onRetry}>{t('重试写作提示','Retry writing prompts')}</button></div>;
}

export default function InterviewComposer({owner,company,problems,panelProps,onClose,onPublished}:{owner:string;company:string;problems:InterviewModule[];panelProps?:WritingPanelProps;onClose:()=>void;onPublished:(post:InterviewPost)=>void}){
 const {t,locale}=useI18n(),{user,capabilities,loading:accountLoading,refresh:refreshAccount}=useAccount();
 const taxonomy=useCommunityTaxonomy();
 const canPublish=canPublishCommunity(user,capabilities)&&!accountLoading;
 const [draft,setDraft]=useState<InterviewDraft>(()=>readInterviewDraft(tabDraftStorage(),owner)||{title:'',body:'',interview:{company:company||'other',role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:[]}});
 const latest=useRef(draft),active=useRef(true),inflight=useRef(false),currentOwner=useRef(user?.id);currentOwner.current=user?.id;
 const [pending,setPending]=useState(()=>readPublication(tabDraftStorage(),owner,'interview','general'));
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[preview,setPreview]=useState(false),[durable,setDurable]=useState(()=>!readInterviewDraft(tabDraftStorage(),owner)||isInterviewDraftDurable(tabDraftStorage(),owner,draft)),[discard,setDiscard]=useState(false);
 const [confirmedPost,setConfirmedPost]=useState<InterviewPost>();
 const editor=useRef<HTMLTextAreaElement>(null),writingSummary=useRef<HTMLElement>(null),contentHint=useId();
 const [writingOpen,setWritingOpen]=useState(false),[writingError,setWritingError]=useState(''),[focusTick,setFocusTick]=useState(0);
 const [outlineEdit,setOutlineEdit]=useState<(InterviewOutlineEdit&{selection:{start:number;end:number;scroll:number}})>();
 const focusRequest=useRef<{opener:HTMLElement;start:number;end:number;scroll?:number}>(undefined);
 useEffect(()=>{active.current=true;return()=>{active.current=false;};},[]);
 useEffect(()=>{
  const request=focusRequest.current;if(!request)return;focusRequest.current=undefined;
  if(document.activeElement!==request.opener&&!(document.activeElement===document.body&&!request.opener.isConnected))return;
  const field=editor.current;if(!field)return;
  field.focus();field.setSelectionRange(request.start,request.end);
  if(request.scroll!==undefined)field.scrollTop=request.scroll;
 },[draft.body,preview,focusTick]);
 function edit(change:Partial<InterviewDraft>){const next={...latest.current,...change};latest.current=next;setDraft(next);setDurable(writeInterviewDraft(tabDraftStorage(),owner,next));if(change.body!==undefined)setWritingError('');}
 function insertOutline(id:WritingOutlineId,opener:HTMLElement){
  if(inflight.current||!active.current||currentOwner.current!==owner)return;
  const body=latest.current.body,field=editor.current;
  const selection={start:field?.selectionStart??body.length,end:field?.selectionEnd??body.length,scroll:field?.scrollTop||0};
  const operation=insertInterviewOutline(body,id,locale==='en'?'en':'zh',selection.end);
  if(!operation){setWritingError(t('正文已接近长度上限，请先精简内容再插入提纲。','Your report is near its length limit. Shorten it before inserting an outline.'));return;}
  edit({body:operation.after});setOutlineEdit({...operation,selection});setPreview(false);
  focusRequest.current={opener,start:operation.caret,end:operation.caret};setFocusTick(value=>value+1);
 }
 function undoOutline(opener:HTMLElement){
  if(inflight.current||!active.current||currentOwner.current!==owner||!outlineEdit||latest.current.body!==outlineEdit.after)return;
  edit({body:outlineEdit.before});setPreview(false);setOutlineEdit(undefined);
  focusRequest.current={opener,...outlineEdit.selection};setFocusTick(value=>value+1);
 }
 const contentReady=hasInterviewDraftContent(draft.body);
 const pendingMatches=!!pending&&samePublication(pending.receipt,interviewPublication(draft));
 async function publish(previous?:PublicationReceipt){
   if(inflight.current||!user||user.id!==owner||!canPublish)return;
   const receipt=previous||(pending&&samePublication(pending.receipt,interviewPublication(latest.current))?pending.receipt:undefined);
   const submitted=receipt?{...(Array.isArray(receipt.payload.tag_ids)?{tagIds:receipt.payload.tag_ids}:{}),anonymous:receipt.payload.anonymous===true,title:receipt.payload.title as string,body:receipt.payload.body as string,interview:JSON.parse(receipt.payload.interview as string)} as InterviewDraft:latest.current;
   if(!receipt&&submitted.tagIds?.length&&!taxonomy.supported){setError(t('标签服务暂不可用，草稿已保留。','Tags are unavailable. Your draft is kept.'));return;}
   if(!receipt&&taxonomy.supported&&(!taxonomy.ready||!taxonomy.data.boards.some(row=>row.id==='career'&&row.enabled))){setError(t('求职版位暂未开放，草稿已保留。','The career board is unavailable. Your draft is kept.'));return;}
   if(!submitted.title.trim()||!submitted.body.trim()||!receipt&&!hasInterviewDraftContent(submitted.body))return;
   if(submitted.anonymous&&capabilities?.community_anonymous_posts!==true){setError(t('匿名发布暂不可用，草稿已保留。','Anonymous posting is unavailable. Your draft is kept.'));return;}
   inflight.current=true;setBusy(true);setError('');
   try{
     const request=preparePublication(tabDraftStorage(),owner,'interview','general',interviewPublication(submitted),receipt);setPending(request);
     const {interview,...content}=request.receipt.payload;
     const confirmed=await socialApi<InterviewPost>(request.receipt.payload.anonymous===true?'community/posts/anonymous':'community/posts',{...content,interview:JSON.parse(interview as string),request_id:request.receipt.requestId,expected_user_id:owner});
     if(!confirmed?.interview||!ownsCommunityPost(confirmed,owner,request.receipt.payload.anonymous===true)||!/^[a-f0-9]{32}$/.test(confirmed.id))throw Error(t('发布结果暂时无法确认，请保留草稿并重试。','Publication could not be confirmed. Keep your draft and retry.'));
     if(!active.current||currentOwner.current!==owner)return;
     const removed=clearInterviewDraft(tabDraftStorage(),owner,submitted);
     const same=samePublication(request.receipt,interviewPublication(latest.current));
     if(!removed&&same){setConfirmedPost(confirmed);setError(t('面经已发布，但暂时无法清理草稿。重试会继续确认同一篇面经。','Your report was published, but its draft could not be cleared. Retrying confirms the same report.'));return;}
     clearPublication(tabDraftStorage(),request.receipt);setPending(undefined);
     if(removed||same)onPublished(confirmed);else setConfirmedPost(confirmed);
   }catch(error){if(active.current&&currentOwner.current===owner)setError(errorMessage(error));if(affiliationRequired(error))void refreshAccount().catch(()=>{});}
   finally{inflight.current=false;if(active.current){setBusy(false);}}
 }
 return <section {...panelProps} className="interview-composer social-panel" aria-labelledby="interview-compose-heading">
  <div className="interview-compose-heading"><div><h2 id="interview-compose-heading">{t('分享面经','Share an interview experience')}</h2></div><button className="icon-button" type="button" disabled={busy} aria-label={t('关闭，保留草稿','Close and keep draft')} onClick={onClose}><X size={18}/></button></div>
  <p className="interview-composer-caption">{t('公司、岗位与知识点会公开展示；结果和地点可以不透露。可选择昵称或匿名发布，所属机构仍展示。','Company, role and linked concepts are public. Outcome and location are optional. Choose named or anonymous posting; affiliation remains visible.')}</p>
  {!durable&&<p role="alert" className="interview-notice">{t('草稿暂时无法保存到此标签页，请保持页面打开或复制内容。','This tab cannot save the draft. Keep this page open or copy your text.')}</p>}
  {error&&<p role="alert" className="interview-error">{error}</p>}
  {confirmedPost&&<p role="status" className="interview-notice">{t('面经已发布；当前草稿仍保留。','Your report was published; the current draft is preserved.')} <a href={communityPostHref(confirmedPost.id,interviewHref({...emptyInterviewFilters(),company:confirmedPost.interview.company}))}>{t('查看已发布面经','View published report')}</a></p>}
  <PublishingAffiliationNotice/>
  {!confirmedPost&&<PublicationRecovery pending={pending} payload={interviewPublication(draft)} busy={busy} disabled={!canPublish} onRetry={()=>void publish(pending?.receipt)}/>}
  <form className="social-form" onSubmit={event=>{event.preventDefault();void publish();}}>
   <TaxonomyStatus/><CommunityTagPicker value={draft.tagIds||[]} onChange={tagIds=>edit({tagIds})} disabled={busy}/><PublishingIdentity anonymous={!!draft.anonymous} onChange={anonymous=>edit({anonymous})} disabled={busy}/>
   <InterviewFields value={draft.interview} onChange={interview=>edit({interview})} problems={problems} disabled={busy}/>
   <label>{t('面经标题','Title')}<input data-writing-title="" required maxLength={180} disabled={busy} value={draft.title} placeholder={t('例如：机器学习工程师技术面试复盘','e.g. ML engineering technical interview recap')} onChange={event=>edit({title:event.target.value})}/></label>
   <details className="interview-writing-guide" onToggle={event=>setWritingOpen(event.currentTarget.open)}><summary ref={writingSummary}><span><BookOpen size={15}/>{t('用提纲整理面试经历','Organize your experience with prompts')}</span><ChevronDown size={14}/></summary>{writingOpen&&<WritingGuide stage={draft.interview.stage} defaultId={defaultInterviewOutline(draft.interview.stage)} outlines={interviewOutlines} hasBody={!!draft.body.trim()} hasContent={contentReady} busy={busy} canUndo={!!outlineEdit&&draft.body===outlineEdit.after&&outlineEdit.before!==outlineEdit.after} editedAfterInsert={!!outlineEdit&&draft.body!==outlineEdit.after} onInsert={insertOutline} onUndo={undoOutline} onRestoreFocus={()=>writingSummary.current?.focus({preventScroll:true})}/>}</details>
   {writingError&&<p className="interview-writing-hint" role="status">{writingError}</p>}
   <MarkdownEditor sectionNavigation scopeKey={JSON.stringify(['interview',owner])} submitShortcut value={draft.body} onChange={body=>edit({body})} preview={preview} onPreviewChange={setPreview}
     textareaRef={editor} label={t('流程、思路与复盘','Process, approach and reflection')} modeLabel={t('面经编辑模式','Report editing mode')}
     rows={12} maxLength={20000} busy={busy} describedBy={draft.body.trim()&&!contentReady&&!pendingMatches?contentHint:undefined}
     placeholder={t('记录轮次、考察方向、你的解题思路和准备建议。只分享可以公开的信息。','Describe the rounds, concepts, your approach and preparation tips. Share information you can make public.')}/>
   {draft.body.trim()&&!contentReady&&!pendingMatches&&<p id={contentHint} className="interview-writing-hint" role="status">{t('提纲已经准备好。补充你实际经历的流程、思路或复盘后即可发布。','Your outline is ready. Add your actual process, approach or reflection before publishing.')}</p>}
   <PublishingRules action="post"/>
   <div className="form-footer"><button type="button" className="text-button" disabled={busy} onClick={()=>setDiscard(true)}>{t('丢弃草稿','Discard draft')}</button><small><Check size={12}/>{durable?t('草稿仅保存在本标签页','Draft stays in this tab'):t('草稿暂存在当前页面','Draft is held in this page')}</small><button className="button" disabled={busy||!canPublish||!draft.title.trim()||!contentReady&&!pendingMatches||!!draft.anonymous&&capabilities?.community_anonymous_posts!==true} aria-describedby={draft.body.trim()&&!contentReady&&!pendingMatches?contentHint:undefined}>{busy?<span className="spinner"/>:<ChevronRight size={15}/>} {pending?pendingMatches?t('重试发布','Retry publication'):t('作为新面经发布','Publish as a new report'):t('发布面经','Publish report')}</button></div>
  </form>
  {discard&&<div className="draft-discard-confirm" role="alert"><p>{t('丢弃当前草稿？已发送但待确认的发布记录会保留，避免误发重复内容。','Discard this draft? Unconfirmed publication receipts are retained so they can be retried safely.')}</p><button type="button" disabled={busy} onClick={()=>{if(discardInterviewDraft(tabDraftStorage(),owner))onClose();else setError(t('暂时无法清除草稿，请重试。','The draft could not be cleared. Try again.'));}}>{t('确认丢弃','Discard draft')}</button><button type="button" onClick={()=>setDiscard(false)}>{t('继续编辑','Keep editing')}</button></div>}
 </section>;
}
