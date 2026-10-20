import InterviewDraftStatus,{type InterviewDraftStep} from './InterviewDraftStatus';
import {controlIsActive} from './activeControls';
import {revealReadingTarget} from './readingViewport';
import {emptyInterviewQuestion,hasInterviewQuestionText,type InterviewQuestionScratch} from './interviewQuestionDraft';
import {insertInterviewQuestion,type QuestionRecordEdit} from './interviewQuestionInsert';
import type {InterviewQuestionBuilderProps} from './InterviewQuestionBuilder';
import {useWorkspaceIntent} from './useWorkspaceIntent';
import './interview-question-builder.css';
import AppLink from './AppLink';
import CommunityTagPicker from './CommunityTagPicker';
import {unavailableCommunityTags} from './communityTagSelection';
import {useCommunityTaxonomy} from './useCommunityTaxonomy';
import {useEffect,useId,useRef,useState} from 'react';
import {BookOpen,Check,ChevronDown,ChevronRight,X} from 'lucide-react';
import {useI18n} from './i18n';
import {socialApi,errorMessage,useAccount} from './social';
import MarkdownEditor from './MarkdownEditor';
import InterviewFields,{type InterviewModule} from './InterviewFields';
import {readInterviewDraft,writeInterviewDraft,isInterviewDraftDurable,clearInterviewDraft,discardInterviewDraft,interviewPublication,sameInterviewDraft,type InterviewDraft} from './interviewDrafts';
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

const QuestionBuilder=deferredFeature(()=>import('./InterviewQuestionBuilder'),['问题记录','question record'],{failure:(props,retry)=><QuestionBuilderFailure {...props} onRetry={()=>{props.onRestoreFocus();retry();}}/>});
function QuestionBuilderFailure(props:InterviewQuestionBuilderProps&{onRetry:()=>void}){
 const {t}=useI18n();
 return <div className="interview-question-fallback"><p role="alert">{t('记录面板暂时无法载入，输入仍在草稿中。可以重试，或先不关联知识点将记录插入正文。','The record panel could not load. Your inputs stay in the draft. Retry or insert the record without a linked concept.')}</p>{(['prompt','approach','followups']as const).map(field=><label key={field}>{({prompt:t('实际问题','Question'),approach:t('我的思路','My approach'),followups:t('追问与复盘','Follow-ups and reflection')})[field]}<textarea data-question-fallback-field={field} rows={2} maxLength={3000} disabled={props.busy} value={props.value[field]} onChange={event=>props.onChange({...props.value,[field]:event.target.value})}/></label>)}{props.error&&<p role="alert">{props.error}</p>}<div><button type="button" className="text-button" onClick={props.onRetry}>{t('重试记录面板','Retry record panel')}</button><button type="button" className="text-button" disabled={props.busy} aria-disabled={!props.value.prompt.trim()} onClick={event=>{event.currentTarget.focus({preventScroll:true});if(props.value.prompt.trim())props.onInsert(event.currentTarget,true);}}>{t('不关联知识点，插入记录','Insert record without a linked concept')}</button><button type="button" className="text-button" disabled={props.busy} aria-disabled={!props.canUndo} onClick={event=>{event.currentTarget.focus({preventScroll:true});if(props.canUndo)props.onUndo(event.currentTarget);}}>{t('撤销本次记录','Undo record insertion')}</button></div></div>;
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
 const ownComposer=useRef<HTMLElement>(null),composer=panelProps?.ref||ownComposer,titleField=useRef<HTMLInputElement>(null);
 const renderedHref=useRef(typeof location==='undefined'?'':location.href).current;
 const draftStatusId=useId(),titleId=useId(),bodyId=useId(),questionId=useId();
 const editor=useRef<HTMLTextAreaElement>(null),writingSummary=useRef<HTMLElement>(null),questionSummary=useRef<HTMLElement>(null),contentHint=useId(),questionHint=useId();
 const intent=useWorkspaceIntent(owner);
 const [questionOpen,setQuestionOpen]=useState(()=>hasInterviewQuestionText(draft.questionScratch)),[questionError,setQuestionError]=useState('');
 const [questionEdit,setQuestionEdit]=useState<(QuestionRecordEdit&{selection:{start:number;end:number;scroll:number;left:number;direction:HTMLTextAreaElement['selectionDirection']}})>();
 const [writingOpen,setWritingOpen]=useState(false),[writingError,setWritingError]=useState(''),[focusTick,setFocusTick]=useState(0);
 const [outlineEdit,setOutlineEdit]=useState<(InterviewOutlineEdit&{selection:{start:number;end:number;scroll:number}})>();
 const focusRequest=useRef<{opener:HTMLElement;start:number;end:number;scroll?:number;left?:number;direction?:HTMLTextAreaElement['selectionDirection'];claim?:()=>boolean;reveal?:boolean}>(undefined);
 useEffect(()=>{active.current=true;return()=>{active.current=false;};},[]);
 useEffect(()=>{
  const request=focusRequest.current;if(!request)return;focusRequest.current=undefined;
  if(!active.current||currentOwner.current!==owner||request.claim&&!request.claim())return;
  if(request.reveal&&!canNavigateDraft(request.opener))return;
  if(document.activeElement!==request.opener&&!(document.activeElement===document.body&&!request.opener.isConnected))return;
  const field=editor.current;if(!field||field.closest('[hidden]'))return;
  field.focus({preventScroll:!!request.reveal});field.setSelectionRange(request.start,request.end,request.direction);
  if(request.scroll!==undefined)field.scrollTop=request.scroll;
  if(request.left!==undefined)field.scrollLeft=request.left;
  if(request.reveal)revealReadingTarget(field,'nearest');
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
 function changeQuestion(value:InterviewQuestionScratch){if(inflight.current||!active.current||currentOwner.current!==owner)return;edit({questionScratch:value.module_id||[value.prompt,value.approach,value.followups].some(text=>text.length>0)?value:undefined});setQuestionError('');}
 function insertQuestion(opener:HTMLElement,unlinked=false){
  if(inflight.current||!active.current||currentOwner.current!==owner)return;
  const current=latest.current,field=editor.current,scratch=current.questionScratch||emptyInterviewQuestion(),selection={start:field?.selectionStart??current.body.length,end:field?.selectionEnd??current.body.length,scroll:field?.scrollTop||0,left:field?.scrollLeft||0,direction:field?.selectionDirection||'none'};
  const result=insertInterviewQuestion(current.body,unlinked?{...scratch,module_id:''}:scratch,current.interview.questions,problems,locale==='en'?'en':'zh',selection.end);
  if(!result.ok){setQuestionError(result.reason==='length'?t('正文已接近长度上限，输入已保留。请精简正文后再插入。','The report is near its length limit. Your inputs are kept; shorten the report before inserting.'):result.reason==='module'?t('关联信息尚未确认，请选择已选知识点或不关联。','Confirm a selected concept or choose no link.'):result.reason==='position'?t('请把光标移到代码块、引用或列表之外，并补全记录中的 Markdown。正文和输入均已保留。','Move the report cursor outside code, quotes or lists and complete the record’s Markdown. The report and inputs are kept.'):t('请先填写实际问题。','Write the actual question first.'));return;}
  const claim=intent.create();edit({body:result.edit.after,questionScratch:undefined});setQuestionEdit({...result.edit,scratch:{...scratch},selection});setQuestionError('');setPreview(false);
  focusRequest.current={opener,start:result.edit.caret,end:result.edit.caret,claim};setFocusTick(value=>value+1);
 }
 function undoQuestion(opener:HTMLElement){
  if(inflight.current||!active.current||currentOwner.current!==owner||!questionEdit||latest.current.body!==questionEdit.after||latest.current.questionScratch!==undefined)return;
  const claim=intent.create();edit({body:questionEdit.before,questionScratch:{...questionEdit.scratch}});setPreview(false);setQuestionEdit(undefined);setQuestionError('');
  focusRequest.current={opener,...questionEdit.selection,claim};setFocusTick(value=>value+1);
 }
 function clearQuestion(){if(inflight.current||!active.current||currentOwner.current!==owner)return;questionSummary.current?.focus({preventScroll:true});edit({questionScratch:undefined});setQuestionError('');}
 const uninserted=hasInterviewQuestionText(draft.questionScratch);
 const contentReady=hasInterviewDraftContent(draft.body);
 const pendingMatches=!!pending&&samePublication(pending.receipt,interviewPublication(draft));
 const draftStep:InterviewDraftStep=busy?'busy':!draft.title.trim()?'title':uninserted?'record':!contentReady&&!pendingMatches?'body':pendingMatches?'retry':'ready';
 function canNavigateDraft(opener:HTMLElement){return !inflight.current&&active.current&&currentOwner.current===owner&&controlIsActive(composer.current)&&controlIsActive(opener)&&composer.current?.contains(opener)&&document.visibilityState!=='hidden'&&location.href===renderedHref;}
 function navigateDraft(opener:HTMLButtonElement){
  if(!canNavigateDraft(opener))return;
  opener.focus({preventScroll:true});
  const target=draftStep==='title'?titleField.current:draftStep==='record'?questionSummary.current:editor.current;
  if(!target)return;
  if(draftStep==='record')setQuestionOpen(true);
  if(draftStep==='title'||draftStep==='record'){target.focus({preventScroll:true});revealReadingTarget(target,'nearest');return;}
  const field=editor.current!;
  focusRequest.current={opener,start:field.selectionStart,end:field.selectionEnd,direction:field.selectionDirection,scroll:field.scrollTop,left:field.scrollLeft,claim:intent.create(),reveal:true};
  setPreview(false);setFocusTick(value=>value+1);
 }
 async function publish(previous?:PublicationReceipt){
   if(inflight.current||!user||user.id!==owner||!canPublish)return;
   const receipt=previous||(pending&&samePublication(pending.receipt,interviewPublication(latest.current))?pending.receipt:undefined);
   if(!previous&&hasInterviewQuestionText(latest.current.questionScratch)){setQuestionError(t('还有问题记录未插入正文。请插入或清空后再发布。','A question record has not been inserted. Insert it or clear its fields before publishing.'));setQuestionOpen(true);return;}
   const submitted=receipt?{...(Array.isArray(receipt.payload.tag_ids)?{tagIds:receipt.payload.tag_ids}:{}),anonymous:receipt.payload.anonymous===true,title:receipt.payload.title as string,body:receipt.payload.body as string,interview:JSON.parse(receipt.payload.interview as string)} as InterviewDraft:latest.current;
   if(!receipt&&submitted.tagIds?.length&&!taxonomy.supported){setError(t('标签服务暂不可用，草稿已保留。','Tags are unavailable. Your draft is kept.'));return;}
   if(!receipt&&taxonomy.supported&&(!taxonomy.ready||!taxonomy.data.boards.some(row=>row.id==='career'&&row.enabled))){setError(t('求职版位暂未开放，草稿已保留。','The career board is unavailable. Your draft is kept.'));return;}
   if(!receipt&&taxonomy.supported&&unavailableCommunityTags(taxonomy.data.tags,submitted.tagIds||[]).length){setError(t('所选标签含停用或不可用项，请在标签区移除后发布。草稿已保留。','Some selected tags are paused or unavailable. Remove them in the tag section before publishing. Your draft is kept.'));return;}
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
     const same=sameInterviewDraft(latest.current,submitted);
     if(!removed&&same){setConfirmedPost(confirmed);setError(t('面经已发布，但暂时无法清理草稿。重试会继续确认同一篇面经。','Your report was published, but its draft could not be cleared. Retrying confirms the same report.'));return;}
     clearPublication(tabDraftStorage(),request.receipt);setPending(undefined);
     if(removed||same)onPublished(confirmed);else setConfirmedPost(confirmed);
   }catch(error){if(active.current&&currentOwner.current===owner)setError(errorMessage(error));if(affiliationRequired(error))void refreshAccount().catch(()=>{});}
   finally{inflight.current=false;if(active.current){setBusy(false);}}
 }
 return <section {...panelProps} ref={composer} className="interview-composer social-panel" aria-labelledby="interview-compose-heading">
  <div className="interview-compose-heading"><div><h2 id="interview-compose-heading">{t('分享面经','Share an interview experience')}</h2></div><button className="icon-button" type="button" disabled={busy} aria-label={t('关闭，保留草稿','Close and keep draft')} onClick={onClose}><X size={18}/></button></div>
  <p className="interview-composer-caption">{t('公司、岗位与知识点会公开展示；结果和地点可以不透露。可选择昵称或匿名发布，所属机构仍展示。','Company, role and linked concepts are public. Outcome and location are optional. Choose named or anonymous posting; affiliation remains visible.')}</p>
  {!durable&&<p role="alert" className="interview-notice">{t('草稿暂时无法保存到此标签页，请保持页面打开或复制内容。','This tab cannot save the draft. Keep this page open or copy your text.')}</p>}
  {error&&<p role="alert" className="interview-error">{error}</p>}
  {confirmedPost&&<p role="status" className="interview-notice">{t('面经已发布；当前草稿仍保留。','Your report was published; the current draft is preserved.')} <AppLink href={communityPostHref(confirmedPost.id,interviewHref({...emptyInterviewFilters(),company:confirmedPost.interview.company}))}>{t('查看已发布面经','View published report')}</AppLink></p>}
  <PublishingAffiliationNotice/>
  {!confirmedPost&&<PublicationRecovery pending={pending} newerDraft={uninserted} payload={interviewPublication(draft)} busy={busy} disabled={!canPublish} onRetry={()=>void publish(pending?.receipt)}/>}
  <form className="social-form" onSubmit={event=>{event.preventDefault();void publish();}}>
   <CommunityTagPicker value={draft.tagIds||[]} onChange={tagIds=>edit({tagIds})} disabled={busy}/><PublishingIdentity anonymous={!!draft.anonymous} onChange={anonymous=>edit({anonymous})} disabled={busy}/>
   <InterviewFields value={draft.interview} onChange={interview=>edit({interview})} problems={problems} disabled={busy}/>
   <label>{t('面经标题','Title')}<input ref={titleField} id={titleId} data-writing-title="" required maxLength={180} disabled={busy} value={draft.title} placeholder={t('例如：机器学习工程师技术面试复盘','e.g. ML engineering technical interview recap')} onChange={event=>edit({title:event.target.value})}/></label>
   <details className="interview-writing-guide" onToggle={event=>setWritingOpen(event.currentTarget.open)}><summary ref={writingSummary}><span><BookOpen size={15}/>{t('用提纲整理面试经历','Organize your experience with prompts')}</span><ChevronDown size={14}/></summary>{writingOpen&&<WritingGuide stage={draft.interview.stage} defaultId={defaultInterviewOutline(draft.interview.stage)} outlines={interviewOutlines} hasBody={!!draft.body.trim()} hasContent={contentReady} busy={busy} canUndo={!!outlineEdit&&draft.body===outlineEdit.after&&outlineEdit.before!==outlineEdit.after} editedAfterInsert={!!outlineEdit&&draft.body!==outlineEdit.after} onInsert={insertOutline} onUndo={undoOutline} onRestoreFocus={()=>writingSummary.current?.focus({preventScroll:true})}/>}</details>
   {writingError&&<p className="interview-writing-hint" role="status">{writingError}</p>}
   <MarkdownEditor sectionNavigation scopeKey={JSON.stringify(['interview',owner])} submitShortcut value={draft.body} onChange={body=>edit({body})} preview={preview} onPreviewChange={setPreview}
     textareaRef={editor} id={bodyId} label={t('流程、思路与复盘','Process, approach and reflection')} modeLabel={t('面经编辑模式','Report editing mode')}
     rows={12} maxLength={20000} busy={busy} describedBy={draft.body.trim()&&!contentReady&&!pendingMatches?contentHint:undefined}
     placeholder={t('记录轮次、考察方向、你的解题思路和准备建议。只分享可以公开的信息。','Describe the rounds, concepts, your approach and preparation tips. Share information you can make public.')}/>
   <details className="interview-question-guide" id={questionId} open={questionOpen} onToggle={event=>setQuestionOpen(event.currentTarget.open)}><summary ref={questionSummary}><BookOpen size={15} aria-hidden="true"/><span className="interview-question-summary-copy"><b>{t('整理一条问题与追问','Record a question and follow-ups')}</b><small>{uninserted?t('有内容尚未插入','Uninserted content'):t('选填记录','Optional record')}</small></span><ChevronDown className="interview-question-chevron" size={15} aria-hidden="true"/></summary>{questionOpen&&<QuestionBuilder value={draft.questionScratch||emptyInterviewQuestion()} questions={draft.interview.questions} problems={problems} busy={busy} error={questionError} canUndo={!!questionEdit&&draft.body===questionEdit.after&&draft.questionScratch===undefined} undoBlocked={!!questionEdit&&(draft.body!==questionEdit.after||draft.questionScratch!==undefined)} onChange={changeQuestion} onInsert={insertQuestion} onUndo={undoQuestion} onClear={clearQuestion} onRestoreFocus={()=>questionSummary.current?.focus({preventScroll:true})}/>}</details>
   {uninserted&&<p id={questionHint} className="interview-writing-hint" role="status">{t('问题记录仅在本标签页草稿中，尚未写入正文。插入或清空后即可发布当前面经；待确认的原发布仍可单独重试。','This question record stays in the tab draft and is not in the report. Insert or clear it before publishing the current draft. An unconfirmed original publication can still be retried separately.')}</p>}
   {draft.body.trim()&&!contentReady&&!pendingMatches&&<p id={contentHint} className="interview-writing-hint" role="status">{t('提纲已经准备好。补充你实际经历的流程、思路或复盘后即可发布。','Your outline is ready. Add your actual process, approach or reflection before publishing.')}</p>}
   <PublishingRules action="post"/>
   <InterviewDraftStatus step={draftStep} id={draftStatusId} controls={draftStep==='title'?titleId:draftStep==='record'?questionId:bodyId} disabled={busy} onNavigate={navigateDraft}/>
   <div className="form-footer"><button type="button" className="text-button" disabled={busy} onClick={()=>setDiscard(true)}>{t('丢弃草稿','Discard draft')}</button><small><Check size={12}/>{durable?t('草稿仅保存在本标签页','Draft stays in this tab'):t('草稿暂存在当前页面','Draft is held in this page')}</small><button className="button" disabled={busy||!canPublish||uninserted||!draft.title.trim()||!contentReady&&!pendingMatches||!!draft.anonymous&&capabilities?.community_anonymous_posts!==true} aria-describedby={[draftStatusId,uninserted?questionHint:draft.body.trim()&&!contentReady&&!pendingMatches?contentHint:undefined].filter(Boolean).join(' ')}>{busy?<span className="spinner"/>:<ChevronRight size={15}/>} {pending?pendingMatches?t('重试发布','Retry publication'):t('作为新面经发布','Publish as a new report'):t('发布面经','Publish report')}</button></div>
  </form>
  {discard&&<div className="draft-discard-confirm" role="alert"><p>{t('丢弃当前草稿？已发送但待确认的发布记录会保留，避免误发重复内容。','Discard this draft? Unconfirmed publication receipts are retained so they can be retried safely.')}</p><button type="button" disabled={busy} onClick={()=>{if(discardInterviewDraft(tabDraftStorage(),owner))onClose();else setError(t('暂时无法清除草稿，请重试。','The draft could not be cleared. Try again.'));}}>{t('确认丢弃','Discard draft')}</button><button type="button" onClick={()=>setDiscard(false)}>{t('继续编辑','Keep editing')}</button></div>}
 </section>;
}
