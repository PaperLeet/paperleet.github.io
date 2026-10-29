import {useWorkspaceIntent} from './useWorkspaceIntent';
import WritingFeedback from './WritingFeedback';
import {TranslatedReply} from './CommunityTranslation';
import {appLocation} from './appLocation';
import AppLink from './AppLink';
import {useEffect,useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import type {ReactNode} from 'react';
import {ChevronDown,ChevronUp,Heart,MessageCircle,PenLine,Quote,Repeat2,X} from 'lucide-react';
import {useI18n} from './i18n';
import {DateLabel,SocialError,SocialNotice,errorMessage,socialApi,useAccount} from './social';
import CommunityAuthor from './CommunityAuthor';
import InlineCommentBody from './InlineCommentBody';
import CardCommentSearch from './CardCommentSearch';
import ReplySearchContext from './ReplySearchContext';
import {FeatureBoundary} from './deferredFeature';
import ReplyEditor from './ReplyEditor';
import {canEditReply,type ReplyEditOrigin,type ReplyEditConfirmation} from './replyEdits';
import MarkdownEditor,{type MarkdownEditorView} from './MarkdownEditor';
import PublicationRecovery from './PublicationRecovery';
import {PublishingRules} from './PublishingRules';
import {canPublishCommunity} from './communityPublishing';
import PublishingAffiliationNotice from './PublishingAffiliationNotice';
import PostReport from './PostReport';
import DiscussionLink from './DiscussionLink';
import {communityPostHref} from './communityNavigation';
import {signInHref} from './accountNavigation';
import {tabDraftStorage} from './communityDrafts';
import {clearPublishedReplyDraft,hasReplyDraft,readReplyDraft,sameReplyDraft,writeReplyDraft,type ReplyDraft} from './replyDrafts';
import {clearPublication,preparePublication,readPublication,replyPublication,samePublication,type PublicationReceipt} from './publicationRequests';
import {withConfirmedReply,withConfirmedReplyEdit,type Post,type PostDetail,type Reply} from './communityDiscussion';
import {useDiscussionReplies,ReplyReadingReturn,ReplyReadingNotice,ReplyAuthorJump,ReplyAuthorNavigation,ReplyAuthorFilter,ReplyListCompact} from './useDiscussionReplies';
import {postPreview} from './postExcerpt';
import {quotePostDraft,quoteReplyDraft,captureReplyQuoteSelection,type DraftQuote,type ReplyQuoteSelection} from './replyQuotes';
import {controlIsActive} from './activeControls';
import './reply-quotes.css';
import {keepReadingTriggerVisible,readingScrollRoot} from './readingViewport';
import './community-card-actions.css';

type CardProps={canAct?:()=>boolean;post:Post;returnTo?:string;showActivity?:boolean;children?:ReactNode};
function CommentReference({parent,disabled,onReveal}:{parent:Reply;disabled:boolean;onReveal:(trigger:HTMLButtonElement)=>void}){
  const {t,locale}=useI18n();
  const name=parent.author.anonymous?t('匿名用户','Anonymous'):parent.author.display_name;
  const summary=useMemo(()=>postPreview(parent.body,72,locale==='en'?'[formula]':'[公式]').text,[parent.body,locale]);
  return <button type="button" className="card-comment-context" disabled={disabled} aria-label={t(`查看${name}的原评论`,`View original comment from ${name}`)} onClick={event=>onReveal(event.currentTarget)}><span>{t('回复','Replying to')} <b>{name}</b></span><small>{summary||t('查看原评论','View original comment')}</small></button>;
}
export default function CommunityCardActions(props:CardProps){
  const {user,loading}=useAccount();
  // Account changes discard rendered private state before the next request.
  return <CardActions key={`${props.post.id}:${user?.id||'guest'}`} {...props} owner={user?.id||''} accountLoading={loading}/>;
}
function CardActions({post:initial,returnTo,showActivity,owner,accountLoading,children,canAct}:CardProps&{owner:string;accountLoading:boolean}){
  const {t,locale}=useI18n(),{user,capabilities,refresh:refreshAccount}=useAccount(),regionId=useId(),fieldId=useId();
  const canPublish=canPublishCommunity(user,capabilities)&&!accountLoading;
  const href=communityPostHref(initial.id,returnTo),path=`community/posts/${encodeURIComponent(initial.id)}`;
  const [replyEdit,setReplyEdit]=useState<{id:string;origin:ReplyEditOrigin}>(),[editedReply,setEditedReply]=useState<string>();
  const editClosing=useRef<{id:string;origin:ReplyEditOrigin}>(undefined);
  const [post,setPost]=useState(initial),[detail,setDetail]=useState<PostDetail>(),[open,setOpen]=useState(false);
  const [postedId,setPostedId]=useState<string>();
  const [quoteNotice,setQuoteNotice]=useState<''|'added'|'exists'|'length'|'unavailable'>('');
  const quoteEdit=useRef<DraftQuote&{selection:ReplyQuoteSelection}>(undefined);
  const [loading,setLoading]=useState(false),[loadError,setLoadError]=useState(''),[error,setError]=useState(''),[replyError,setReplyError]=useState(''),[notice,setNotice]=useState('');
  const [busy,setBusy]=useState<'like'|'reply'|'account'|''>(''),[pendingLike,setPendingLike]=useState<boolean>();
  const [accountRecovery,setAccountRecovery]=useState(false),[recoveryFrom,setRecoveryFrom]=useState<'like'|'reply'>('like');
  const [draft,setDraft]=useState<ReplyDraft>(()=>readReplyDraft(tabDraftStorage(),owner,initial.id)||{body:''});
  const [saved,setSaved]=useState(true),[preview,setPreview]=useState(false),[writing,setWriting]=useState(false),[focusTick,setFocusTick]=useState(0);
  const [publication,setPublication]=useState(()=>readPublication(tabDraftStorage(),owner,'reply',initial.id));
  const current=useRef(draft),mounted=useRef(true),working=useRef(false),readVersion=useRef(0),request=useRef<AbortController|undefined>(undefined);
  const commentSearch=useRef<HTMLInputElement>(null);
  const textarea=useRef<HTMLTextAreaElement>(null),toggle=useRef<HTMLButtonElement>(null),composePrompt=useRef<HTMLButtonElement>(null);
  const card=useRef<HTMLDivElement>(null),shareHref=location.href,sharing=useRef({owner,accountLoading,href:shareHref});sharing.current={owner,accountLoading,href:shareHref};
  const canShare=()=>mounted.current&&!sharing.current.accountLoading&&sharing.current.owner===owner&&sharing.current.href===shareHref&&location.href===shareHref&&!!card.current?.isConnected&&!card.current.closest('[hidden],[aria-hidden="true"],[inert]');
  const editorView=useRef<MarkdownEditorView|undefined>(undefined);
  const minimized=useRef<{opener:HTMLElement;href:string}|undefined>(undefined),opening=useRef<{opener:Element|null;href:string;source:string;selection?:ReplyQuoteSelection;atEnd?:boolean}|undefined>(undefined),formId=fieldId+'-form';
  const closing=useRef<{opener:HTMLElement;href:string}|undefined>(undefined);
  const replyOperationHref=useRef('');
  const likeControl=useRef<HTMLButtonElement>(null),likeReturn=useRef<{href:string;claim:()=>boolean}>(undefined);
  const likeIntent=useWorkspaceIntent(JSON.stringify(['card-like-retry',owner,initial.id]));
  const retryLike=useRef<HTMLButtonElement>(null),accountFocus=useRef<{opener:Element|null;href:string}|undefined>(undefined);
  useLayoutEffect(()=>{
    const intent=closing.current;if(open||!intent)return;closing.current=undefined;
    const active=document.activeElement;
    if(location.href!==intent.href||active!==intent.opener&&!(active===document.body&&!intent.opener.isConnected))return;
    const control=toggle.current;if(!control)return;
    control.focus({preventScroll:true});
    keepReadingTriggerVisible(control,100);
  },[open]);
  useLayoutEffect(()=>{
    const intent=editClosing.current;if(replyEdit||!intent)return;editClosing.current=undefined;
    if(!open||intent.origin.href!==location.href)return;
    const active=document.activeElement,opener=intent.origin.opener;
    if(active!==opener&&!(active===document.body&&opener instanceof HTMLElement&&(!opener.isConnected||opener.matches(':disabled')||!!opener.closest('[hidden]'))))return;
    const control=[...replies.root.current?.querySelectorAll<HTMLElement>('[data-card-edit-reply]')||[]].find(node=>node.dataset.cardEditReply===intent.id)||replies.root.current||toggle.current;
    control?.focus({preventScroll:true});if(control)keepReadingTriggerVisible(control,100);
  },[replyEdit,open]);
  useLayoutEffect(()=>{
    const intent=minimized.current;if(writing||!intent)return;minimized.current=undefined;
    const active=document.activeElement;
    if(location.href===intent.href&&(active===intent.opener||active===document.body&&!!intent.opener.closest('[hidden]')))composePrompt.current?.focus({preventScroll:true});
  },[writing]);
  useLayoutEffect(()=>{
    const intent=opening.current;if(!intent||!writing||preview)return;opening.current=undefined;
    const active=document.activeElement;
    if(location.href===intent.href&&intent.source===current.current.body&&(active===intent.opener||active===document.body&&intent.opener instanceof HTMLElement&&(!intent.opener.isConnected||!!intent.opener.closest('[hidden]')))){
      const field=textarea.current;if(!field||field.value!==intent.source)return;
      if(readingScrollRoot(field)){field.focus({preventScroll:true});keepReadingTriggerVisible(field);}else field.focus();
      if(intent.selection){field.setSelectionRange(intent.selection.start,intent.selection.end,intent.selection.direction);field.scrollTop=intent.selection.scroll;field.scrollLeft=intent.selection.left;}
      else if(intent.atEnd){field.setSelectionRange(field.value.length,field.value.length,'none');field.scrollTop=field.scrollHeight;field.scrollLeft=0;}
    }
  },[writing,preview,focusTick]);
  useLayoutEffect(()=>{
    const view=editorView.current,field=textarea.current;
    if(!view||!open||!writing||preview||accountLoading||!field)return;
    editorView.current=undefined;
    if(view.value!==current.current.body||field.value!==view.value)return;
    field.setSelectionRange(view.selection.start,view.selection.end,view.selection.direction);
    field.scrollTop=view.selection.scroll;field.scrollLeft=view.selection.left||0;
  },[open,writing,preview,accountLoading,focusTick]);
  const rowIdPrefix=regionId+'-reply-';
  const replies=useDiscussionReplies({replies:detail?.replies,authorId:detail?.author.id||post.author.id,disabled:loading||accountLoading||accountRecovery||!!loadError||!!replyEdit,batchSize:3,initialOrder:'newest',rowIdPrefix,revealBlock:'start'});
  function revealComment(id:string,trigger:HTMLElement){replies.reveal(id,trigger);}
  const canFilterAuthor=()=>!busy&&!replies.disabled&&canShare()&&canAct?.()!==false;
  function showAllComments(trigger:HTMLButtonElement){
    if(!canFilterAuthor()||!controlIsActive(trigger))return;
    if(document.activeElement===trigger)replies.root.current?.querySelector<HTMLButtonElement>('[data-author-reply-filter]')?.focus({preventScroll:true});
    replies.chooseAuthor(false);
  }
  useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;readVersion.current++;request.current?.abort();};},[]);
  useEffect(()=>{
    if(busy||accountLoading||accountRecovery||!accountFocus.current)return;
    const intent=accountFocus.current;accountFocus.current=undefined;
    if(location.href!==intent.href||document.activeElement!==intent.opener&&!(document.activeElement===document.body&&!intent.opener?.isConnected))return;
    ((recoveryFrom==='like'?retryLike.current:undefined)||(writing?(!preview?textarea.current:toggle.current):composePrompt.current)||toggle.current)?.focus({preventScroll:true});
  },[busy,accountLoading,accountRecovery,preview,writing,recoveryFrom]);
  useLayoutEffect(()=>{
    const request=likeReturn.current;if(busy||!request)return;likeReturn.current=undefined;
    const allowed=request.claim();
    if(!allowed||location.href!==request.href||loading||accountLoading||accountRecovery||replyEdit||!canShare()||canAct?.()===false)return;
    const target=pendingLike===undefined?likeControl.current:retryLike.current;
    if(!target||target.disabled||!controlIsActive(target)||target.closest('.community-card-interaction')!==card.current)return;
    target.focus({preventScroll:true});keepReadingTriggerVisible(target,100);
  },[busy,error,pendingLike,loading,accountLoading,accountRecovery,replyEdit]);
  async function load(){
    const version=++readVersion.current;
    request.current?.abort();const controller=new AbortController();request.current=controller;
    setLoading(true);setLoadError('');const timer=setTimeout(()=>controller.abort(),20_000);
    try{
      const value=await socialApi<PostDetail>(path,undefined,controller.signal);
      if(!mounted.current||version!==readVersion.current)return false;
      if(value.id!==initial.id||!Array.isArray(value.replies))throw Error(t('评论响应不完整，请重试。','The comments response is incomplete. Please retry.'));
      setDetail(value);setPost(value);return true;
    }catch(e){if(mounted.current&&version===readVersion.current)setLoadError(controller.signal.aborted?t('加载评论超时，请重试。','Loading comments timed out. Please retry.'):errorMessage(e));return false;}
    finally{clearTimeout(timer);if(mounted.current&&version===readVersion.current)setLoading(false);}
  }
  function startReplyEdit(row:Reply,opener:HTMLElement){
    if(replyEdit||working.current||busy||loading||accountLoading||accountRecovery||!canEditReply(row,owner))return;
    setReplyEdit({id:row.id,origin:{opener,href:location.href}});setEditedReply(undefined);
  }
  function closeReplyEdit(origin:ReplyEditOrigin={opener:document.activeElement,href:location.href}){
    if(replyEdit)editClosing.current={id:replyEdit.id,origin};setReplyEdit(undefined);
  }
  function savedReplyEdit(confirmed:Reply,confirmation?:ReplyEditConfirmation){
    if(!replyEdit)return;
    readVersion.current++;request.current?.abort();request.current=undefined;setLoading(false);setLoadError('');
    const base=confirmation?.baseVersion||detail?.replies.find(row=>row.id===confirmed.id)?.edit_version||'';
    setDetail(previous=>withConfirmedReplyEdit(previous,confirmed,base));
    if(!confirmation?.keepEditing){setEditedReply(confirmed.id);closeReplyEdit(confirmation?.origin);}
  }
  function collapse(){
    setOpen(false);replies.clearReadingReturn();
    // A hidden preview must not keep other card actions waiting on its read.
    readVersion.current++;request.current?.abort();request.current=undefined;setLoading(false);
  }
  function dismissComments(opener:HTMLElement){closing.current={opener,href:location.href};collapse();}
  function expand(){
    closing.current=undefined;
    if(open){collapse();return;}
    // Read the same per-thread draft used by the full discussion, including its reply target.
    const restored=busy||!saved?current.current:readReplyDraft(tabDraftStorage(),owner,initial.id)||{body:''};
    current.current=restored;setDraft(restored);setPublication(readPublication(tabDraftStorage(),owner,'reply',initial.id));
    setOpen(true);if(!busy)void load();
  }
  function focusReply(selection?:ReplyQuoteSelection,atEnd=false){
    opening.current={opener:document.activeElement,href:location.href,source:current.current.body,selection,atEnd};setFocusTick(value=>value+1);
  }
  function writeComment(){
    if(replyEdit)return;
    if(!owner){appLocation.hash=signInHref(returnTo||'community');return;}
    if(!open)expand();
    setWriting(true);setPreview(false);focusReply();
  }
  function edit(change:Partial<ReplyDraft>){
    const next={...current.current,...change};current.current=next;setDraft(next);
    setSaved(writeReplyDraft(tabDraftStorage(),owner,initial.id,next));setNotice('');setQuoteNotice('');
  }
  function quoteComment(row:Reply){
    if(replyEdit||!owner||!mounted.current||working.current||busy||loading||accountLoading||accountRecovery||!detail?.replies.some(reply=>reply.id===row.id))return;
    const previous=quoteEdit.current;
    if(previous?.replyId===row.id&&previous.sourceBody===row.body&&sameReplyDraft(previous.after,current.current)){setQuoteNotice('exists');setWriting(true);setPreview(false);focusReply(undefined,true);return;}
    const name=row.author.anonymous?t('匿名用户','Anonymous'):row.author.display_name||t('用户','User');
    const result=quoteReplyDraft(current.current,initial.id,row,name,locale==='en');
    if(!result.ok){setQuoteNotice(result.reason);return;}
    quoteEdit.current={...result.quote,selection:captureReplyQuoteSelection(textarea.current,current.current.body.length)};
    edit(result.quote.after);setQuoteNotice('added');setWriting(true);setPreview(false);focusReply(undefined,true);
  }
  function quotePost(trigger:HTMLButtonElement){
    if(replyEdit||working.current||busy||accountLoading||accountRecovery||!canShare()||canAct?.()===false||!controlIsActive(trigger))return;
    if(!owner){appLocation.hash=signInHref(returnTo||'community');return;}
    if(!open)expand();
    const result=quotePostDraft(current.current,initial.id,initial,locale==='en');
    setWriting(true);setPreview(false);
    if(!result.ok){setQuoteNotice(result.reason);focusReply(undefined,true);return;}
    quoteEdit.current={...result.quote,selection:captureReplyQuoteSelection(textarea.current,current.current.body.length)};
    edit(result.quote.after);setQuoteNotice('added');focusReply(undefined,true);
  }
  function undoQuote(trigger:HTMLButtonElement){
    const operation=quoteEdit.current;
    if(replyEdit||!operation||working.current||busy||accountLoading||accountRecovery||!canShare()||canAct?.()===false||!controlIsActive(trigger)||!sameReplyDraft(operation.after,current.current))return;
    edit({body:operation.before.body,parentId:operation.before.parentId});quoteEdit.current=undefined;setWriting(true);setPreview(false);focusReply(operation.selection);
  }
  async function reportFailure(e:unknown,operation:'like'|'reply'){
    if(!mounted.current)return;
    (operation==='reply'?setReplyError:setError)(errorMessage(e));
    if(e instanceof SocialError&&(e.status===401||e.code==='account_changed')){
      setRecoveryFrom(operation);setAccountRecovery(true);
      try{await refreshAccount();if(mounted.current)setAccountRecovery(false);}
      catch{/* Keep the saved draft and show explicit sign-in/account recovery controls. */}
    }
    if(e instanceof SocialError&&e.code==='affiliation_required')void refreshAccount().catch(()=>{});
  }
  async function confirmAccount(){
    if(working.current)return;
    accountFocus.current={opener:document.activeElement,href:location.href};
    working.current=true;setBusy('account');
    try{await refreshAccount();if(mounted.current)setAccountRecovery(false);}
    catch(e){if(mounted.current)(recoveryFrom==='reply'?setReplyError:setError)(errorMessage(e));}
    finally{working.current=false;if(mounted.current)setBusy('');}
  }
  async function like(active:boolean,opener?:HTMLButtonElement){
    if(opener&&(!canShare()||canAct?.()===false||!controlIsActive(opener)))return;
    if(replyEdit||working.current||accountLoading||loading||accountRecovery)return;
    if(!owner){appLocation.hash=signInHref(returnTo||'community');return;}
    if(opener&&document.activeElement===opener)likeReturn.current={href:location.href,claim:likeIntent.create()};
    working.current=true;setBusy('like');setError('');
    try{
      const value=await socialApi<Post>(`${path}/like`,{active,expected_user_id:owner},AbortSignal.timeout(20_000));
      if(value.id!==initial.id||value.viewer.liked!==active||!Number.isSafeInteger(value.counts.likes))throw Error(t('点赞状态尚未确认，请重试。','The like state is unconfirmed. Please retry.'));
      if(mounted.current){setPost(value);setPendingLike(undefined);}
    }catch(e){if(mounted.current)setPendingLike(active);await reportFailure(e,'like');}
    finally{working.current=false;if(mounted.current)setBusy('');}
  }
  async function publish(previous?:PublicationReceipt){
    const submitted={...current.current};
    if(replyEdit||!owner||!canPublish||!writing||working.current||accountLoading||accountRecovery||loading||!detail||!previous&&(!submitted.body.trim()||submitted.parentId&&!detail.replies.some(row=>row.id===submitted.parentId)))return;
    replyOperationHref.current=location.href;working.current=true;setBusy('reply');setReplyError('');setNotice('');
    try{
      const pending=preparePublication(tabDraftStorage(),owner,'reply',initial.id,previous?.payload||replyPublication(submitted),previous);
      setPublication(pending);
      const row=await socialApi<Reply>(`${path}/replies`,{...pending.receipt.payload,request_id:pending.receipt.requestId,expected_user_id:owner},AbortSignal.timeout(20_000));
      if(row.post_id!==initial.id||!row.id)throw Error(t('回复状态尚未确认，请重试。','The reply state is unconfirmed. Please retry.'));
      clearPublication(tabDraftStorage(),pending.receipt);
      const sendsCurrent=samePublication(pending.receipt,replyPublication(submitted));
      if(sendsCurrent)clearPublishedReplyDraft(tabDraftStorage(),owner,initial.id,submitted);
      if(!mounted.current)return;
      setPublication(undefined);
      setDetail(value=>withConfirmedReply(value,row));
      setPost(value=>({...value,counts:{...value.counts,replies:value.counts.replies+(detail.replies.some(item=>item.id===row.id)?0:1)},last_reply_at:row.created_at}));
      if(sendsCurrent&&sameReplyDraft(current.current,submitted)){current.current={body:''};setDraft(current.current);setSaved(true);setPreview(false);setQuoteNotice('');quoteEdit.current=undefined;}
      setNotice(t('回复已发布。','Reply posted.'));
      setPostedId(row.id);
    }catch(e){await reportFailure(e,'reply');}
    finally{working.current=false;if(mounted.current)setBusy('');}
  }
  const editingRow=replyEdit?detail?.replies.find(row=>row.id===replyEdit.id):undefined;
  const target=draft.parentId?detail?.replies.find(row=>row.id===draft.parentId):undefined;
  const missingTarget=!!draft.parentId&&!!detail&&!target;
  const replyFeedbackVisible=open&&writing&&!replyEdit&&!accountLoading&&!!owner&&!!user;
  const inlineAccountRecovery=replyFeedbackVisible&&recoveryFrom==='reply';
  const accountRecoveryControls=accountRecovery&&<div className="card-account-recovery"><p>{t('账号状态尚未确认，草稿已保留。确认账号后可重试原操作。','Your account is unconfirmed. Your draft is kept. Confirm your account before retrying.')}</p><button type="button" className="text-button" disabled={!!busy} onClick={()=>void confirmAccount()}>{t('刷新账号','Refresh account')}</button><AppLink href={signInHref(returnTo||'community')}>{t('重新登录','Sign in again')}</AppLink></div>;
  const replyErrorId=fieldId+'-publication-error';
  return <div ref={card} className="community-card-interaction">
    <div className="post-card-counts card-actions">
      <button ref={toggle} type="button" onClick={expand} aria-expanded={open} aria-controls={regionId} disabled={accountLoading}>
        <MessageCircle size={15}/><span>{post.counts.replies} <span className="card-count-label">{t('评论',post.counts.replies===1?'comment':'comments')}</span></span>{open?<ChevronUp size={13}/>:<ChevronDown size={13}/>}
      </button>
      <button ref={likeControl} type="button" onClick={()=>void like(!post.viewer.liked)} className={post.viewer.liked?'is-liked':''} aria-pressed={!!post.viewer.liked} disabled={!!replyEdit||!!busy||loading||accountLoading||accountRecovery||pendingLike!==undefined}>
        <Heart size={15} fill={post.viewer.liked?'currentColor':'none'}/><span>{post.counts.likes} {busy==='like'?t('确认中…','Confirming…'):<span className="card-count-label">{t('赞',post.counts.likes===1?'like':'likes')}</span>}</span>
      </button>
      <button type="button" onClick={writeComment} aria-controls={regionId} disabled={!!replyEdit||!!busy||accountLoading||accountRecovery}><PenLine size={15}/>{t('写评论','Write comment')}</button>
      <button type="button" className="card-post-quote" aria-label={t('引用正文到评论草稿','Quote post into comment draft')} title={t('将正文原文摘要加入评论草稿','Add a source-linked excerpt of the original post to your comment draft')} disabled={!!replyEdit||!!busy||accountLoading||accountRecovery} onClick={event=>quotePost(event.currentTarget)}><Quote size={15} aria-hidden="true"/><span>{t('引用正文','Quote post')}</span></button>
      <span className="card-reposts"><Repeat2 size={14} aria-hidden="true"/><span>{post.counts.reposts} <span className="card-count-label">{t('转发',post.counts.reposts===1?'repost':'reposts')}</span></span></span>
      <PostReport postId={post.id} postTitle={post.title} canAct={()=>canShare()&&canAct?.()!==false} returnTo={href} own={post.viewer.can_edit||post.author.id===owner} disabled={accountLoading||!!replyEdit||!!busy}/>
      <DiscussionLink key={`post-share:${accountLoading}`} postId={post.id} returnTo={returnTo} compact name={post.title} disabled={accountLoading} canAct={canShare}/>
      {showActivity&&post.last_reply_at&&<span className="post-card-activity">{t('最近回复','Last reply')} <DateLabel value={post.last_reply_at}/></span>}
      {children&&<div className="card-actions-extras">{children}</div>}
    </div>
    {(error||replyError&&!replyFeedbackVisible||accountRecovery&&!inlineAccountRecovery)&&<div className="card-action-error">
      {error&&<div className="card-like-error"><SocialNotice error={error}/>{pendingLike!==undefined&&<button ref={retryLike} type="button" className="text-button" disabled={!!busy||loading||accountRecovery} onClick={event=>void like(pendingLike,event.currentTarget)}>{t('重试原点赞操作','Retry like action')}</button>}</div>}
      {replyError&&!replyFeedbackVisible&&<div className="card-comment-error"><SocialNotice error={replyError}/><button type="button" className="text-button" aria-controls={formId} disabled={!!replyEdit||!!busy||accountLoading||accountRecovery} onClick={event=>{if(location.href===replyOperationHref.current&&canShare()&&canAct?.()!==false&&controlIsActive(event.currentTarget))writeComment();}}>{t('继续处理评论','Review comment')}</button></div>}
      {!inlineAccountRecovery&&accountRecoveryControls}
    </div>}
    {open&&<section id={regionId} ref={replies.root} tabIndex={-1} className="card-comments" aria-label={t('评论预览','Comment preview')} aria-keyshortcuts="Escape" onKeyDown={event=>{
      if(event.key!=='Escape'||event.defaultPrevented||event.repeat||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
      const active=document.activeElement;
      if(!(active instanceof HTMLElement)||!event.currentTarget.contains(active)||active.closest('.card-reply-form,input,textarea,select,[contenteditable]:not([contenteditable="false"])'))return;
      event.preventDefault();event.stopPropagation();dismissComments(active);
    }}>
      <header><strong>{t('评论','Comments')}</strong>{(detail?.replies.length||0)>1&&<select className="card-comment-order" aria-label={t('评论排序','Sort comments')} aria-controls={regionId+'-list'} value={replies.order} disabled={replies.disabled} onChange={event=>replies.chooseOrder(event.target.value as 'newest'|'oldest')}><option value="newest">{t('最新回复','Newest first')}</option><option value="oldest">{t('最早回复','Oldest first')}</option></select>}{detail&&<ReplyAuthorFilter list={replies} disabled={!!busy} controls={regionId+'-list'} canAct={canFilterAuthor}/>}<ReplyAuthorJump list={replies} disabled={!!busy}/><span role="status">{!loading&&!loadError&&detail?t(`显示 ${replies.shown.length} / ${replies.total} 条`,`${replies.shown.length} of ${replies.total} shown`):''}</span><AppLink href={href}>{t('完整讨论','Full discussion')}</AppLink></header>
      <ReplyReadingNotice list={replies}/><ReplyReadingReturn list={replies}/>
      {detail&&(detail.replies.length>0||replies.query)&&<CardCommentSearch title={post.title} list={replies} total={detail.replies.length} controls={regionId+'-list'} canAct={canShare} inputRef={commentSearch}/>}
      <div id={regionId+'-list'} className="card-comment-results" aria-busy={loading||undefined}>
      {loading&&<p role="status">{t('正在加载评论…','Loading comments…')}</p>}
      {loadError&&<div><SocialNotice error={loadError}/><button type="button" className="text-button" disabled={!!busy} onClick={()=>void load()}>{t('重试加载评论','Retry loading comments')}</button></div>}
      {!loading&&!loadError&&detail&&<>
        {replies.total?<ol className="card-comment-list">{replies.shown.map(row=>{const parent=row.parent_id?replies.byId.get(row.parent_id):undefined;return <li key={row.id} id={rowIdPrefix+row.id} tabIndex={-1} data-card-reply={row.id} data-discussion-reply={row.id} data-comment-located={replies.revealed===row.id||undefined}>
          <div className="card-comment-meta"><CommunityAuthor user={row.author} compact/>{row.author.id===post.author.id&&<span className="reply-author-badge">{t('楼主','Author')}</span>}<DateLabel value={row.created_at}/><ReplyAuthorNavigation list={replies} replyId={row.id} disabled={!!busy}/>{row.updated_at&&row.updated_at!==row.created_at&&<span className="card-comment-edited">{t('已编辑','Edited')} <DateLabel value={row.updated_at}/></span>}</div>
          {row.parent_id&&(parent?<CommentReference parent={parent} disabled={accountLoading||!!replyEdit} onReveal={trigger=>revealComment(parent.id,trigger)}/>:<span className="card-comment-context unavailable">{t('原评论已删除或暂不可用','The original comment is deleted or unavailable')}</span>)}
          {replies.query?<ReplySearchContext reply={row} list={replies} lazy canAct={canShare}><FeatureBoundary fallback={<><p className="inline-reading-error" role="status">{t('格式暂不可用，以下为原文。收起后可重新展开。','Formatting is unavailable. Original text is shown below; collapse and reopen to retry.')}</p><pre className="inline-reading-plain">{row.body}</pre></>}><TranslatedReply reply={row}/></FeatureBoundary></ReplySearchContext>:<TranslatedReply reply={row}>{body=><InlineCommentBody canAct={()=>canShare()&&canAct?.()!==false} body={row.body} displayBody={body}/>}</TranslatedReply>}
          <div className="card-comment-links"><AppLink href={communityPostHref(post.id,returnTo,row.id)}>{t('在帖子中查看','View in discussion')}</AppLink>
            {owner&&<button type="button" disabled={!!replyEdit||!!busy||accountLoading} onClick={()=>{edit({parentId:row.id});setWriting(true);setPreview(false);focusReply();}}>{t('回复','Reply')}</button>}{owner&&<button type="button" disabled={!!replyEdit||!!busy||loading||accountLoading||accountRecovery} aria-label={t(`引用${row.author.anonymous?'匿名用户':row.author.display_name||'用户'}的评论`,`Quote comment from ${row.author.anonymous?'Anonymous':row.author.display_name||'user'}`)} onClick={()=>quoteComment(row)}><Quote size={12} aria-hidden="true"/>{t('引用','Quote')}</button>}
            {canEditReply(row,owner)&&<button type="button" data-card-edit-reply={row.id} aria-controls={fieldId+'-edit'} aria-expanded={replyEdit?.id===row.id&&open} disabled={!!replyEdit||!!busy||loading||accountLoading||accountRecovery} onClick={event=>startReplyEdit(row,event.currentTarget)}><PenLine size={12} aria-hidden="true"/>{t('编辑评论','Edit comment')}</button>}
            <DiscussionLink key={`reply-share:${accountLoading}`} postId={post.id} replyId={row.id} returnTo={returnTo} compact showReplyLink={false} name={row.author.anonymous?t('匿名用户','Anonymous'):row.author.display_name} disabled={accountLoading} canAct={canShare}/>
            <ReplyReadingReturn list={replies} replyId={row.id}/>
          </div>
        </li>;})}</ol>:<div className="card-comments-empty" role="status">{replies.query?<><p>{replies.authorOnly?t('楼主回复中没有匹配结果','No matching author replies'):t('没有匹配的评论','No matching comments')}</p><button type="button" className="text-button" disabled={replies.disabled} onClick={event=>{if(replies.disabled||!canShare())return;if(document.activeElement===event.currentTarget)commentSearch.current?.focus({preventScroll:true});replies.chooseQuery('');}}>{t('清空评论搜索','Clear comment search')}</button></>:replies.authorOnly?t('楼主暂未公开回复，其他人的讨论仍可查看。','The author has not replied publicly yet. You can still read everyone else’s comments.'):t('还没有评论，来聊聊你的想法。','No comments yet. Share your thoughts.')}{replies.authorOnly&&<button type="button" className="text-button" disabled={!!busy||replies.disabled} onClick={event=>showAllComments(event.currentTarget)}>{t('查看全部评论','Show all comments')}</button>}</div>}
        {(replies.total>replies.shown.length||replies.canCompact)&&<div className="card-comment-paging">{replies.total>replies.shown.length&&<button type="button" className="card-more-comments" disabled={accountLoading||!!replyEdit} onClick={replies.showMore}><ChevronDown size={14}/>{t('再展开','Show')} {Math.min(3,replies.total-replies.shown.length)} {t('条评论','more comments')}</button>}<ReplyListCompact list={replies} disabled={!!busy} controls={regionId+'-list'} canAct={canFilterAuthor}/></div>}
      </>}
      </div>
      {quoteNotice&&<div className="reply-quote-notice card-reply-quote-notice" role="status"><span>{quoteNotice==='added'?t('引用摘要已加入草稿，可继续编辑后发布。','A source-linked excerpt is in your draft. Edit it before posting.'):quoteNotice==='exists'?t('该引用已在草稿中，可继续编辑。','This excerpt is already in your draft. You can keep editing.'):quoteNotice==='length'?t('引用会超过评论长度上限，现有草稿已保留。请精简后再引用。','This quote would exceed the comment limit. Your draft is kept; shorten it before quoting.'):t('暂时无法生成引用，仍可直接回复。','This content could not be quoted. You can still reply directly.')}</span>{(quoteNotice==='added'||quoteNotice==='exists')&&quoteEdit.current&&sameReplyDraft(quoteEdit.current.after,draft)&&<button type="button" className="text-button" disabled={!!replyEdit||!!busy||accountLoading||accountRecovery} onClick={event=>undoQuote(event.currentTarget)}>{t('撤销引用','Undo quote')}</button>}</div>}
      {editedReply&&<p className="card-comment-edit-notice" role="status">{t('评论修改已保存。','Comment changes saved.')} {replies.byId.has(editedReply)&&<button type="button" className="text-button" onClick={event=>revealComment(editedReply,event.currentTarget)}>{t('查看修改的评论','View updated comment')}</button>}</p>}
      {accountLoading?<p className="card-account-checking" role="status">{t('正在确认账号，评论草稿已保留…','Checking your account. Your comment draft is kept…')}</p>:owner&&user?<>
      <button ref={composePrompt} type="button" className="card-comment-compose-prompt" hidden={writing||!!replyEdit} aria-expanded={false} aria-controls={formId} disabled={!!busy||accountRecovery} onClick={writeComment}><PenLine size={16} aria-hidden="true"/><span><b>{publication?t('继续处理待确认评论','Continue pending comment'):hasReplyDraft(draft)?t('继续编辑评论草稿','Continue comment draft'):t('写下你的想法…','Share your thoughts…')}</b><small>{!saved?t('草稿未能保存，请先复制内容。','Draft could not be saved. Copy it first.'):publication||hasReplyDraft(draft)?t('草稿已保留，点击继续','Your draft is kept. Pick up where you left off.'):t('无需离开当前列表','Stay in this feed')}</small></span></button>
      <form id={formId} hidden={!writing||!!replyEdit} aria-label={t('评论编辑器','Comment editor')} className="social-form card-reply-form" onSubmit={event=>{event.preventDefault();void publish();}}>
        <div className="card-reply-heading"><CommunityAuthor user={post.anonymous&&post.viewer.can_edit?post.author:user} compact/><button type="button" className="text-button" disabled={!!busy} aria-expanded={true} aria-controls={formId} onClick={event=>{minimized.current={opener:event.currentTarget,href:location.href};setWriting(false);}}><ChevronUp size={14} aria-hidden="true"/>{t('收起编辑器','Minimize editor')}</button></div>
        <PublishingAffiliationNotice/>
        {draft.parentId&&<div className="reply-target"><span>{t('回复','Replying to')} {target?.author.anonymous?t('匿名用户','Anonymous'):target?.author.display_name||t('原评论','original comment')}</span><button type="button" className="icon-button" aria-label={t('取消指定回复','Cancel reply target')} disabled={!!busy} onClick={()=>edit({parentId:undefined})}><X size={14}/></button></div>}
        {missingTarget&&<SocialNotice error={t('原评论已不可用，请取消指定回复后再发布。草稿已保留。','The original comment is unavailable. Cancel the reply target before posting. Your draft is kept.')}/>}
        <MarkdownEditor scopeKey={JSON.stringify(['inline-reply',owner,initial.id])} submitShortcut textareaRef={textarea} viewStateRef={editorView} id={fieldId} label={t('写评论','Write a comment')} modeLabel={t('评论编辑模式','Comment editor mode')} value={draft.body} onChange={body=>edit({body})} rows={3} maxLength={10000} busy={!!busy} disabled={accountRecovery} preview={preview} onPreviewChange={value=>{setPreview(value);if(!value)focusReply();}} placeholder={t('写下你的想法…','Share your thoughts…')}/>
        <PublishingRules action="reply"/>
        {(replyError||publication||accountRecovery&&inlineAccountRecovery)&&<WritingFeedback error={replyFeedbackVisible?replyError:undefined} errorId={replyErrorId}>
          {inlineAccountRecovery&&accountRecoveryControls}
          <PublicationRecovery pending={publication} payload={replyPublication(draft)} busy={!!busy||loading||accountRecovery} disabled={!canPublish} onRetry={()=>void publish(publication?.receipt)}/>
        </WritingFeedback>}
        <footer><div className="card-reply-feedback"><small role="status">{!saved?t('草稿未能保存，离开前请复制内容。','Draft could not be saved. Copy it before leaving.'):notice|| (hasReplyDraft(draft)?t('草稿已保存，可在完整讨论中继续编辑。','Draft saved. Continue editing in the full discussion.'):t('折叠评论不会丢失草稿。','Collapsing keeps your draft.'))}</small>{postedId&&replies.byId.has(postedId)&&<button type="button" className="text-button card-posted-reply" disabled={loading||accountLoading||!!loadError} onClick={event=>revealComment(postedId,event.currentTarget)}>{t('查看刚发布的评论','View your posted comment')}</button>}</div><button className="button" aria-describedby={replyFeedbackVisible&&replyError?replyErrorId:undefined} disabled={!!busy||!canPublish||accountRecovery||loading||!detail||!!loadError||!draft.body.trim()||missingTarget}>{busy==='reply'?t('发布中…','Posting…'):publication?samePublication(publication.receipt,replyPublication(draft))?t('重试发布','Retry publication'):t('作为新内容发布','Publish as new content'):t('发布评论','Post comment')}</button></footer>
      </form></>:<AppLink className="card-comment-signin" href={signInHref(returnTo||'community')}>{t('登录后参与评论','Sign in to comment')}</AppLink>}
      <button type="button" className="card-collapse-comments" aria-keyshortcuts="Escape" title={t('收起评论（Esc）','Collapse comments (Esc)')} onClick={event=>dismissComments(event.currentTarget)}><ChevronUp size={14} aria-hidden="true"/>{t('收起评论','Collapse comments')}</button>
    </section>}
    {replyEdit&&<div id={fieldId+'-edit'} className="card-comment-editor" hidden={!open}>
      {editingRow&&canEditReply(editingRow,owner)?<ReplyEditor canNavigate={()=>canShare()&&canAct?.()!==false} key={owner+':'+editingRow.id} compact reply={editingRow} owner={owner} focusOrigin={replyEdit.origin} disabled={!open||!!busy||loading||accountLoading||accountRecovery||!!loadError} onRefresh={load} onClose={()=>closeReplyEdit()} onSaved={savedReplyEdit}/>:<div role="status"><p>{t('这条评论已删除或暂不可编辑，修改草稿已保留。','This comment is deleted or cannot be edited now. Your edit draft is kept.')}</p><button type="button" className="text-button" onClick={()=>closeReplyEdit()}>{t('关闭编辑','Close editor')}</button></div>}
    </div>}
  </div>;
}
