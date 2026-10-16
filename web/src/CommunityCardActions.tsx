import {useEffect,useId,useLayoutEffect,useMemo,useRef,useState} from 'react';
import type {ReactNode} from 'react';
import {ChevronDown,ChevronUp,Heart,MessageCircle,PenLine,Repeat2,X} from 'lucide-react';
import {useI18n} from './i18n';
import {DateLabel,SocialError,SocialNotice,errorMessage,socialApi,useAccount} from './social';
import CommunityAuthor from './CommunityAuthor';
import PostPreview from './PostPreview';
import MarkdownEditor from './MarkdownEditor';
import PublicationRecovery from './PublicationRecovery';
import {PublishingRules} from './PublishingRules';
import {communityPostHref} from './communityNavigation';
import {signInHref} from './accountNavigation';
import {tabDraftStorage} from './communityDrafts';
import {clearPublishedReplyDraft,hasReplyDraft,readReplyDraft,sameReplyDraft,writeReplyDraft,type ReplyDraft} from './replyDrafts';
import {clearPublication,preparePublication,readPublication,replyPublication,samePublication,type PublicationReceipt} from './publicationRequests';
import {withConfirmedReply,type Post,type PostDetail,type Reply} from './communityDiscussion';
import {useDiscussionReplies,ReplyReadingReturn,ReplyReadingNotice} from './useDiscussionReplies';
import {postPreview} from './postExcerpt';
import './community-card-actions.css';

type CardProps={post:Post;returnTo?:string;showActivity?:boolean;children?:ReactNode};
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
function CardActions({post:initial,returnTo,showActivity,owner,accountLoading,children}:CardProps&{owner:string;accountLoading:boolean}){
  const {t}=useI18n(),{user,refresh:refreshAccount}=useAccount(),regionId=useId(),fieldId=useId();
  const href=communityPostHref(initial.id,returnTo),path=`community/posts/${encodeURIComponent(initial.id)}`;
  const [post,setPost]=useState(initial),[detail,setDetail]=useState<PostDetail>(),[open,setOpen]=useState(false);
  const [postedId,setPostedId]=useState<string>();
  const [loading,setLoading]=useState(false),[loadError,setLoadError]=useState(''),[error,setError]=useState(''),[notice,setNotice]=useState('');
  const [busy,setBusy]=useState<'like'|'reply'|'account'|''>(''),[pendingLike,setPendingLike]=useState<boolean>();
  const [accountRecovery,setAccountRecovery]=useState(false);
  const [draft,setDraft]=useState<ReplyDraft>(()=>readReplyDraft(tabDraftStorage(),owner,initial.id)||{body:''});
  const [saved,setSaved]=useState(true),[preview,setPreview]=useState(false),[writing,setWriting]=useState(false),[focusTick,setFocusTick]=useState(0);
  const [publication,setPublication]=useState(()=>readPublication(tabDraftStorage(),owner,'reply',initial.id));
  const current=useRef(draft),mounted=useRef(true),working=useRef(false),readVersion=useRef(0),request=useRef<AbortController|undefined>(undefined);
  const textarea=useRef<HTMLTextAreaElement>(null),toggle=useRef<HTMLButtonElement>(null),composePrompt=useRef<HTMLButtonElement>(null);
  const minimized=useRef<{opener:HTMLElement;href:string}|undefined>(undefined),opening=useRef<{opener:Element|null;href:string}|undefined>(undefined),formId=fieldId+'-form';
  const retryLike=useRef<HTMLButtonElement>(null),accountFocus=useRef<{opener:Element|null;href:string}|undefined>(undefined);
  useLayoutEffect(()=>{
    const intent=minimized.current;if(writing||!intent)return;minimized.current=undefined;
    const active=document.activeElement;
    if(location.href===intent.href&&(active===intent.opener||active===document.body&&!!intent.opener.closest('[hidden]')))composePrompt.current?.focus({preventScroll:true});
  },[writing]);
  useLayoutEffect(()=>{
    const intent=opening.current;if(!intent||!writing||preview)return;opening.current=undefined;
    const active=document.activeElement;
    if(location.href===intent.href&&(active===intent.opener||active===document.body&&intent.opener instanceof HTMLElement&&!!intent.opener.closest('[hidden]')))textarea.current?.focus();
  },[writing,preview,focusTick]);
  const rowIdPrefix=regionId+'-reply-';
  const replies=useDiscussionReplies({replies:detail?.replies,disabled:loading||accountLoading||!!loadError,batchSize:3,initialOrder:'newest',rowIdPrefix,revealBlock:'center'});
  function revealComment(id:string,trigger:HTMLElement){replies.reveal(id,trigger);}
  useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;readVersion.current++;request.current?.abort();};},[]);
  useEffect(()=>{
    if(busy||accountLoading||accountRecovery||!accountFocus.current)return;
    const intent=accountFocus.current;accountFocus.current=undefined;
    if(location.href!==intent.href||document.activeElement!==intent.opener&&!(document.activeElement===document.body&&!intent.opener?.isConnected))return;
    (retryLike.current||(writing?(!preview?textarea.current:toggle.current):composePrompt.current)||toggle.current)?.focus({preventScroll:true});
  },[busy,accountLoading,accountRecovery,preview,writing]);
  async function load(){
    const version=++readVersion.current;
    request.current?.abort();const controller=new AbortController();request.current=controller;
    setLoading(true);setLoadError('');const timer=setTimeout(()=>controller.abort(),20_000);
    try{
      const value=await socialApi<PostDetail>(path,undefined,controller.signal);
      if(!mounted.current||version!==readVersion.current)return;
      if(value.id!==initial.id||!Array.isArray(value.replies))throw Error(t('评论响应不完整，请重试。','The comments response is incomplete. Please retry.'));
      setDetail(value);setPost(value);
    }catch(e){if(mounted.current&&version===readVersion.current)setLoadError(controller.signal.aborted?t('加载评论超时，请重试。','Loading comments timed out. Please retry.'):errorMessage(e));}
    finally{clearTimeout(timer);if(mounted.current&&version===readVersion.current)setLoading(false);}
  }
  function expand(){
    if(open){setOpen(false);replies.clearReadingReturn();return;}
    // Read the same per-thread draft used by the full discussion, including its reply target.
    const restored=busy||!saved?current.current:readReplyDraft(tabDraftStorage(),owner,initial.id)||{body:''};
    current.current=restored;setDraft(restored);setPublication(readPublication(tabDraftStorage(),owner,'reply',initial.id));
    setOpen(true);if(!busy)void load();
  }
  function focusReply(){
    opening.current={opener:document.activeElement,href:location.href};setFocusTick(value=>value+1);
  }
  function writeComment(){
    if(!owner){location.hash=signInHref(returnTo||'community');return;}
    if(!open)expand();
    setWriting(true);setPreview(false);focusReply();
  }
  function edit(change:Partial<ReplyDraft>){
    const next={...current.current,...change};current.current=next;setDraft(next);
    setSaved(writeReplyDraft(tabDraftStorage(),owner,initial.id,next));setNotice('');
  }
  async function reportFailure(e:unknown){
    if(!mounted.current)return;
    setError(errorMessage(e));
    if(e instanceof SocialError&&(e.status===401||e.code==='account_changed')){
      setAccountRecovery(true);
      try{await refreshAccount();if(mounted.current)setAccountRecovery(false);}
      catch{/* Keep the saved draft and show explicit sign-in/account recovery controls. */}
    }
  }
  async function confirmAccount(){
    if(working.current)return;
    accountFocus.current={opener:document.activeElement,href:location.href};
    working.current=true;setBusy('account');
    try{await refreshAccount();if(mounted.current)setAccountRecovery(false);}
    catch(e){if(mounted.current)setError(errorMessage(e));}
    finally{working.current=false;if(mounted.current)setBusy('');}
  }
  async function like(active:boolean){
    if(working.current||accountLoading||loading||accountRecovery)return;
    if(!owner){location.hash=signInHref(returnTo||'community');return;}
    working.current=true;setBusy('like');setError('');
    try{
      const value=await socialApi<Post>(`${path}/like`,{active,expected_user_id:owner},AbortSignal.timeout(20_000));
      if(value.id!==initial.id||value.viewer.liked!==active||!Number.isSafeInteger(value.counts.likes))throw Error(t('点赞状态尚未确认，请重试。','The like state is unconfirmed. Please retry.'));
      if(mounted.current){setPost(value);setPendingLike(undefined);}
    }catch(e){if(mounted.current)setPendingLike(active);await reportFailure(e);}
    finally{working.current=false;if(mounted.current)setBusy('');}
  }
  async function publish(previous?:PublicationReceipt){
    const submitted={...current.current};
    if(!owner||!writing||working.current||accountLoading||accountRecovery||loading||!detail||!previous&&(!submitted.body.trim()||submitted.parentId&&!detail.replies.some(row=>row.id===submitted.parentId)))return;
    working.current=true;setBusy('reply');setError('');setNotice('');
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
      if(sendsCurrent&&sameReplyDraft(current.current,submitted)){current.current={body:''};setDraft(current.current);setSaved(true);setPreview(false);}
      setNotice(t('回复已发布。','Reply posted.'));
      setPostedId(row.id);
    }catch(e){await reportFailure(e);}
    finally{working.current=false;if(mounted.current)setBusy('');}
  }
  const target=draft.parentId?detail?.replies.find(row=>row.id===draft.parentId):undefined;
  const missingTarget=!!draft.parentId&&!!detail&&!target;
  return <div className="community-card-interaction">
    <div className="post-card-counts card-actions">
      <button ref={toggle} type="button" onClick={expand} aria-expanded={open} aria-controls={regionId} disabled={accountLoading}>
        <MessageCircle size={15}/><span>{post.counts.replies} {t('评论','comments')}</span>{open?<ChevronUp size={13}/>:<ChevronDown size={13}/>}
      </button>
      <button type="button" onClick={()=>void like(!post.viewer.liked)} className={post.viewer.liked?'is-liked':''} aria-pressed={!!post.viewer.liked} disabled={!!busy||loading||accountLoading||accountRecovery||pendingLike!==undefined}>
        <Heart size={15} fill={post.viewer.liked?'currentColor':'none'}/><span>{post.counts.likes} {busy==='like'?t('确认中…','Confirming…'):t('赞','likes')}</span>
      </button>
      <button type="button" onClick={writeComment} aria-controls={regionId} disabled={!!busy||accountLoading||accountRecovery}><PenLine size={15}/>{t('写评论','Write comment')}</button>
      <span className="card-reposts"><Repeat2 size={13}/>{post.counts.reposts} {t('转发','reposts')}</span>
      {showActivity&&post.last_reply_at&&<span className="post-card-activity">{t('最近回复','Last reply')} <DateLabel value={post.last_reply_at}/></span>}
      {children&&<div className="card-actions-extras">{children}</div>}
    </div>
    {error&&<div className="card-action-error"><SocialNotice error={error}/>{accountRecovery&&<div className="card-account-recovery"><p>{t('账号状态尚未确认，草稿已保留。确认账号后可重试原操作。','Your account is unconfirmed. Your draft is kept. Confirm your account before retrying.')}</p><button type="button" className="text-button" disabled={!!busy} onClick={()=>void confirmAccount()}>{t('刷新账号','Refresh account')}</button><a href={signInHref(returnTo||'community')}>{t('重新登录','Sign in again')}</a></div>}{pendingLike!==undefined&&<button ref={retryLike} type="button" className="text-button" disabled={!!busy||loading||accountRecovery} onClick={()=>void like(pendingLike)}>{t('重试原点赞操作','Retry like action')}</button>}</div>}
    {open&&<section id={regionId} ref={replies.root} tabIndex={-1} className="card-comments" aria-label={t('评论预览','Comment preview')}>
      <header><strong>{t('评论','Comments')}</strong><span role="status">{!loading&&!loadError&&detail?t(`最新回复 · ${replies.shown.length} / ${replies.total}`,`Newest replies · ${replies.shown.length} / ${replies.total}`):t('最新回复','Newest replies')}</span><a href={href}>{t('完整讨论','Full discussion')}</a></header>
      <ReplyReadingNotice list={replies}/><ReplyReadingReturn list={replies}/>
      {loading&&<p role="status">{t('正在加载评论…','Loading comments…')}</p>}
      {loadError&&<div><SocialNotice error={loadError}/><button type="button" className="text-button" disabled={!!busy} onClick={()=>void load()}>{t('重试加载评论','Retry loading comments')}</button></div>}
      {!loading&&!loadError&&detail&&<>
        {replies.total?<ol className="card-comment-list">{replies.shown.map(row=>{const parent=row.parent_id?replies.byId.get(row.parent_id):undefined;return <li key={row.id} id={rowIdPrefix+row.id} tabIndex={-1} data-card-reply={row.id} data-discussion-reply={row.id} data-comment-located={replies.revealed===row.id||undefined}>
          <div className="card-comment-meta"><CommunityAuthor user={row.author} compact/>{row.author.id===post.author.id&&<span className="reply-author-badge">{t('楼主','Author')}</span>}<DateLabel value={row.created_at}/></div>
          {row.parent_id&&(parent?<CommentReference parent={parent} disabled={accountLoading} onReveal={trigger=>revealComment(parent.id,trigger)}/>:<span className="card-comment-context unavailable">{t('原评论已删除或暂不可用','The original comment is deleted or unavailable')}</span>)}
          <PostPreview body={row.body} limit={280}/>
          <div className="card-comment-links"><a href={communityPostHref(post.id,returnTo,row.id)}>{t('查看完整回复','Read full reply')}</a>
            {owner&&<button type="button" disabled={!!busy||accountLoading} onClick={()=>{edit({parentId:row.id});setWriting(true);setPreview(false);focusReply();}}>{t('回复','Reply')}</button>}
            <ReplyReadingReturn list={replies} replyId={row.id}/>
          </div>
        </li>;})}</ol>:<p className="card-comments-empty">{t('还没有评论，来聊聊你的想法。','No comments yet. Share your thoughts.')}</p>}
        {replies.total>replies.shown.length&&<button type="button" className="card-more-comments" disabled={accountLoading} onClick={replies.showMore}><ChevronDown size={14}/>{t('再展开','Show')} {Math.min(3,replies.total-replies.shown.length)} {t('条评论','more comments')}</button>}
      </>}
      {accountLoading?<p className="card-account-checking" role="status">{t('正在确认账号，评论草稿已保留…','Checking your account. Your comment draft is kept…')}</p>:owner&&user?<>
      <button ref={composePrompt} type="button" className="card-comment-compose-prompt" hidden={writing} aria-expanded={false} aria-controls={formId} disabled={!!busy||accountRecovery} onClick={writeComment}><PenLine size={16} aria-hidden="true"/><span><b>{publication?t('继续处理待确认评论','Continue pending comment'):hasReplyDraft(draft)?t('继续编辑评论草稿','Continue comment draft'):t('写下你的想法…','Share your thoughts…')}</b><small>{!saved?t('草稿未能保存，请先复制内容。','Draft could not be saved. Copy it first.'):publication||hasReplyDraft(draft)?t('草稿已保留，点击继续','Your draft is kept. Pick up where you left off.'):t('无需离开当前列表','Stay in this feed')}</small></span></button>
      <form id={formId} hidden={!writing} aria-label={t('评论编辑器','Comment editor')} className="social-form card-reply-form" onSubmit={event=>{event.preventDefault();void publish();}}>
        <div className="card-reply-heading"><CommunityAuthor user={post.anonymous&&post.viewer.can_edit?post.author:user} compact/><button type="button" className="text-button" disabled={!!busy} aria-expanded={true} aria-controls={formId} onClick={event=>{minimized.current={opener:event.currentTarget,href:location.href};setWriting(false);}}><ChevronUp size={14} aria-hidden="true"/>{t('收起编辑器','Minimize editor')}</button></div>
        {draft.parentId&&<div className="reply-target"><span>{t('回复','Replying to')} {target?.author.anonymous?t('匿名用户','Anonymous'):target?.author.display_name||t('原评论','original comment')}</span><button type="button" className="icon-button" aria-label={t('取消指定回复','Cancel reply target')} disabled={!!busy} onClick={()=>edit({parentId:undefined})}><X size={14}/></button></div>}
        {missingTarget&&<SocialNotice error={t('原评论已不可用，请取消指定回复后再发布。草稿已保留。','The original comment is unavailable. Cancel the reply target before posting. Your draft is kept.')}/>}
        <MarkdownEditor submitShortcut textareaRef={textarea} id={fieldId} label={t('写评论','Write a comment')} modeLabel={t('评论编辑模式','Comment editor mode')} value={draft.body} onChange={body=>edit({body})} rows={3} maxLength={10000} busy={!!busy} disabled={accountRecovery} preview={preview} onPreviewChange={setPreview} placeholder={t('写下你的想法…','Share your thoughts…')}/>
        <PublicationRecovery pending={publication} payload={replyPublication(draft)} busy={!!busy||loading||accountRecovery} onRetry={()=>void publish(publication?.receipt)}/>
        <PublishingRules action="reply"/>
        <footer><div className="card-reply-feedback"><small role="status">{!saved?t('草稿未能保存，离开前请复制内容。','Draft could not be saved. Copy it before leaving.'):notice|| (hasReplyDraft(draft)?t('草稿已保存，可在完整讨论中继续编辑。','Draft saved. Continue editing in the full discussion.'):t('折叠评论不会丢失草稿。','Collapsing keeps your draft.'))}</small>{postedId&&replies.byId.has(postedId)&&<button type="button" className="text-button card-posted-reply" disabled={loading||accountLoading||!!loadError} onClick={event=>revealComment(postedId,event.currentTarget)}>{t('查看刚发布的评论','View your posted comment')}</button>}</div><button className="button" disabled={!!busy||accountRecovery||loading||!detail||!!loadError||!draft.body.trim()||missingTarget}>{busy==='reply'?t('发布中…','Posting…'):publication?samePublication(publication.receipt,replyPublication(draft))?t('重试发布','Retry publication'):t('作为新内容发布','Publish as new content'):t('发布评论','Post comment')}</button></footer>
      </form></>:<a className="card-comment-signin" href={signInHref(returnTo||'community')}>{t('登录后参与评论','Sign in to comment')}</a>}
      <button type="button" className="card-collapse-comments" onClick={()=>{setOpen(false);replies.clearReadingReturn();toggle.current?.focus();}}><ChevronUp size={14}/>{t('收起评论','Collapse comments')}</button>
    </section>}
  </div>;
}
