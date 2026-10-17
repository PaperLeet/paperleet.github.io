import PostSearchMatch from './PostSearchMatch';
import {sourcePostReadingReturn} from './readingReturn';
import {readProfileRoute,profileFeedHref,type ProfileTab} from './profileNavigation';
import AffiliationBadge from "./AffiliationBadge";
import { resolveAffiliation } from './affiliationNames';
import InterviewMeta from './InterviewMeta';
import {readInterviewFeed} from './interviewModels';
import {InlinePostBody} from './InlineCommentBody';
import {useReadingOrigin,useReadingPosition} from './useReadingPosition';
import ReadingReturnNotice from './ReadingReturnNotice';
import {readInterviewRoute} from './interviewNavigation';
import {withLibraryContext} from './libraryNavigation';
import { PublishingRules } from './PublishingRules';
import { affiliationRequired, canPublishCommunity } from './communityPublishing';
import PublishingAffiliationNotice from './PublishingAffiliationNotice';
import { useEffect, useLayoutEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";
import CommunityMarkdown from "./CommunityMarkdown";
import DiscussionReading from './DiscussionReading';
import SolutionReadingCard from './SolutionReadingCard';
import CommunityFeedFilters from './CommunityFeedFilters';
import CommunityFeedPeriod from './CommunityFeedPeriod';
import {useWritingDisclosure,type WritingPanelProps} from './useWritingDisclosure';
import {useDiscussionReplies,ReplyListControls,MoreReplies,ReplyReadingReturn,ReplyReadingNotice} from './useDiscussionReplies';
import CommunityAuthor,{CommunityAuthorAvatar} from './CommunityAuthor';
import PublishingIdentity from './PublishingIdentity';
import {quoteReplyDraft,captureReplyQuoteSelection,type ReplyQuote,type ReplyQuoteSelection} from './replyQuotes';
import './reply-quotes.css';
import './profile-solutions.css';
import MarkdownEditor from "./MarkdownEditor";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Bookmark,
  BriefcaseBusiness,
  Building2,
  Check,
  ChevronDown,
  Code2,
  Coffee,
  Globe2,
  GraduationCap,
  Heart,
  HelpCircle,
  MessageCircle,
  MessagesSquare,
  Plus,
  RefreshCw,
  Repeat2,
  Quote,
  Settings2,
  Trash2,
  UserPlus,
  Users,
  X,
} from "lucide-react";
import { useI18n } from "./i18n";
import { communityCategories, communityCategoryHint } from './communityCategories';
import ProfileProgress from "./ProfileProgress";
import ProfileAchievements from "./ProfileAchievements";
import { practiceHref, topicHref, topicTitle, type ProblemTopic } from "./communityTopics";
import { signInHref } from "./accountNavigation";
import { COMMUNITY_PAGE_SIZE as PAGE_SIZE, solutionsHref, communityFeedHref, communityPostHref, communityPostReturn, communityReplyTarget, emptyCommunityFeed, readCommunityFeed, type CommunityFeedState } from './communityNavigation';
import { clearPublishedDraft, hasDraft, readPostDraft, tabDraftStorage, writePostDraft, type PostDraft } from "./communityDrafts";
import { clearPublishedReplyDraft, hasReplyDraft, readReplyDraft, sameReplyDraft, writeReplyDraft, type ReplyDraft } from "./replyDrafts";
import {
  Avatar,
  DateLabel,
  PersonLink,
  SocialNotice,
  TestAccountBadge,
  errorMessage,
  profileHref,
  socialApi,
  useAccount,
} from "./social";
import type { PublicUser } from "./social";
import { withConfirmedReply, withConfirmedReplyEdit, type Counts, type Post, type Reply, type PostDetail } from './communityDiscussion';
import { readPublication, preparePublication, clearPublication, postPublication, replyPublication, samePublication, type PublicationReceipt } from './publicationRequests';
import {deferredFeature} from './deferredFeature';
import {communityOutlines,hasCommunityDraftContent,insertCommunityOutline,type CommunityOutlineEdit,type CommunityOutlineId} from './communityWriting';
import './community-writing.css';
import PublicationRecovery from './PublicationRecovery';
import PostEditor from './PostEditor';
import ReplyEditor from './ReplyEditor';
import DiscussionLink from './DiscussionLink';
import PostBookmark from './PostBookmark';
import CommunityCardActions from './CommunityCardActions';
import CommunitySearch from './CommunitySearch';
import Pagination from './CommunityPagination';
import {revealReadingTarget,keepReadingTriggerVisible} from './readingViewport';
import RelatedModuleSearch from './RelatedModuleSearch';
import {searchRelatedModules} from './relatedModuleMatching';
import { canEditReply,type ReplyEditOrigin } from './replyEdits';
import { canEditPost } from './communityEdits';
import PaperLinkField from './PaperLinkField';
import PaperPreviewCard from './PaperPreviewCard';
import { canonicalArxivUrl } from './communityPapers';
const CommunityWritingGuide=deferredFeature(()=>import('./CommunityWritingGuide'),['写作提示','writing prompts'],{
  failure:(props,retry)=><CommunityWritingFailure onRetry={()=>{props.onRestoreFocus();retry();}}/>,
});
function CommunityWritingFailure({onRetry}:{onRetry:()=>void}){
  const {t}=useI18n();
  return <div className="community-writing-failure"><p role="alert">{t('写作提示暂时无法加载，正文仍可继续编辑。','Writing prompts could not load. You can keep editing your post.')}</p><button type="button" onClick={onRetry}>{t('重试写作提示','Retry writing prompts')}</button></div>;
}
type Profile = PublicUser & {
  counts: Counts & { posts: number; solutions?:number; interviews?:number; followers: number; following: number; likes_received: number; reposts_received: number };
  viewer: { following: boolean; is_self: boolean };
};
type Connection = PublicUser & {
  viewer: { following: boolean; is_self: boolean };
};
type Activity = { kind: string; created_at: string; post: Post; reply?: Reply };
type Page<T> = { items: T[]; total: number; limit?: number; offset?: number };
const categoryIcons = { papers: BookOpen, implementation: Code2, questions: HelpCircle, admissions: GraduationCap, career: BriefcaseBusiness, lounge: Coffee };
const CATEGORIES = communityCategories.map(category => ({ ...category, icon: categoryIcons[category.id] }));
const postHref = communityPostHref;
const postPath = (id: string) => `community/posts/${encodeURIComponent(id)}`;
export const COMMUNITY_FEED_TIMEOUT = 20_000;
type FeedFailure = {requestKey:string;kind:'timeout'}|{requestKey:string;kind:'request';message:string};
function TopicContext({ topic, discussion = false, solutions=false,practiceReturnTo }: {
  topic: ProblemTopic;
  discussion?: boolean;
  solutions?:boolean;
  practiceReturnTo?:string;
}) {
  const { t, locale } = useI18n();
  return (
    <div className="problem-discussion-context">
      <div>
        <strong>{topicTitle(topic, locale === "en")}</strong>
        {topic.retired && (
          <small>{t("这道题已下架，历史讨论仍可阅读。", "This module is retired; its discussions remain readable.")}</small>
        )}
      </div>
      <div className="problem-discussion-links">
        {discussion && (
          <a href={solutions?solutionsHref(topic.id):topicHref(topic.id)}>{solutions?t("查看本题题解","All module solutions"):t("查看题目讨论", "All module discussions")}</a>
        )}
        <a href={solutions?topicHref(topic.id):solutionsHref(topic.id)}>{solutions?t("同题讨论","Module discussions"):t("同题题解","Module solutions")}</a>
        {!topic.retired && (
          <a className="button subtle" href={practiceReturnTo?withLibraryContext(topic.id,practiceReturnTo):practiceHref(topic.id)}>
            <Code2 size={15} />{t("返回练习", "Back to practice")}
          </a>
        )}
      </div>
    </div>
  );
}
function Category({ value }: { value: string }) {
  const { t } = useI18n();
  const c = value === "solutions" ? {zh:"题解",en:"Solution"} : CATEGORIES.find((c) => c.id === value);
  return (
    <span className="community-category">{c ? t(c.zh, c.en) : value}</span>
  );
}
function Empty({
  title,
  children,
  icon = <MessagesSquare size={31} />,
}: {
  title: string;
  children?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="community-empty">
      <div className="community-empty-icon">{icon}</div>
      <h3>{title}</h3>
      {children && <div>{children}</div>}
    </div>
  );
}
function Loading() {
  const { t } = useI18n();
  return (
    <div className="social-loading">
      <span className="spinner" />
      {t("正在加载…", "Loading…")}
    </div>
  );
}
function JoinPrompt({ returnTo = "community" }: { returnTo?: string }) {
  const { t } = useI18n();
  const { capabilities } = useAccount();
  const testAuth = capabilities?.auth_mode === "test";
  return (
    <div className="join-prompt">
      <p>
        {testAuth
          ? t(
              "使用管理员提供的内测账号加入讨论。",
              "Join the discussion with your administrator-issued test account.",
            )
          : t(
              "使用已验证的机构邮箱加入讨论。",
              "Join the conversation with a verified institutional email.",
            )}
      </p>
      <a className="button" href={signInHref(returnTo)}>
        {testAuth
          ? t("内测账号登录", "Test account sign-in")
          : t("登录 / 注册", "Sign in / Join")}
        <ArrowRight size={14} />
      </a>
    </div>
  );
}
function PostCard({ post, activity, returnTo:sourceReturnTo, resumed, showActivity, readInPlace=false, onCode,privateProfile=false,savedView=false,onReloadSaved,searchQuery }: { post: Post; activity?: ReactNode; returnTo?: string; resumed?:boolean; showActivity?:boolean; readInPlace?:boolean; onCode?:()=>void;privateProfile?:boolean;savedView?:boolean;onReloadSaved?:()=>void;searchQuery?:string }) {
  const { t,locale } = useI18n();
  const {user,capabilities,loading:accountLoading,refresh:refreshAccount}=useAccount();
  const [bookmarked,setBookmarked]=useState(post.viewer.bookmarked),[bookmarkBusy,setBookmarkBusy]=useState(false);
  const bookmarkAvailable=capabilities?.community_bookmarks===true&&typeof bookmarked==='boolean';
  const returnTo=privateProfile?undefined:sourceReturnTo;
  return (
    <article className="community-post-card" data-reading-post={post.id} data-reading-private-profile={privateProfile?"true":undefined} data-reading-resumed={resumed||undefined}>
      {activity && <div className="activity-label">{activity}</div>}
      <div className="post-card-layout">
        <CommunityAuthorAvatar user={post.author}/>
        <div className="post-card-main">
          <div className="post-card-meta">
            <CommunityAuthor user={post.author} avatar={false}/>
            <span>·</span>
            <DateLabel value={post.created_at} />
            <Category value={post.solution ? "solutions" : post.category} />
          </div>
          {readInPlace&&post.solution ? <SolutionReadingCard id={post.id} title={post.title} body={post.body} returnTo={returnTo} onCode={onCode} query={searchQuery}/> : <><a data-reading-link="title" className="post-title" href={postHref(post.id, returnTo)}>
            <PostSearchMatch text={post.title} query={searchQuery}/>
          </a>
          {post.problem && (
            <a className="post-problem-link" href={post.solution?solutionsHref(post.problem.id):topicHref(post.problem.id)}>
              <Code2 size={13} />{topicTitle(post.problem, locale === "en")}
            </a>
          )}
          {post.interview && <InterviewMeta details={post.interview} compact browseFrom={returnTo}/>}
          <InlinePostBody postId={post.id} body={post.body} limit={190} query={searchQuery}/>
          <PaperPreviewCard url={post.paper_url} paper={post.paper} compact />
          </>}
          <CommunityCardActions post={post} returnTo={returnTo} showActivity={showActivity}>
            <PostBookmark key={`${post.id}:${user?.id||'guest'}`} postId={post.id} owner={user?.id||''} bookmarked={bookmarked}
              enabled={capabilities?.community_bookmarks===true} returnTo={postHref(post.id,returnTo)} disabled={accountLoading} hidden={accountLoading}
              onChange={setBookmarked} onBusyChange={setBookmarkBusy} onAccountRefresh={refreshAccount}/>
          </CommunityCardActions>
          {savedView&&user&&bookmarkAvailable&&bookmarked===false&&!bookmarkBusy&&!accountLoading&&<div className="post-card-unsaved-notice" role="status">
            <span>{t('已取消收藏。可以重新收藏，或刷新列表后移除。','Unsaved. Save it again, or refresh to remove it from this list.')}</span>
            {onReloadSaved&&<button type="button" onClick={onReloadSaved}>{t('刷新收藏列表','Refresh saved posts')}</button>}
          </div>}
        </div>
      </div>
    </article>
  );
}
function Composer({
  category,
  topic,
  owner,
  scope,
  papersAvailable,
  solution=false,
  topics=[],
  returnTo,
  onDraftChange,
  panelProps,
  onClose,
}: {
  category: string;
  topic?: ProblemTopic;
  owner: string;
  scope: string;
  papersAvailable: boolean;
  solution?:boolean;
  topics?:ProblemTopic[];
  returnTo?:string;
  panelProps?:WritingPanelProps;
  onDraftChange: (available: boolean) => void;
  onClose: () => void;
}) {
  const { t, locale } = useI18n();
  const {user,capabilities,loading:accountLoading,refresh:refreshAccount}=useAccount();
  const canPublish=canPublishCommunity(user,capabilities)&&!accountLoading&&(!solution||capabilities?.community_solutions===true);
  const anonymousAvailable=capabilities?.community_anonymous_posts===true;
  const initialCategory = solution ? "implementation" : category || (topic ? "questions" : "papers");
  const [draft, setDraft] = useState<PostDraft>(() => readPostDraft(tabDraftStorage(), owner, scope) || { title: "", body: "", category: initialCategory });
  const currentDraft = useRef(draft);
  const currentActor=useRef(user?.id);currentActor.current=user?.id;
  const writingRoute=location.href;
  const [publication,setPublication] = useState(() => readPublication(tabDraftStorage(),owner,'post',scope));
  const publishing = useRef(false);
  const { title, body, category: board } = draft;
  const [moduleSearch,setModuleSearch]=useState('');
  const moduleHintId=useId(),solutionTopics=topics.filter(row=>!row.retired),matchingTopics=searchRelatedModules(solutionTopics,moduleSearch);
  const selectedTopic=topics.find(row=>row.id===draft.problemId),selectedOutsideSearch=!!draft.problemId&&!matchingTopics.some(row=>row.id===draft.problemId);
  const categoryHint = communityCategoryHint(board), categoryHintId = useId();
  const [preview, setPreview] = useState(false),
    [discardRequested, setDiscardRequested] = useState(false),
    [saved, setSaved] = useState<boolean | null>(() => hasDraft(draft) ? true : null);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  function edit(change: Partial<PostDraft>) {
    const next = { ...currentDraft.current, ...change };
    currentDraft.current = next;
    setDraft(next);
    if(change.body!==undefined){setWritingError('');setWritingNotice('');}
    setSaved(writePostDraft(tabDraftStorage(), owner, scope, next));
    onDraftChange(hasDraft(next));
  }
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const editor=useRef<HTMLTextAreaElement>(null),writingSummary=useRef<HTMLElement>(null),writingHint=useId();
  const [writingOpen,setWritingOpen]=useState(false),[writingError,setWritingError]=useState(''),[writingNotice,setWritingNotice]=useState('');
  const writingEdit=useRef<(CommunityOutlineEdit&{selection:{start:number;end:number;scroll:number}})>(undefined);
  const [writingFocus,setWritingFocus]=useState<{opener:HTMLElement;href:string;source:string;start:number;end:number;scroll?:number}>();
  useLayoutEffect(()=>{
    if(!writingFocus||preview)return;setWritingFocus(undefined);
    if(busy||!mounted.current||currentActor.current!==owner||location.href!==writingFocus.href||currentDraft.current.body!==writingFocus.source)return;
    if(document.activeElement!==writingFocus.opener&&!(document.activeElement===document.body&&(!writingFocus.opener.isConnected||writingFocus.opener.matches('button:disabled'))))return;
    const field=editor.current;if(!field)return;
    field.focus({preventScroll:true});field.scrollIntoView({block:'center'});field.setSelectionRange(writingFocus.start,writingFocus.end);
    if(writingFocus.scroll!==undefined)field.scrollTop=writingFocus.scroll;
  },[writingFocus,preview,busy,owner]);
  function insertOutline(id:CommunityOutlineId,opener:HTMLElement){
    if(publishing.current||!mounted.current||currentActor.current!==owner||writingRoute!==location.href)return;
    const body=currentDraft.current.body,previous=writingEdit.current,en=locale==='en';
    if(previous&&previous.after===body&&previous.id===id&&previous.en===en){setWritingNotice(t('该提纲已在草稿中，可继续填写。','This outline is already in your draft. Keep writing.'));setPreview(false);setWritingFocus({opener,href:location.href,source:body,start:previous.caret,end:previous.caret});return;}
    const field=editor.current,selection={start:field?.selectionStart??body.length,end:field?.selectionEnd??body.length,scroll:field?.scrollTop||0};
    const operation=insertCommunityOutline(body,id,en,selection.end);
    if(!operation.ok){setWritingError(operation.reason==='length'?t('正文接近长度上限，请精简后再插入提纲。','Your post is near its length limit. Shorten it before inserting an outline.'):t('提纲需要独立段落，请将光标移到代码、公式或 HTML 块之外再插入。','The outline needs separate paragraphs. Move the cursor outside code, math or HTML blocks and try again.'));return;}
    writingEdit.current={...operation.edit,selection};edit({body:operation.edit.after});setWritingNotice('');setPreview(false);
    setWritingFocus({opener,href:location.href,source:operation.edit.after,start:operation.edit.caret,end:operation.edit.caret});
  }
  function undoOutline(opener:HTMLElement){
    const operation=writingEdit.current;
    if(publishing.current||!mounted.current||currentActor.current!==owner||writingRoute!==location.href||!operation||operation.after!==currentDraft.current.body)return;
    writingEdit.current=undefined;edit({body:operation.before});setWritingNotice('');setPreview(false);
    setWritingFocus({opener,href:location.href,source:operation.before,...operation.selection});
  }
  const contentReady=hasCommunityDraftContent(body);
  const pendingMatches=!!publication&&samePublication(publication.receipt,postPublication(draft,topic?.id,solution));
  const outlineCategory=(solution?'solution':Object.hasOwn(communityOutlines,board)?board:'questions') as CommunityOutlineId;
  const draftExists = hasDraft(draft);
  const draftMessage = saved === false
    ? t("当前标签页无法保存草稿，离开前请复制内容。", "This tab cannot save your draft. Copy the text before leaving.")
    : draftExists
      ? t("未发布草稿已保存在本标签页，仅当前账号可恢复。", "Unpublished draft saved in this tab, separately for your account.")
      : t("草稿仅保存在本标签页，发布后才会公开。", "Drafts stay in this tab and become public only when published.");
  function discardDraft() {
    writingEdit.current=undefined;setWritingNotice('');
    edit({ title: "", body: "", category: initialCategory, paperUrl: "", anonymous:false });
    setDiscardRequested(false);
    setPreview(false);
  }
  async function publish(previous?:PublicationReceipt) {
    if (busy || publishing.current || !canPublish) return;
    const submitted = { ...currentDraft.current };
    previous ||= publication&&samePublication(publication.receipt,postPublication(submitted,topic?.id,solution))?publication.receipt:undefined;
    if(solution&&!previous&&!topic&&!topics.some(row=>row.id===submitted.problemId&&!row.retired)){setError(t("请选择题解对应的题目。","Choose the practice module for your solution."));return;}
    if(!previous&&!hasCommunityDraftContent(submitted.body)){setError(t('提纲已准备好，请补充自己的问题、观察或思考后再发布。','Your outline is ready. Add your own question, observations or thoughts before publishing.'));return;}
    const paperUrl = previous ? previous.payload.paper_url : (submitted.category === 'papers' ? submitted.paperUrl : undefined);
    if (paperUrl && !papersAvailable) {
      setError(t('论文链接暂时无法保存，请保留草稿并稍后重试。', 'Paper links cannot be saved right now. Keep your draft and try again shortly.'));
      return;
    }
    if (!previous && submitted.category === 'papers' && papersAvailable && !canonicalArxivUrl(submitted.paperUrl)) {
      setError(t('请添加有效的 arXiv 论文链接。', 'Add a valid arXiv paper link.'));
      return;
    }
    const submittedRoute = location.hash;
    if((previous?previous.payload.anonymous===true:submitted.anonymous)&&!anonymousAvailable){setError(t('匿名发布暂不可用，草稿已保留。','Anonymous posting is unavailable. Your draft is kept.'));return;}
    setBusy(true);
    publishing.current = true;
    setError("");
    try {
      const pending = preparePublication(tabDraftStorage(),owner,'post',scope,previous?.payload || postPublication(submitted,topic?.id,solution),previous);
      setPublication(pending);
      const post = await socialApi<Post>(pending.receipt.payload.anonymous===true?"community/posts/anonymous":"community/posts", {
        ...pending.receipt.payload,
        request_id: pending.receipt.requestId,
        expected_user_id: owner,
      });
      if(solution&&(post.solution!==true||post.problem_id!==pending.receipt.payload.problem_id))throw Error(t("暂时无法确认题解的题目关联，草稿已保留，请重试。","Could not confirm the linked solution. Your draft is kept; please retry."));
      clearPublication(tabDraftStorage(),pending.receipt);
      if (samePublication(pending.receipt,postPublication(submitted,topic?.id,solution))) clearPublishedDraft(tabDraftStorage(), owner, scope, submitted);
      if (mounted.current && location.hash === submittedRoute) location.hash = postHref(post.id, returnTo||submittedRoute);
    } catch (e) {
      if (mounted.current) setError(errorMessage(e));
      if (affiliationRequired(e)) void refreshAccount().catch(() => {});
    } finally {
      publishing.current = false;
      if (mounted.current) setBusy(false);
    }
  }
  return (
    <section {...panelProps} className="social-panel community-composer">
      <div className="panel-title">
        <h2>{solution?t("发布题解","Share a solution"):board === 'papers' ? t('发起论文讨论', 'Discuss a paper') : t("发布帖子", "Start a discussion")}</h2>
        <button
          className="icon-button"
          aria-label={t("关闭发帖", "Close composer")}
          title={t("关闭，保留草稿", "Close and keep draft")}
          onClick={onClose}
          disabled={busy}
        >
          <X size={18} />
        </button>
      </div>
      {error && <SocialNotice error={error} />}
      {topic && (
        <a className="post-problem-link composer-problem" href={practiceHref(topic.id)}>
          <Code2 size={13} />{topicTitle(topic, locale === "en")}
        </a>
      )}
      <div className="community-draft-status" role="status">
        <span>{draftMessage}</span>
        {draftExists && (
          <button className="text-button" type="button" disabled={busy} onClick={() => setDiscardRequested(true)}>
            {t("丢弃草稿", "Discard draft")}
          </button>
        )}
      </div>
      {discardRequested && (
        <div className="draft-discard-confirm" role="alert">
          <span>{t("丢弃后无法恢复，确定清空标题、正文和论文链接？", "Discard this draft? Its title, body, and paper link cannot be recovered.")}</span>
          <button type="button" disabled={busy} onClick={discardDraft}>{t("确认丢弃", "Discard")}</button>
          <button type="button" onClick={() => setDiscardRequested(false)}>{t("保留草稿", "Keep draft")}</button>
        </div>
      )}
      <PublishingAffiliationNotice/>
      <PublicationRecovery pending={publication} payload={postPublication(draft,topic?.id,solution)} busy={busy} disabled={!canPublish} onRetry={() => void publish(publication?.receipt)} />
      <form
        className="social-form"
        onSubmit={(e) => {
          e.preventDefault();
          void publish();
        }}
      >
        <PublishingIdentity anonymous={!!draft.anonymous} onChange={anonymous=>edit({anonymous})} disabled={busy}/>
        {solution&&!topic&&<div className="solution-module-selector">
          <RelatedModuleSearch value={moduleSearch} onChange={setModuleSearch} label={t('筛选关联题目','Filter practice modules')} disabled={busy}/>
          <label>{t("关联题目","Practice module")}<select required value={draft.problemId||""} disabled={busy} aria-describedby={moduleHintId} onChange={event=>edit({problemId:event.target.value})}>
            <option value="">{t("选择一道题目","Choose a practice module")}</option>
            {selectedOutsideSearch&&<optgroup label={t('保留当前选择','Current selection')}><option value={draft.problemId} disabled={!selectedTopic||selectedTopic.retired}>{selectedTopic?topicTitle(selectedTopic,locale==='en'):draft.problemId}{(!selectedTopic||selectedTopic.retired)&&t('（暂不可用）',' (unavailable)')}</option></optgroup>}
            {matchingTopics.map(row=><option key={row.id} value={row.id}>{topicTitle(row,locale==="en")}</option>)}
          </select></label>
          <p id={moduleHintId} role="status">{t(`找到 ${matchingTopics.length} 道题目`,`Found ${matchingTopics.length} ${matchingTopics.length===1?'module':'modules'}`)}{selectedOutsideSearch?t(' · 已选题目保留，搜索不会更换关联。',' · Your selection is kept. Searching does not change the linked module.'):!matchingTopics.length?t(' · 试试题号、名称或模块 ID。',' · Try a number, title or module ID.'):''}</p>
        </div>}
        <label>
          {t("版块", "Category")}
          <select value={board} disabled={busy||solution} aria-describedby={categoryHint ? categoryHintId : undefined} onChange={(e) => edit({ category: e.target.value })}>
            {CATEGORIES.map((c) => (
              <option key={c.id} value={c.id}>
                {t(c.zh, c.en)}
              </option>
            ))}
          </select>
          {categoryHint && <small id={categoryHintId}>{t(...categoryHint)}</small>}
        </label>
        {board === 'papers' && <PaperLinkField value={draft.paperUrl || ''} onChange={paperUrl => edit({ paperUrl })} owner={owner} available={papersAvailable} required={papersAvailable} disabled={busy} onUseTitle={paperTitle => edit({ title: paperTitle.slice(0, 160) })} />}
        <label>
          {t("标题", "Title")}
          <input
            data-writing-title=""
            required
            maxLength={160}
            minLength={3}
            value={title}
            disabled={busy}
            placeholder={solution?t("例如：推导思路、关键实现与复杂度分析","For example: derivation, implementation and complexity"):board === 'papers' ? t('这篇论文的哪个观点值得讨论？', 'What would you like to discuss about this paper?') : t("你想讨论什么？", "What would you like to discuss?")}
            onChange={(e) => edit({ title: e.target.value })}
          />
        </label>
        <details className="community-writing-guide" onToggle={event=>setWritingOpen(event.currentTarget.open)}><summary ref={writingSummary}><span><BookOpen size={15}/>{t('用提纲组织内容','Organize your post with prompts')} · {communityOutlines[outlineCategory].name[locale==='en'?'en':'zh']}</span><ChevronDown size={14}/></summary>{writingOpen&&<CommunityWritingGuide category={outlineCategory} outline={communityOutlines[outlineCategory]} hasBody={!!body.trim()} hasContent={contentReady} busy={busy} canUndo={!!writingEdit.current&&writingEdit.current.after===body} editedAfterInsert={!!writingEdit.current&&writingEdit.current.after!==body} onInsert={insertOutline} onUndo={undoOutline} onRestoreFocus={()=>writingSummary.current?.focus({preventScroll:true})}/>}</details>
        {writingError&&<p className="community-writing-hint" role="alert">{writingError}</p>}
        {writingNotice&&<p className="community-writing-hint" role="status">{writingNotice}</p>}
        <MarkdownEditor scopeKey={JSON.stringify(['post',owner,scope])} submitShortcut value={body} onChange={body=>edit({body})} preview={preview} onPreviewChange={setPreview}
          textareaRef={editor} describedBy={body.trim()&&!contentReady&&!pendingMatches?writingHint:undefined} label={t('正文','Post body')} modeLabel={t('帖子编辑模式','Post editor mode')} maxLength={20000} busy={busy}
          placeholder={solution?t('分享解题思路、关键代码、复杂度与边界情况…','Explain your approach, key code, complexity and edge cases…'):board === 'papers' ? t('写下你的问题、观点或复现结果，让大家有话可聊…', 'Share a question, an observation, or a reproduction result to start the discussion…') : t('分享背景、思考或代码片段…','Share context, ideas, or a code snippet…')}/>
        {body.trim()&&!contentReady&&!pendingMatches&&<p id={writingHint} className="community-writing-hint" role="status">{t('提纲已准备好。补充自己的问题、观察或思考后，就可以发布。','Your outline is ready. Add your own question, observations or thoughts to publish.')}</p>}
        <PublishingRules action="post" />
        <div className="form-footer">
          <small>
            {t(
              draft.anonymous?"匿名发布，所属机构仍公开。":"帖子以你的公开昵称发布。",
              draft.anonymous?"Posting anonymously; affiliation remains public.":"Posts are published under your public display name.",
            )}
            {topic && " " + t("仅发布你填写的内容，练习草稿与判题报告不会自动附带。", "Only the text you enter is published; practice drafts and reports are not attached.")}
          </small>
          <button className="button" disabled={busy || !canPublish || solution&&!topic&&!topics.some(row=>row.id===draft.problemId&&!row.retired) || !body.trim() || !contentReady&&!pendingMatches || !!draft.anonymous&&!anonymousAvailable}>
            {busy ? <span className="spinner" /> : <Plus size={15} />}
            {publication ? samePublication(publication.receipt,postPublication(draft,topic?.id,solution)) ? t('重试发布','Retry publication') : t('作为新内容发布','Publish as new content') : t("发布", "Publish")}
          </button>
        </div>
      </form>
    </section>
  );
}
type CommunityProps = {
  route?: string;
  postId?: string;
  topicId?: string;
  topic?: ProblemTopic;
  catalogLoading?: boolean;
  catalogError?: string;
  reloadCatalog?: () => void;
  topics?:ProblemTopic[];
  embedded?:boolean;
  onCode?:()=>void;
};
export default function Community({ postId, route, ...scope }: CommunityProps) {
  const { user } = useAccount();
  return postId ? (
    <Discussion key={`${postId}:${user?.id || "guest"}`} id={postId} returnTo={communityPostReturn(route || '', postId)} replyTarget={communityReplyTarget(route || '', postId)} topics={scope.topics} catalogLoading={scope.catalogLoading} catalogError={scope.catalogError} reloadCatalog={scope.reloadCatalog}/>
  ) : (
    <CommunityFeed key={`${scope.topicId || "all"}:${readCommunityFeed(route||"")?.state.solutions?"solutions":"posts"}:${user?.id || "guest"}`} route={route} {...scope} />
  );
}
export function ProblemSolutions({topic,onCode}:{topic:ProblemTopic;onCode?:()=>void}) {
  return <Community embedded onCode={onCode} topic={topic} topicId={topic.id} route={solutionsHref(topic.id)}/>;
}
function CommunityFeed({ route, topicId, topic, catalogLoading, catalogError, reloadCatalog, topics, embedded=false, onCode }: Omit<CommunityProps, "postId">) {
  const { t, locale } = useI18n();
  const { user, capabilities, loading: accountLoading } = useAccount();
  const solutions=embedded||readCommunityFeed(route||"")?.state.solutions===true;
  const solutionAvailable=capabilities?.community_solutions===true;
  const solutionUnavailable=solutions&&!accountLoading&&!solutionAvailable;
  const [localFilters,setLocalFilters]=useState<CommunityFeedState>(()=>({...emptyCommunityFeed(),solutions:true}));
  const draftScope = (topicId || "general")+(solutions?":solutions":"");
  const [draftAvailable, setDraftAvailable] = useState(() => !!(user && readPostDraft(tabDraftStorage(), user.id, draftScope)));
  const filters = embedded?localFilters:readCommunityFeed(route || (topicId ? topicHref(topicId) : 'community'))?.state || emptyCommunityFeed();
  const { category, following, search, page, sort, days } = filters;
  const saved = filters.saved === true;
  const sortingAvailable = capabilities?.community_feed_sorting === true;
  const sortingUnavailable = !!sort && !accountLoading && !sortingAvailable;
  const periodAvailable=capabilities?.community_feed_period===true;
  const periodUnavailable=!!days&&!accountLoading&&!periodAvailable;
  const savedAvailable = capabilities?.community_bookmarks === true;
  const savedGate = saved && savedAvailable;
  const waitingForSavedAccount = saved && accountLoading;
  const feedAccessReady = (!following || !!user) && (!saved || !!user && savedAvailable && !accountLoading);
  const offset = (page - 1) * PAGE_SIZE;
  const returnTo = communityFeedHref(topicId, filters);
  const requestKey = JSON.stringify([topicId, user?.id, solutions, solutionAvailable, category, following, saved, savedGate, search, page, sort, sortingAvailable, days, periodUnavailable]);
  const boards = useRef<HTMLDivElement>(null),toolbar=useRef<HTMLDivElement>(null),feedList=useRef<HTMLDivElement>(null),resultGeneration=useRef(0);
  const [data, setData] = useState<Page<Post> & { requestKey?: string;generation?:number }>({ items: [], total: 0 }),
    [loading, setLoading] = useState(true),
    [failure, setFailure] = useState<FeedFailure>(),
    [retry, setRetry] = useState(0);
  const writing=useWritingDisclosure(JSON.stringify([user?.id,draftScope]));
  const pending = loading || data.requestKey !== requestKey;
  const error = failure?.requestKey===requestKey ? failure.kind==='timeout'
    ? t('讨论加载超时，请重试。当前搜索与筛选已保留。','Loading discussions timed out. Retry with your current search and filters.')
    : failure.message : '';
  const reading=useReadingPosition(accountLoading||embedded?undefined:user?'account:'+user.id:'guest',returnTo,!pending&&!error&&!accountLoading&&feedAccessReady&&!sortingUnavailable&&!solutionUnavailable&&!periodUnavailable);
  function choose(change: Partial<CommunityFeedState>) {
    const next={...filters,page:1,...change,...(embedded?{solutions:true}:{})};
    if(embedded)setLocalFilters(next);else location.hash = communityFeedHref(topicId,next);
  }
  useEffect(() => {
    const board = boards.current;
    if (!board) return;
    const reveal = () => {
      const selected = board.querySelector('button[aria-pressed="true"]');
      if (!selected || board.scrollWidth <= board.clientWidth) return;
      const frame = board.getBoundingClientRect(), item = selected.getBoundingClientRect();
      const delta = item.left < frame.left ? item.left - frame.left : item.right > frame.right ? item.right - frame.right : 0;
      if (delta) board.scrollBy({ left: delta });
    };
    reveal();
    window.addEventListener('resize', reveal);
    return () => window.removeEventListener('resize', reveal);
  }, [category, locale]);
  useEffect(() => {
    let live = true;
    let redirected = false;
    const generation=++resultGeneration.current;
    const controller = new AbortController();
    setLoading(true);
    setFailure(undefined);
    const params = new URLSearchParams({
      category,
      q: search,
      ...(sort ? {sort} : {}),
      ...(days ? {days} : {}),
      following: following ? "true" : "false",
      ...(saved ? {saved: 'true'} : {}),
      ...(solutions ? {solutions:"true"} : {}),
      offset: String(offset),
      limit: String(PAGE_SIZE),
      ...(topicId ? { problem_id: topicId } : {}),
    });
    if (waitingForSavedAccount || (sort && accountLoading) || (solutions && accountLoading) || (days && accountLoading)) return;
    if(solutionUnavailable){setData({items:[],total:0,requestKey});setLoading(false);return;}
    if (sortingUnavailable) { setData({items:[],total:0,requestKey});setLoading(false);return; }
    if (periodUnavailable) { setData({items:[],total:0,requestKey});setLoading(false);return; }
    if ((saved && !savedGate) || ((following || saved) && !user)) {
      setData({ items: [], total: 0, requestKey });
      setLoading(false);
      return;
    }
    const timer=setTimeout(()=>{
      if(!live)return;
      live=false;controller.abort();
      setFailure({requestKey,kind:'timeout'});setLoading(false);
    },COMMUNITY_FEED_TIMEOUT);
    socialApi<Page<Post> & {sort?:string;solutions?:boolean;days?:string}>(`community/posts?${params}`,undefined,controller.signal)
      .then((data) => {
        if (!live) return;
        if(days&&data?.days!==days)throw Error(t('暂时无法确认发布时间筛选，请重试。','Could not confirm the publication period. Please retry.'));
        if (!data || !Array.isArray(data.items) || !Number.isSafeInteger(data.total) || data.total < 0 || (sort && data.sort !== sort) || (solutions && (data.solutions!==true || data.items.some(post=>post.solution!==true || !post.problem_id || (topicId && post.problem_id!==topicId))))) throw Error(t('暂时无法确认讨论列表的排序，请重试。','Could not confirm the discussion order. Please retry.'));
        const lastPage = Math.max(1, Math.ceil(data.total / PAGE_SIZE));
        if (page > lastPage) {
          redirected = true;
          if(embedded){setLocalFilters(previous=>({...previous,page:lastPage}));return;}
          location.replace(communityFeedHref(topicId, { category, following, ...(solutions?{solutions:true}:{}), ...(saved ? {saved:true} : {}), ...(sort ? {sort} : {}), ...(days?{days}:{}), search, page: lastPage }));
          return;
        }
        setData({ ...data, requestKey,generation });
      })
      .catch((e) => {
        if (live) setFailure({requestKey,kind:'request',message:errorMessage(e)});
      })
      .finally(() => {
        clearTimeout(timer);
        if(!live)return;live=false;
        if (!redirected) setLoading(false);
      });
    return () => {
      live = false; clearTimeout(timer); controller.abort();
    };
  }, [category, following, saved, savedGate, waitingForSavedAccount, search, offset, user?.id, retry, topicId, sort, accountLoading, sortingAvailable, sortingUnavailable, solutions, solutionUnavailable, solutionAvailable, embedded, days, periodUnavailable]);
  const FeedRoot=embedded?"section":"main";
  return (
    <FeedRoot ref={reading.container} className={embedded?"problem-solutions community-page":"social-page community-page"}>
      {!embedded&&topicId && (
        <a className="social-back" href={solutions?solutionsHref():"#/community"}>
          <ArrowLeft size={14} />{solutions?t("所有社区题解","All community solutions"):t("所有社区讨论", "All community discussions")}
        </a>
      )}
      <div className="community-layout">
        {!embedded&&<aside className="community-sidebar">
          <div className="social-page-heading community-heading">
            <h1>{solutions?t("社区题解","Community solutions"):topicId ? t("题目讨论", "Module discussions") : category === 'papers' ? t('论文讨论', 'Paper discussions') : t("社区", "Community")}</h1>
          </div>
          <div ref={boards} className="community-board-scroll">
            <section className="social-panel board-list">
              <span className="sidebar-label">{t("版块", "Boards")}</span>
              <button
                className={!category&&!solutions ? "selected" : ""}
                aria-pressed={!category&&!solutions}
                onClick={() => choose({ category: '',solutions:false })}
              >
                <MessagesSquare size={17} />
                {t("所有讨论", "All discussions")}
              </button>
              <button type="button" className={solutions?"selected":""} aria-pressed={solutions} onClick={()=>choose({solutions:true,category:""})}><Code2 size={17}/>{t("题解","Solutions")}</button>
              {CATEGORIES.map((c) => (
                <button
                  key={c.id}
                  className={!solutions&&category === c.id ? "selected" : ""}
                  aria-pressed={!solutions&&category === c.id}
                  onClick={() => choose({ category: c.id,solutions:false })}
                >
                  <c.icon size={17} />
                  {t(c.zh, c.en)}
                </button>
              ))}
            </section>
          </div>
          {user && (!topicId || topic) ? (
            <button className="button community-compose-button writing-disclosure-trigger" {...writing.triggerProps}>
              <Plus size={16} />
              {draftAvailable ? t("继续草稿", "Resume draft") : solutions?t("发布题解","Share a solution"):category === 'papers' ? t('讨论论文', 'Discuss a paper') : t("发布帖子", "New post")}
            </button>
          ) : !user ? (
            <a className="button community-compose-button" href={signInHref(returnTo)}>
              <Plus size={16} />
              {t("登录发帖", "Sign in to post")}
            </a>
          ) : null}
        </aside>}
        <section className="community-feed">
          {embedded&&<div className="problem-solutions-heading"><div><h2>{t("社区题解","Community solutions")}</h2><p>{t("阅读不同思路，分享你的推导与实现。","Learn from different approaches and share your own.")}</p></div><div>{user?<button type="button" className="button writing-disclosure-trigger" disabled={accountLoading||!solutionAvailable} {...writing.triggerProps}><Plus size={16}/>{draftAvailable?t("继续草稿","Resume draft"):t("发布题解","Share a solution")}</button>:<a className="button" href={signInHref(returnTo)}>{t("登录写题解","Sign in to contribute")}</a>}<a href={returnTo}>{t("全部题解","All solutions")}<ArrowRight size={14}/></a></div></div>}
          <div ref={toolbar} className="community-feed-toolbar" role="group" tabIndex={-1} aria-label={solutions?t('题解浏览控制','Solution browsing controls'):t('讨论浏览控制','Discussion browsing controls')}>
            <div className="social-tabs compact">
              <button
                className={!following && !saved ? "selected" : ""}
                aria-pressed={!following && !saved}
                onClick={() => choose({ following: false, saved: false })}
              >
                {solutions?t("全部题解","All solutions"):t("所有讨论", "All")}
              </button>
              <button
                className={following && !saved ? "selected" : ""}
                aria-pressed={following && !saved}
                onClick={() => choose({ following: true, saved: false })}
              >
                <Users size={14} />
                {t("关注", "Following")}
              </button>
              {savedAvailable&&<button type="button" className={saved?'selected':''} aria-pressed={saved}
                onClick={()=>choose({saved:true,following:false})}>
                <Bookmark size={14}/>{t('我的收藏','Saved')}
              </button>}
            </div>
            <div className="community-feed-controls">
              <CommunitySearch applied={search} onApply={query=>choose({search:query})} label={solutions?t('搜索题解','Search solutions'):undefined} placeholder={solutions?t('搜索题解…','Search solutions…'):undefined}/>
              <CommunityFeedPeriod days={days} available={periodAvailable} pending={accountLoading} onChange={days=>choose({days})}/>
              {sortingAvailable && <label className="community-sort" htmlFor="community-feed-sort"><Settings2 size={14} aria-hidden="true"/><span>{t('排序','Sort')}</span><select id="community-feed-sort" aria-label={t('排序','Sort')} value={sort || 'latest'} onChange={event=>choose({sort:event.target.value==='latest'?undefined:event.target.value as 'active'|'top'})}>
                <option value="latest">{saved?t('最近收藏','Recently saved'):t('最新发布','Newest posts')}</option><option value="active">{t('最近活跃','Recently active')}</option><option value="top">{t('最多赞','Most liked')}</option>
              </select></label>}
            </div>
          </div>
          <CommunityFeedFilters filters={filters} onChange={change=>{
            // Keep keyboard focus on a stable control group when a chip disappears.
            // Synchronous intent cannot override a later route/account/editor action.
            toolbar.current?.focus({preventScroll:true});choose(change);
          }}/>
          {!embedded&&topicId && (
            catalogLoading ? <Loading /> : catalogError ? (
              <div className="feed-error">
                <SocialNotice error={catalogError} />
                <button className="button subtle" onClick={reloadCatalog}>
                  {t("重新加载题目信息", "Reload module details")}
                </button>
              </div>
            ) : <TopicContext topic={topic || { id: topicId, title: topicId, retired: true }} solutions={solutions} />
          )}
          {writing.expanded && user && (
            <Composer key={draftScope} returnTo={returnTo} solution={solutions} topics={topics} category={category} topic={topic} owner={user.id} scope={draftScope} papersAvailable={capabilities?.community_papers === true} onDraftChange={setDraftAvailable} panelProps={writing.panelProps} onClose={writing.close} />
          )}
          <ReadingReturnNotice notice={reading.notice} onRestore={reading.restore}/>
          {days&&!periodUnavailable&&<p className="community-period-note">{t('按帖子发布时间筛选；点赞与回复数为累计数据。','Filtered by publication date; likes and replies are lifetime totals.')}</p>}
          <div ref={feedList} className="social-panel feed-items" data-reading-list="" tabIndex={-1} role="region" aria-label={solutions?t('题解列表','Solution feed'):t('讨论列表','Discussion feed')}>
            {solutionUnavailable ? <Empty title={t("题解服务暂不可用","Community solutions are unavailable")}><p>{t("当前服务尚未提供社区题解，请稍后再试。","This service does not provide community solutions yet. Please try again later.")}</p></Empty> : periodUnavailable ? <Empty title={t("发布时间筛选暂不可用","Publication periods are unavailable")}><p>{t("当前服务尚未提供此筛选。可以切回不限时间，保留其他筛选继续浏览。","This service does not support publication periods yet. Browse all time with your other filters.")}</p><button type="button" className="button subtle" onClick={()=>{toolbar.current?.focus({preventScroll:true});choose({days:undefined});}}>{t("查看不限时间","Browse all time")}</button></Empty> : sortingUnavailable ? <Empty title={t('当前服务暂不支持此排序','This order is currently unavailable')}><p>{t('可切回默认排序继续浏览，原有搜索和版块会保留。','Switch to the default order to keep browsing with your search and board filters.')}</p><button type="button" className="button subtle" onClick={()=>choose({sort:undefined})}>{t('使用默认排序','Use default order')}</button></Empty> : error ? (
              <div className="feed-error">
                <SocialNotice error={error} />
                <button
                  className="button subtle"
                  onClick={() => setRetry((v) => v + 1)}
                >
                  {t("重试", "Retry")}
                </button>
              </div>
            ) : pending ? (
              <Loading />
            ) : saved && !savedAvailable ? (
              <Empty title={t('收藏功能暂不可用','Saved posts are unavailable')} icon={<Bookmark size={31}/>}>
                <p>{t('当前服务尚未提供收藏。你仍可以浏览其他讨论。','This service does not currently provide saved posts. You can still browse discussions.')}</p>
                <button type="button" className="button subtle" onClick={()=>choose({saved:false,following:false})}>{t('查看最新讨论','View latest discussions')}</button>
              </Empty>
            ) : (following || saved) && !user ? (
              <Empty
                title={t(
                  saved?"登录查看你的收藏":"关注你感兴趣的人",
                  saved?"Sign in to see your saved posts":"Follow people you want to learn with",
                )}
                icon={saved?<Bookmark size={31}/>:<Users size={31} />}
              >
                <JoinPrompt returnTo={returnTo} />
              </Empty>
            ) : !data.items.length ? (
              <Empty
                title={
                  search
                    ? t("没有找到相关讨论", "No matching discussions")
                    : saved ? t('还没有收藏的讨论','No saved discussions yet') : following
                      ? t(
                          "关注动态会显示在这里",
                          "Your following feed starts here",
                        )
                      : days?t(`近 ${days} 天还没有${solutions?"题解":"帖子"}`,`No ${solutions?"solutions":"posts"} published in the past ${days} days`): solutions?t("还没有题解，分享第一篇吧","No solutions yet. Share the first approach."):topicId ? t("还没有这道题的讨论", "No discussions for this module yet") : t("这里还没有帖子", "No discussions yet")
                }
              >
                <p>
                  {search
                    ? t("换个关键词试试。", "Try a different search term.")
                    : days ? t('可以扩大发布时间范围继续查找，其他筛选会保留。','Broaden the publication period to keep looking with your other filters.')
                    : saved ? t('在帖子中点击收藏，之后可以在这里继续阅读。','Save a post to return to it here later.') : following
                      ? t(
                          "关注其他研究者，查看他们发布的帖子。",
                          "Follow other members to see their posts here.",
                        )
                      : solutions?t("写下你的解题思路、关键代码和边界情况，帮助大家一起理解。","Share your approach, key code and edge cases to help others learn."):t(
                          topicId ? "可以分享你对公式、实现或测试结果的疑问。" : "分享一篇论文或一个实现问题，开启第一个讨论。",
                          topicId ? "Ask about the formulas, implementation, or test results." : "Share a paper or an implementation question to start the first discussion.",
                        )}
                </p>
                {days&&<button type="button" className="button subtle" onClick={()=>{toolbar.current?.focus({preventScroll:true});choose({days:undefined});}}>{t('查看不限时间','Browse all time')}</button>}
              </Empty>
            ) : (
              data.items.map((post) => <PostCard key={`${requestKey}:${data.generation}:${post.id}`} post={post} returnTo={returnTo} showActivity={sort==='active'} resumed={reading.resumedPost===post.id} searchQuery={search} readInPlace={embedded} onCode={onCode} savedView={saved} onReloadSaved={()=>setRetry(value=>value+1)}/>)
            )}
          </div>
          {!pending && !error && feedAccessReady && !sortingUnavailable && !solutionUnavailable && !periodUnavailable && (
            <Pagination key={requestKey} offset={offset} total={data.total} count={data.items.length} label={solutions?t('题解分页','Solution pages'):t('讨论分页','Discussion pages')}
              onChange={next=>{if(next===offset)return;const target=feedList.current;if(target){target.focus({preventScroll:true});revealReadingTarget(target);}choose({page:next/PAGE_SIZE+1});}}/>
          )}
        </section>
      </div>
    </FeedRoot>
  );
}
function Discussion({ id, returnTo:providedReturnTo, replyTarget,topics,catalogLoading,catalogError,reloadCatalog }: { id: string; returnTo?: string; replyTarget?:string;topics?:ProblemTopic[];catalogLoading?:boolean;catalogError?:string;reloadCatalog?:()=>void }) {
  const { t,locale } = useI18n();
  const { user, capabilities, refresh: refreshAccount, loading:accountLoading } = useAccount();
  const canPublish=canPublishCommunity(user,capabilities)&&!accountLoading;
  const readingOwner=accountLoading?undefined:user?'account:'+user.id:'guest';
  const returnTo=providedReturnTo||sourcePostReadingReturn(readingOwner,id,typeof location==='undefined'?'':location.hash);
  useReadingOrigin(readingOwner,id,returnTo);
  const [editingPost,setEditingPost]=useState(false), [postEdited,setPostEdited]=useState(false);
  const postEditButton=useRef<HTMLButtonElement>(null);
  const [editingReply,setEditingReply]=useState<string>(), [replyEdited,setReplyEdited]=useState<string>();
  const replyEditOrigin=useRef<ReplyEditOrigin>(undefined),replyEditClosing=useRef<{identifier:string;origin:ReplyEditOrigin}>(undefined);
  useLayoutEffect(()=>{
    const intent=replyEditClosing.current;if(editingReply||!intent)return;replyEditClosing.current=undefined;
    const active=document.activeElement,opener=intent.origin.opener;
    if(location.href!==intent.origin.href||active!==opener&&!(active===document.body&&opener instanceof HTMLElement&&(!opener.isConnected||opener.matches(':disabled')||!!opener.closest('[hidden]'))))return;
    const target=document.getElementById(`edit-reply-${intent.identifier}`)||document.getElementById('discussion-replies-heading');
    target?.focus({preventScroll:true});if(target)keepReadingTriggerVisible(target,100);
  },[editingReply]);
  function closeReplyEditing(identifier:string,origin:ReplyEditOrigin={opener:document.activeElement,href:location.href}) {
    replyEditClosing.current={identifier,origin};replyEditOrigin.current=undefined;setEditingReply(undefined);
  }
  function closePostEditing() {
    setEditingPost(false);
    requestAnimationFrame(()=>postEditButton.current?.focus());
  }
  const [post, setPost] = useState<PostDetail>(),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [deleteTarget, setDeleteTarget] = useState<string>();
  const repliesLocked=busy||loading||!!editingReply;
  const replyList=useDiscussionReplies({replies:post?.replies,authorId:post?.author.id,linkedReply:replyTarget,disabled:repliesLocked});
  // Community keys this component by both post and account before it renders.
  const owner = user?.id || "";
  const [draft, setDraft] = useState<ReplyDraft>(() => readReplyDraft(tabDraftStorage(), owner, id) || { body: "" });
  const currentDraft = useRef(draft);
  const [publication,setPublication] = useState(() => readPublication(tabDraftStorage(),owner,'reply',id));
  const publishing = useRef(false);
  const [draftSaved, setDraftSaved] = useState<boolean | null>(() => hasReplyDraft(draft) ? true : null);
  const [discardReply, setDiscardReply] = useState(false);
  const [replyPreview, setReplyPreview] = useState(false);
  const replyField=useRef<HTMLTextAreaElement>(null);
  const quoteEdit=useRef<(ReplyQuote&{selection:ReplyQuoteSelection})>(undefined);
  const [quoteNotice,setQuoteNotice]=useState('');
  const [replyFocus,setReplyFocus]=useState<{href:string;source:string;opener:Element|null;selection?:ReplyQuoteSelection}>();
  useLayoutEffect(()=>{
    if(!replyFocus||replyPreview)return;setReplyFocus(undefined);
    if(replyFocus.href!==location.href||replyFocus.source!==draft.body||!mounted.current||busy)return;
    if(document.activeElement!==replyFocus.opener&&!(document.activeElement===document.body&&!replyFocus.opener?.isConnected))return;
    const field=replyField.current;if(!field)return;
    field.focus({preventScroll:true});field.scrollIntoView({block:'center'});
    field.setSelectionRange(replyFocus.selection?.start??draft.body.length,replyFocus.selection?.end??draft.body.length,replyFocus.selection?.direction||'none');
    field.scrollTop=replyFocus.selection?.scroll??field.scrollHeight;field.scrollLeft=replyFocus.selection?.left??0;
  },[replyFocus,replyPreview,draft.body,busy]);
  const [replyNotice, setReplyNotice] = useState<'' | 'posted' | 'refresh-failed'>('');
  const [postedReplyId, setPostedReplyId] = useState<string>();
  const [replyDraftKept,setReplyDraftKept] = useState(false);
  const readVersion = useRef(0);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  const { body } = draft;
  const target = draft.parentId ? post?.replies.find((reply) => reply.id === draft.parentId) : undefined;
  const authorName=(author:PublicUser|undefined)=>author?.anonymous?t('匿名用户','Anonymous'):author?.display_name;
  const missingTarget = !!draft.parentId && !!post && !target;
  function editReply(change: Partial<ReplyDraft>) {
    const next = { ...currentDraft.current, ...change };
    currentDraft.current = next;
    setDraft(next);
    setDraftSaved(writeReplyDraft(tabDraftStorage(), owner, id, next));
    setReplyNotice("");
    setReplyDraftKept(false);
    setQuoteNotice('');
  }
  function focusReply(selection?:ReplyQuoteSelection){
    if(!user||busy||loading||editingReply)return;
    setReplyPreview(false);setReplyFocus({href:location.href,source:currentDraft.current.body,opener:document.activeElement,selection});
  }
  function quoteReply(row:Reply){
    if(!user||busy||loading||editingReply||!post?.replies.some(reply=>reply.id===row.id))return;
    const previous=quoteEdit.current;
    if(previous?.replyId===row.id&&previous.sourceBody===row.body&&sameReplyDraft(previous.after,currentDraft.current)){setQuoteNotice(t('该引用已在草稿中，可继续编辑。','This excerpt is already in your draft. You can keep editing.'));focusReply();return;}
    const result=quoteReplyDraft(currentDraft.current,id,row,authorName(row.author)||t('用户','User'),locale==='en');
    if(!result.ok){setQuoteNotice(result.reason==='length'?t('引用会超过回复长度上限，现有草稿已保留。请精简后再引用。','This quote would exceed the reply limit. Your draft is kept; shorten it before quoting.'):t('暂时无法生成引用，仍可直接回复。','This reply could not be quoted. You can still reply directly.'));return;}
    const field=replyField.current;
    quoteEdit.current={...result.quote,selection:captureReplyQuoteSelection(field,currentDraft.current.body.length)};
    editReply(result.quote.after);setQuoteNotice(t('引用摘要已加入草稿，可继续编辑后发布。','A source-linked excerpt is in your draft. Edit it before posting.'));focusReply();
  }
  function undoQuote(){
    const operation=quoteEdit.current;
    if(!operation||busy||loading||!sameReplyDraft(operation.after,currentDraft.current))return;
    editReply({body:operation.before.body,parentId:operation.before.parentId});quoteEdit.current=undefined;focusReply(operation.selection);
  }
  async function refresh() {
    if (!mounted.current) return false;
    const version = ++readVersion.current;
    setLoading(true); setError('');
    try {
      const data = await socialApi<PostDetail>(postPath(id));
      if (!mounted.current || version !== readVersion.current) return false;
      setPost(data);
      setReplyNotice(previous => previous === 'refresh-failed' ? 'posted' : previous);
      return true;
    } catch (e) {
      if (mounted.current && version === readVersion.current) setError(errorMessage(e));
      return false;
    } finally {
      if (mounted.current && version === readVersion.current) setLoading(false);
    }
  }
  useEffect(() => {
    void refresh();
    return () => { readVersion.current++; };
  }, [id, user?.id, replyTarget]);
  async function react(kind: "like" | "repost") {
    if (loading || busy) return;
    if (!user) {
      location.hash = signInHref(postHref(id, providedReturnTo, replyTarget));
      return;
    }
    if (!post) return;
    setBusy(true);
    setError("");
    try {
      const data = await socialApi<Post>(`${postPath(id)}/${kind}`, {
        active: kind === "like" ? !post.viewer.liked : !post.viewer.reposted,
        expected_user_id: user.id,
      });
      setPost((prev) => (prev ? { ...data, replies: prev.replies } : prev));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function reply(previous?:PublicationReceipt) {
    if (!user || !canPublish || busy || publishing.current || loading || !previous && (!currentDraft.current.body.trim() || missingTarget)) return;
    const submitted = { ...currentDraft.current };
    const submittedRoute = location.hash;
    setBusy(true);
    publishing.current = true;
    setError("");
    setReplyNotice("");
    try {
      const pending=preparePublication(tabDraftStorage(),owner,'reply',id,previous?.payload || replyPublication(submitted),previous);
      setPublication(pending);
      const published = await socialApi<Reply>(`${postPath(id)}/replies`, {
        ...pending.receipt.payload,
        request_id: pending.receipt.requestId,
        expected_user_id: owner,
      });
      clearPublication(tabDraftStorage(),pending.receipt);
      const sendsCurrent=samePublication(pending.receipt,replyPublication(submitted));
      if(sendsCurrent)clearPublishedReplyDraft(tabDraftStorage(), owner, id, submitted);
      if (!mounted.current || location.hash !== submittedRoute) return;
      setPublication(current=>current?.receipt.requestId===pending.receipt.requestId?undefined:current);
      setPost(previous => withConfirmedReply(previous,published));
      setPostedReplyId(published.id);
      const clearCurrent=sendsCurrent && sameReplyDraft(currentDraft.current, submitted);
      setReplyDraftKept(!clearCurrent);
      if (clearCurrent) {
        currentDraft.current = { body: "" };
        setDraft(currentDraft.current);
        setDraftSaved(null);
        setDiscardReply(false);
        setReplyPreview(false);setQuoteNotice('');quoteEdit.current=undefined;
      }
      setReplyNotice('posted');
      if (!await refresh() && mounted.current) {
        setReplyNotice('refresh-failed');
      }
    } catch (e) {
      if (mounted.current) setError(errorMessage(e));
      if (affiliationRequired(e)) void refreshAccount().catch(() => {});
    } finally {
      publishing.current = false;
      if (mounted.current) setBusy(false);
    }
  }
  async function remove() {
    if (!deleteTarget || busy || loading) return;
    const submittedRoute = location.hash;
    setBusy(true);
    setError("");
    try {
      await socialApi(
        deleteTarget === "post"
          ? `${postPath(id)}/delete`
          : `community/replies/${encodeURIComponent(deleteTarget)}/delete`,
        { expected_user_id: user?.id },
      );
      if (deleteTarget === "post") {
        if (mounted.current && location.hash === submittedRoute)
          location.hash = returnTo || (post?.problem ? post.solution?solutionsHref(post.problem.id):topicHref(post.problem.id) : '#/community');
      }
      else await refresh();
      setDeleteTarget(undefined);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="social-page discussion-page">
      <div className="discussion-navigation"><a className="social-back" href={returnTo || (post?.problem ? post.solution?solutionsHref(post.problem.id):topicHref(post.problem.id) : "#/community")}>
        <ArrowLeft size={14} />
        {returnTo&&readProfileRoute(returnTo)?t('返回个人主页','Back to profile'):returnTo&&readInterviewRoute(returnTo) ? t('返回面经','Back to interviews') : readCommunityFeed(returnTo||"")?.state.solutions?t("返回题解列表","Back to solutions"):returnTo ? t('返回讨论列表', 'Back to discussions') : post?.problem ? post.solution?t("返回本题题解","Back to module solutions"):t("返回题目讨论", "Back to module discussions") : t("返回社区", "Back to community")}
      </a><DiscussionLink postId={id}/><button className="button subtle" type="button" disabled={loading || busy} onClick={() => void refresh()}>
        <RefreshCw size={14} />{loading ? t('正在加载讨论…','Loading discussion…') : error ? t('重试加载讨论','Retry loading discussion') : t('刷新讨论','Refresh discussion')}
      </button></div>
      {error && <SocialNotice error={error} />}
      {loading && !post ? (
        <Loading />
      ) : post ? (
        <>
          {post.problem && <TopicContext topic={post.problem} discussion solutions={post.solution} practiceReturnTo={postHref(id)}/>}
          {postEdited&&<p role="status">{t('帖子修改已保存。','Post changes saved.')}</p>}
          {editingPost&&canEditPost(post,owner)&&<PostEditor post={post} owner={owner} disabled={busy||loading} onRefresh={refresh} onClose={closePostEditing} onSaved={confirmed=>{
            setPost(previous=>previous&&previous.id===confirmed.id?{...confirmed,replies:previous.replies}:previous);
            closePostEditing();setPostEdited(true);
          }}/>}
          <article className="social-panel discussion-post">
            <div className="discussion-heading">
              <Category value={post.solution ? "solutions" : post.category} />
              <h1>{post.title}</h1>
              <div className="discussion-author">
                <CommunityAuthor user={post.author}/>
                <DateLabel value={post.created_at} />
                {post.updated_at!==post.created_at&&<span>{t('已编辑','Edited')} <DateLabel value={post.updated_at}/></span>}
                {canEditPost(post,owner)&&<button ref={postEditButton} type="button" className="text-button" disabled={loading||busy||editingPost} onClick={()=>{setEditingPost(true);setPostEdited(false);}}>{t('编辑帖子','Edit post')}</button>}
                {post.viewer.can_delete && (
                  <button
                    className="icon-button delete-action"
                    aria-label={t("删除帖子", "Delete post")}
                    onClick={() => setDeleteTarget("post")}
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            </div>
            {post.interview && <InterviewMeta details={post.interview} returnTo={postHref(id)} browseFrom={returnTo} topics={topics} catalogLoading={catalogLoading} catalogError={catalogError} onRetryCatalog={reloadCatalog}/>}
            <DiscussionReading body={post.body} postId={post.id} />
            <PaperPreviewCard url={post.paper_url} paper={post.paper} />
            <div className="discussion-actions">
              <button
                className={post.viewer.liked ? "active" : ""}
                disabled={busy || loading}
                aria-pressed={post.viewer.liked}
                onClick={() => void react("like")}
              >
                <Heart
                  size={16}
                  fill={post.viewer.liked ? "currentColor" : "none"}
                />
                {t("点赞", "Like")}
                <span>{post.counts.likes}</span>
              </button>
              <button
                className={post.viewer.reposted ? "active" : ""}
                disabled={busy || loading}
                aria-pressed={post.viewer.reposted}
                onClick={() => void react("repost")}
              >
                <Repeat2 size={16} />
                {post.viewer.reposted
                  ? t("已转发", "Reposted")
                  : t("转发", "Repost")}
                <span>{post.counts.reposts}</span>
              </button>
              <PostBookmark key={`${id}:${owner}`} postId={id} owner={owner} bookmarked={post.viewer.bookmarked}
                enabled={capabilities?.community_bookmarks===true} returnTo={postHref(id,providedReturnTo,replyTarget)}
                disabled={busy||loading||accountLoading||editingPost||!!editingReply} hidden={accountLoading} onBusyChange={setBusy} onAccountRefresh={refreshAccount}
                onChange={bookmarked=>setPost(previous=>previous&&previous.id===id?{...previous,viewer:{...previous.viewer,bookmarked}}:previous)}/>
              <button type="button" onClick={()=>{const heading=document.getElementById('discussion-replies-heading');heading?.focus({preventScroll:true});heading?.scrollIntoView({block:'start'});}}><MessageCircle size={15}/>{post.counts.replies} {t('条回复','replies')}</button>
              {user?<button type="button" className="discussion-compose-action" disabled={repliesLocked} onClick={()=>focusReply()}><MessageCircle size={15}/>{t('写回复','Write a reply')}</button>:<a className="discussion-compose-action" href={signInHref(postHref(id,providedReturnTo,replyTarget))}>{t('登录回复','Sign in to reply')}</a>}
            </div>
          </article>
          <section ref={replyList.root} tabIndex={-1} aria-labelledby="discussion-replies-heading" className="social-panel discussion-replies">
            {replyTarget&&!post.replies.some(row=>row.id===replyTarget)&&<p role="status">{t('链接指向的回复已删除或暂不可用，其他讨论内容仍可阅读。','The linked reply is deleted or unavailable. The rest of the discussion is still readable.')} <a href={postHref(id,providedReturnTo)}>{t('查看完整讨论','View full discussion')}</a></p>}
            <div className="panel-title">
              <h2 id="discussion-replies-heading" tabIndex={-1}>
                {t("回复", "Replies")} <span>{post.counts.replies}</span>
              </h2>
            </div>
            {post.replies.length>0&&<ReplyListControls list={replyList} disabled={repliesLocked}/>}
            <ReplyReadingNotice list={replyList}/><ReplyReadingReturn list={replyList}/>
            {replyList.shown.length ? (
              replyList.shown.map((r) => (
                <article
                  key={r.id}
                  className={`community-reply ${r.parent_id ? "nested-reply" : ""} ${replyTarget===r.id?'linked-reply':''} ${replyList.revealed===r.id?'context-reply':''}`}
                  id={`reply-${r.id}`}
                  data-discussion-reply={r.id}
                  tabIndex={-1}
                >
                  <div className="reply-heading">
                    <CommunityAuthor user={r.author} compact/>
                    {r.author.id===post.author.id&&<span className="reply-author-badge">{t('楼主','Author')}</span>}
                    <DateLabel value={r.created_at} />
                    <DiscussionLink key={r.id} postId={id} replyId={r.id} returnTo={providedReturnTo}/>
                    {r.updated_at&&r.updated_at!==r.created_at&&<span>{t('已编辑','Edited')} <DateLabel value={r.updated_at}/></span>}
                    {canEditReply(r,owner)&&<button id={`edit-reply-${r.id}`} type="button" className="text-button" disabled={busy||loading||!!editingReply} onClick={event=>{replyEditOrigin.current={opener:event.currentTarget,href:location.href};setEditingReply(r.id);setReplyEdited(undefined);}}>{t('编辑回复','Edit reply')}</button>}
                    {r.parent_id && (
                      <button
                        className="reply-context"
                        disabled={repliesLocked||!replyList.byId.has(r.parent_id)}
                        onClick={event=>replyList.reveal(r.parent_id!,event.currentTarget)}
                      >
                        {t("回复", "Replying to")}{" "}
                        {authorName(replyList.byId.get(r.parent_id)?.author) ||
                          t("已删除的回复", "a deleted reply")}
                      </button>
                    )}
                    <ReplyReadingReturn list={replyList} replyId={r.id}/>
                    {r.can_delete && (
                      <button
                        className="icon-button delete-action"
                        aria-label={t("删除回复", "Delete reply")}
                        onClick={() => setDeleteTarget(r.id)}
                      >
                        <Trash2 size={13} />
                      </button>
                    )}
                  </div>
                  <CommunityMarkdown body={r.body} />
                  {replyEdited===r.id&&<p role="status">{t('回复修改已保存。','Reply changes saved.')}</p>}
                  {editingReply===r.id&&canEditReply(r,owner)&&<ReplyEditor reply={r} owner={owner} focusOrigin={replyEditOrigin.current} disabled={busy||loading} onRefresh={refresh} onClose={()=>closeReplyEditing(r.id)} onSaved={(confirmed,confirmation)=>{
                    readVersion.current++;setLoading(false);
                    setPost(previous=>withConfirmedReplyEdit(previous,confirmed,confirmation?.baseVersion||r.edit_version!));
                    if(!confirmation?.keepEditing){closeReplyEditing(r.id,confirmation?.origin);setReplyEdited(r.id);}
                  }}/>}
                  {user && (
                    <div className="reply-respond-actions"><button type="button"
                      className="reply-link"
                      disabled={repliesLocked}
                      onClick={() => {
                        editReply({ parentId: r.id });
                        setReplyPreview(false);
                        focusReply();
                      }}
                    >
                      <MessageCircle size={12} />
                      {t("回复", "Reply")}
                    </button><button type="button" className="reply-link" disabled={repliesLocked} aria-label={t(`引用${authorName(r.author)||'用户'}的回复`,`Quote reply from ${authorName(r.author)||'user'}`)} onClick={()=>quoteReply(r)}><Quote size={12} aria-hidden="true"/>{t('引用','Quote')}</button></div>
                  )}
                </article>
              ))
            ) : replyList.query ? (
              <div className="discussion-reply-empty"><p>{t('没有匹配的回复，可换个关键词或清空搜索。','No replies match. Try another term or clear the search.')}</p><button type="button" className="button subtle" disabled={repliesLocked} onClick={()=>replyList.chooseQuery('')}>{t('清空回复搜索','Clear reply search')}</button></div>
            ) : replyList.authorOnly&&post.replies.length ? (
              <div className="discussion-reply-empty"><p>{t('楼主暂未回复，其他人的讨论仍可查看。','The author has not replied yet. You can still read everyone else’s replies.')}</p><button type="button" className="button subtle" disabled={repliesLocked} onClick={()=>replyList.chooseAuthor(false)}>{t('查看全部回复','Show all replies')}</button></div>
            ) : (
              <p className="no-replies">
                {t(
                  "还没有回复，分享你的想法吧。",
                  "No replies yet. Share your thoughts.",
                )}
              </p>
            )}
            <MoreReplies list={replyList} disabled={repliesLocked}/>
          </section>
          {editingReply&&(!post.replies.some(row=>row.id===editingReply&&canEditReply(row,owner)))&&<div role="status"><p>{t('这条回复已删除或暂不可编辑，修改草稿已保留。','This reply is deleted or cannot be edited now. Your edit draft is kept.')}</p><button type="button" className="text-button" onClick={()=>closeReplyEditing(editingReply)}>{t('关闭编辑','Close editor')}</button></div>}
          {user ? (
            <section className="social-panel reply-composer">
              <div className="panel-title">
                <h3>{t("参与讨论", "Join the discussion")}</h3>
                <CommunityAuthor user={post.anonymous&&post.viewer.can_edit?post.author:user} compact/>
              </div>
              {draft.parentId && (
                <div className="reply-target">
                  <span>
                    {t("回复", "Replying to")} {authorName(target?.author) || t("已删除或暂不可用的回复", "a deleted or unavailable reply")}
                  </span>
                  <button
                    className="icon-button"
                    aria-label={t("取消指定回复", "Cancel reply target")}
                    disabled={busy}
                    onClick={() => editReply({ parentId: undefined })}
                  >
                    <X size={14} />
                  </button>
                </div>
              )}
              {missingTarget && <SocialNotice error={t("原回复已删除或暂不可用。草稿已保留；请取消指定回复，或选择另一条回复后再发布。", "The original reply is deleted or unavailable. Your draft is kept; cancel the reply target or choose another reply before posting.")} />}
              <div className="community-draft-status" role="status">
                <span>{draftSaved === false
                  ? t("当前标签页无法保存回复草稿，离开前请复制内容。", "This tab cannot save your reply draft. Copy the text before leaving.")
                  : hasReplyDraft(draft)
                    ? t("回复草稿已保存在本标签页，仅当前账号和此帖子可恢复。", "Reply draft saved in this tab, separately for your account and this post.")
                    : t("回复草稿仅保存在本标签页，发布后才会公开。", "Reply drafts stay in this tab and become public only when posted.")}</span>
                {hasReplyDraft(draft) && <button className="text-button" type="button" disabled={busy} onClick={() => setDiscardReply(true)}>{t("丢弃回复草稿", "Discard reply draft")}</button>}
              </div>
              {discardReply && <div className="draft-discard-confirm" role="alert">
                <span>{t("丢弃后无法恢复，确定清空回复内容和回复对象？", "Discard this reply draft? Its text and reply target cannot be recovered.")}</span>
                <button type="button" disabled={busy} onClick={() => { editReply({ body: "", parentId: undefined }); setDiscardReply(false); setReplyPreview(false); }}>{t("确认丢弃", "Discard draft")}</button>
                <button type="button" disabled={busy} onClick={() => setDiscardReply(false)}>{t("保留草稿", "Keep draft")}</button>
              </div>}
              {quoteNotice&&<div className="reply-quote-notice" role="status"><span>{quoteNotice}</span>{quoteEdit.current&&sameReplyDraft(quoteEdit.current.after,draft)&&<button type="button" className="text-button" disabled={busy||loading} onClick={undoQuote}>{t('撤销引用','Undo quote')}</button>}</div>}
              <PublishingAffiliationNotice/>
              <PublicationRecovery pending={publication} payload={replyPublication(draft)} busy={busy || loading} disabled={!canPublish} onRetry={() => void reply(publication?.receipt)} />
              {replyNotice && <div className="reply-publication-notice" role="status"><p>{replyNotice === 'posted'
                ? t('回复已发布。','Reply posted.')
                : t('回复已发布并已显示。暂未获取到讨论的最新状态，可重试加载讨论，无需再次发布。','Your reply is posted and visible. The latest discussion could not load; retry loading without posting again.')}</p>
                {replyDraftKept && <p>{t('当前未发布的回复草稿已保留。','Your current unpublished reply draft is kept.')}</p>}
                {postedReplyId && post.replies.some(row => row.id === postedReplyId) && <button className="text-button" type="button" onClick={event=>replyList.reveal(postedReplyId,event.currentTarget)}>{t('查看刚发布的回复','View your posted reply')}</button>}
              </div>}
              <form
                className="social-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  void reply();
                }}
              >
                <MarkdownEditor scopeKey={JSON.stringify(['reply',owner,id])} submitShortcut value={body} onChange={body=>editReply({body})} preview={replyPreview} onPreviewChange={setReplyPreview}
                  textareaRef={replyField} id="reply-body" label={t('回复内容','Your reply')} modeLabel={t('回复编辑模式','Reply editor mode')}
                  rows={5} maxLength={10000} busy={busy} placeholder={t('写下你的回复…（支持 Markdown）','Write a reply… (Markdown supported)')}/>
                <PublishingRules action="reply" />
                <div className="form-footer">
                  <small>
                    {t(
                      "回复会公开展示在你的个人主页。",
                      "Replies appear publicly on your profile.",
                    )}
                  </small>
                  <button className="button" disabled={busy || loading || !canPublish || !body.trim() || missingTarget}>
                    <MessageCircle size={14} />
                    {publication ? samePublication(publication.receipt,replyPublication(draft)) ? t('重试发布','Retry publication') : t('作为新内容发布','Publish as new content') : t("发布回复", "Post reply")}
                  </button>
                </div>
              </form>
            </section>
          ) : (
            <section className="social-panel">
              <JoinPrompt returnTo={postHref(id, providedReturnTo, replyTarget)} />
            </section>
          )}
          {deleteTarget && (
            <div className="delete-confirm" role="alert">
              <span>
                {t(
                  "确认删除？此操作无法撤销。",
                  "Delete this content? This cannot be undone.",
                )}
              </span>
              <button
                className="button subtle"
                disabled={busy}
                onClick={() => setDeleteTarget(undefined)}
              >
                {t("取消", "Cancel")}
              </button>
              <button
                className="button danger"
                disabled={busy}
                onClick={() => void remove()}
              >
                {t("删除", "Delete")}
              </button>
            </div>
          )}
        </>
      ) : !error ? (
        <Empty title={t("帖子不存在", "Post not found")} />
      ) : null}
    </main>
  );
}
export function UserProfile({ username,route=profileFeedHref(username) }: { username: string;route?:string }) {
  const {user}=useAccount();
  return <ProfilePage key={`${username}:${user?.id||'guest'}`} username={username} route={route}/>;
}
function ProfilePage({ username,route }: { username: string;route:string }) {
  const { user } = useAccount();
  // Account changes must remove a previous viewer's private progress immediately.
  return <UserProfileContents key={`${username}:${user?.id || "guest"}`} username={username} route={route}/>;
}
function UserProfileContents({ username,route }: { username: string;route:string }) {
  const { t } = useI18n();
  const { user, capabilities,loading:accountLoading } = useAccount();
  const profileSolutionsAvailable=capabilities?.community_profile_solutions===true;
  const profileInterviewsAvailable=capabilities?.community_profile_interviews===true;
  const [profile, setProfile] = useState<Profile>(),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [activity, setActivity] = useState<Page<Activity> & {generation?:number}>({ items: [], total: 0 }),
    [connections, setConnections] = useState<Page<Connection>>({
      items: [],
      total: 0,
    }),
    [activityLoading, setActivityLoading] = useState(true),
    [loadedActivityKey,setLoadedActivityKey]=useState<string>(),
    [activityError, setActivityError] = useState(""),
    [busy, setBusy] = useState(false),
    [revision, setRevision] = useState(0),
    [activityReveal, setActivityReveal] = useState(0);
  const activityGeneration=useRef(0);
  const requested=readProfileRoute(route),view=requested?.username===username?requested:{username,tab:'posts' as ProfileTab,page:1};
  const waitingForSolutions=view.tab==='solutions'&&accountLoading;
  const waitingForInterviews=view.tab==='interviews'&&accountLoading;
  const unavailableInterviews=view.tab==='interviews'&&!accountLoading&&!profileInterviewsAvailable;
  const unavailableSolutions=view.tab==='solutions'&&!accountLoading&&!profileSolutionsAvailable;
  const tab=unavailableSolutions?'posts':view.tab,page=unavailableSolutions?1:view.page,offset=(page-1)*PAGE_SIZE;
  const returnTo=profileFeedHref(username,{tab,page}),activityKey=JSON.stringify([username,tab,offset,user?.id]);
  function currentProfileHref(){
    const actual=readProfileRoute(location.hash);
    return actual?.username===username?profileFeedHref(username,actual):undefined;
  }
  const activityPending=waitingForInterviews||!unavailableInterviews&&(activityLoading||loadedActivityKey!==activityKey||waitingForSolutions);
  const revealIntent=useRef<{href:string;active:Element|null;top:number}|undefined>(undefined);
  const communityContent = useRef<HTMLElement>(null);
  const path = `community/profiles/${encodeURIComponent(username)}`;
  const connectionTab = tab === "following" || tab === "followers";
  const solutionCount=typeof profile?.counts.solutions==="number"&&Number.isSafeInteger(profile.counts.solutions)&&profile.counts.solutions>=0?profile.counts.solutions:undefined;
  const interviewCount=typeof profile?.counts.interviews==="number"&&Number.isSafeInteger(profile.counts.interviews)&&profile.counts.interviews>=0?profile.counts.interviews:undefined;
  const privateProfileOwner=!!profile?.viewer.is_self&&profile.id===user?.id;
  const displayActivityError=activityError||(tab==='interviews'&&!activityPending&&!privateProfileOwner&&activity.items.some(item=>item.post?.anonymous===true)?t('暂时无法确认面经列表的可见范围，请重试。','Could not confirm the visibility of these experiences. Please retry.'):'');
  const reading=useReadingPosition(accountLoading?undefined:user?'account:'+user.id:'guest',returnTo,!!profile&&!loading&&!activityPending&&!displayActivityError&&!unavailableInterviews&&!connectionTab);
  useEffect(()=>{
    if(unavailableSolutions&&currentProfileHref()===profileFeedHref(username,view))location.replace(returnTo);
  },[unavailableSolutions,route,returnTo,username]);
  useEffect(() => {
    let live = true;
    const controller=new AbortController();
    setLoading(true);
    setError("");
    socialApi<Profile>(path,undefined,controller.signal)
      .then((data) => {
        if (live) setProfile(data);
      })
      .catch((e) => {
        if (live) {
          setProfile(undefined);
          setError(errorMessage(e));
        }
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false; controller.abort();
    };
  }, [username, user?.id, revision]);
  useEffect(() => {
    let live = true;
    const controller=new AbortController();
    let redirected=false;
    setActivityLoading(true);
    setActivityError("");
    if(waitingForSolutions||waitingForInterviews||unavailableInterviews)return;
    const generation=++activityGeneration.current;
    const params = new URLSearchParams({
      offset: String(offset),
      limit: String(PAGE_SIZE),
      ...(connectionTab ? { kind: tab } : { tab }),
    });
    const timer=setTimeout(()=>{
      if(!live)return;live=false;controller.abort();
      setActivity({items:[],total:0});setConnections({items:[],total:0});
      setLoadedActivityKey(activityKey);
      setActivityError(t("社区动态加载超时，请重试。","Loading community activity timed out. Please retry."));setActivityLoading(false);
    },COMMUNITY_FEED_TIMEOUT);
    const promise = connectionTab
      ? socialApi<Page<Connection>>(`${path}/connections?${params}`,undefined,controller.signal).then(
          (data) => {
            if (live){setConnections(data);setLoadedActivityKey(activityKey);}
          },
        )
      : socialApi<Page<Activity> & {solutions?:boolean;interviews?:boolean}>(`${path}/activity?${params}`,undefined,controller.signal).then((data) => {
          if(!live)return;
          if(tab==="solutions"&&(!data||data.solutions!==true||!Array.isArray(data.items)||!Number.isSafeInteger(data.total)||data.total<0||data.items.some(item=>item.kind!=="solutions"||item.post?.solution!==true||!item.post.problem_id)))throw Error(t("暂时无法确认题解列表，请重试。","Could not confirm the solution list. Please retry."));
          if(tab==="interviews"&&(!data||data.interviews!==true||!Array.isArray(data.items)||data.limit!==PAGE_SIZE||data.offset!==offset||data.items.some(item=>!item||item.kind!=="interviews"||item.post?.solution===true)||!readInterviewFeed({items:data.items.map(item=>item.post),total:data.total,limit:PAGE_SIZE,offset})||data.items.length>Math.max(0,data.total-offset)))throw Error(t("暂时无法确认面经列表，请重试。","Could not confirm the interview experiences. Please retry."));
          if(tab==="solutions"||tab==="interviews"){
            const lastOffset=Math.max(0,Math.floor((data.total-1)/PAGE_SIZE)*PAGE_SIZE);
            if(offset>lastOffset){redirected=true;if(currentProfileHref()===returnTo)location.replace(profileFeedHref(username,{tab,page:lastOffset/PAGE_SIZE+1}));return;}
          }
          setActivity({...data,generation});setLoadedActivityKey(activityKey);
        });
    promise
      .catch((e) => {
        if (live) {
          setActivity({ items: [], total: 0 });
          setConnections({ items: [], total: 0 });
          setLoadedActivityKey(activityKey);setActivityError(errorMessage(e));
        }
      })
      .finally(() => {
        clearTimeout(timer);
        if (live&&!redirected) setActivityLoading(false);
      });
    return () => {
      live = false; clearTimeout(timer); controller.abort();
    };
  }, [username, tab, offset, user?.id, revision,waitingForSolutions,waitingForInterviews,unavailableInterviews]);
  async function follow(target: string, active: boolean) {
    if (!user) {
      location.hash = signInHref(returnTo);
      return;
    }
    setBusy(true);
    setError("");
    try {
      const updated = await socialApi<Profile>(
        `community/follow/${encodeURIComponent(target)}`,
        { active, expected_user_id: user.id },
      );
      if (target === username) setProfile(updated);
      setRevision((v) => v + 1);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  const tabs:readonly (readonly [ProfileTab,string])[] = [
    ["posts", t("帖子", "Posts")],
    ...(profileSolutionsAvailable?[["solutions",t("题解","Solutions")]] as const:[]),
    ...(profileInterviewsAvailable||view.tab==='interviews'?[["interviews",t("面经","Experiences")]] as const:[]),
    ["replies", t("回复", "Replies")],
    ["likes", t("点赞", "Likes")],
    ["reposts", t("转发", "Reposts")],
    ["following", t("关注", "Following")],
    ["followers", t("粉丝", "Followers")],
  ] as const;
  const chooseTab = (value: ProfileTab, reveal = false) => {
    if (value !== tab || offset !== 0) setActivityLoading(true);
    const href=profileFeedHref(username,{tab:value,page:1});
    revealIntent.current=reveal?{href,active:document.activeElement,top:window.scrollY}:undefined;
    if(href!==returnTo)location.hash=href;
    if (reveal) setActivityReveal((value) => value + 1);
  };
  useLayoutEffect(() => {
    const pending=revealIntent.current;if(!pending||pending.href!==returnTo)return;
    revealIntent.current=undefined;if(pending.active!==document.activeElement||pending.top!==window.scrollY||currentProfileHref()!==returnTo)return;
    const section = communityContent.current;
    const selected = section?.querySelector<HTMLButtonElement>('[role="tab"][aria-selected="true"]');
    // Commit the selection before moving focus; repeated shortcuts reveal the same tab too.
    selected?.focus({ preventScroll: true });
    selected?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "auto" });
    section?.scrollIntoView({ block: "start", inline: "nearest", behavior: "auto" });
  }, [activityReveal,returnTo]);
  return (
    <main ref={reading.container} className="social-page profile-page">
      <a className="social-back" href="#/community">
        <ArrowLeft size={14} />
        {t("社区", "Community")}
      </a>
      {error && <SocialNotice error={error} />}
      {loading && !profile ? (
        <Loading />
      ) : profile ? (
        <div className="profile-dashboard">
          <aside className="profile-sidebar">
          <section className="social-panel profile-header">
            <div className="profile-header-inner">
              <Avatar user={profile} size={84} />
              <div className="profile-primary">
                <h1>{profile.display_name}</h1>
                <span className="profile-handle">@{profile.username}</span>
                <div className="profile-facts">
                  <TestAccountBadge isTestUser={profile.is_test_user} />
                  {!profile.is_test_user && (resolveAffiliation(profile.affiliation, profile.institution) ? (
                    <AffiliationBadge affiliation={profile.affiliation} institution={profile.institution} />
                  ) : (
                    <span className="profile-affiliation-empty">
                      <Building2 size={13} />
                      {t("机构未认证", "Affiliation not verified")}
                    </span>
                  ))}
                  {profile.country && (
                    <span>
                      <Globe2 size={13} />
                      {profile.country}
                    </span>
                  )}
                  <span>
                    {t("加入于", "Joined")}{" "}
                    <DateLabel value={profile.created_at} />
                  </span>
                </div>
                {profile.bio && <p className="profile-bio">{profile.bio}</p>}
              </div>
              {profile.viewer.is_self ? (
                <a className="button subtle profile-edit" href="#/account">
                  <Settings2 size={14} />
                  {t("编辑资料", "Edit profile")}
                </a>
              ) : (
                <button
                  className={`button ${profile.viewer.following ? "subtle" : ""} profile-edit`}
                  disabled={busy}
                  aria-label={
                    profile.viewer.following
                      ? t("取消关注", "Unfollow")
                      : t("关注", "Follow")
                  }
                  onClick={() =>
                    void follow(username, !profile.viewer.following)
                  }
                >
                  {profile.viewer.following ? (
                    <Check size={14} />
                  ) : (
                    <UserPlus size={14} />
                  )}
                  {profile.viewer.following
                    ? t("已关注", "Following")
                    : t("关注", "Follow")}
                </button>
              )}
              {!profile.viewer.is_self && capabilities?.direct_messages && <a className="button subtle profile-edit" href={user ? `#/messages/${encodeURIComponent(profile.username)}` : signInHref(`#/messages/${encodeURIComponent(profile.username)}`)}>
                <MessageCircle size={14} />{t('私信', 'Message')}
              </a>}
            </div>
            <div className="profile-stats">
              <button type="button" aria-controls="profile-community-panel" onClick={() => chooseTab("following", true)}>
                <b>{profile.counts.following}</b>
                {t("关注", "Following")}
              </button>
              <button type="button" aria-controls="profile-community-panel" onClick={() => chooseTab("followers", true)}>
                <b>{profile.counts.followers}</b>
                {t("粉丝", "Followers")}
              </button>
            </div>
            {profile.viewer.is_self && <a className="text-button profile-workspace-settings" href="#/achievements?from=profile">{t('我的学习勋章', 'My learning achievements')}<ArrowRight size={14} aria-hidden="true" /></a>}
          </section>
          <section className="social-panel profile-community-summary">
            <h2>{t("社区贡献", "Community")}</h2>
            <dl>
              <div><dt><MessagesSquare size={16} />{t("帖子", "Posts")}</dt><dd><button type="button" aria-controls="profile-community-panel" aria-label={t("查看帖子", "View posts")} onClick={() => chooseTab("posts", true)}>{profile.counts.posts}</button></dd></div>
              <div><dt><MessageCircle size={16} />{t("回复", "Replies")}</dt><dd><button type="button" aria-controls="profile-community-panel" aria-label={t("查看回复", "View replies")} onClick={() => chooseTab("replies", true)}>{profile.counts.replies}</button></dd></div>
              <div><dt><Heart size={16} />{t("获赞", "Likes received")}</dt><dd>{profile.counts.likes_received}</dd></div>
              <div><dt><Repeat2 size={16} />{t("获转发", "Reposts received")}</dt><dd>{profile.counts.reposts_received}</dd></div>
              {profileSolutionsAvailable&&<div className="profile-solution-contribution"><dt><Code2 size={16}/>{t("题解","Solutions")}</dt><dd><button type="button" aria-controls="profile-community-panel" aria-label={t("查看题解","View solutions")} onClick={()=>chooseTab("solutions",true)}>{solutionCount??"—"}</button></dd></div>}
              {profileInterviewsAvailable&&<div className="profile-interview-contribution"><dt><BriefcaseBusiness size={16}/>{t("面经","Experiences")}</dt><dd><button type="button" aria-controls="profile-community-panel" aria-label={t("查看面经","View interview experiences")} onClick={()=>chooseTab("interviews",true)}>{interviewCount??"—"}</button></dd></div>}
            </dl>
            {profile.viewer.is_self && <a href="#/community">{t("加入讨论", "Join a discussion")}<ArrowRight size={14} /></a>}
          </section>
          </aside>
          <div className="profile-main">
          <ProfileAchievements username={username} userId={profile.id} isSelf={profile.viewer.is_self} isPublic={profile.show_progress !== false} />
          <ProfileProgress username={username} isSelf={profile.viewer.is_self} />
          <section ref={communityContent} className="profile-community-content" aria-labelledby="profile-community-title">
          <h2 id="profile-community-title">{t("社区动态", "Community activity")}</h2>
          <div className="social-tabs profile-tabs" role="tablist" aria-label={t("社区动态分类", "Community activity categories")}>
            {tabs.map(([key, label]) => (
              <button
                key={key}
                role="tab"
                id={`profile-tab-${key}`}
                aria-selected={tab === key}
                aria-controls="profile-community-panel"
                tabIndex={tab === key ? 0 : -1}
                className={tab === key ? "selected" : ""}
                onKeyDown={(event) => {
                  if(event.defaultPrevented||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.nativeEvent.isComposing||event.nativeEvent.keyCode===229)return;
                  const index = tabs.findIndex(([value]) => value === key);
                  let next = index;
                  if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
                  else if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
                  else if (event.key === "Home") next = 0;
                  else if (event.key === "End") next = tabs.length - 1;
                  else return;
                  event.preventDefault();
                  chooseTab(tabs[next][0]);
                  document.getElementById(`profile-tab-${tabs[next][0]}`)?.focus();
                }}
                onClick={() => chooseTab(key)}
              >
                {label}
                <span>{key==="solutions"?solutionCount??"—":key==="interviews"?interviewCount??"—":profile.counts[key]}</span>
              </button>
            ))}
          </div>
          <ReadingReturnNotice notice={reading.notice} onRestore={reading.restore}/>
          <section data-reading-list="" tabIndex={-1} className="social-panel profile-activity" id="profile-community-panel" role="tabpanel" aria-labelledby={`profile-tab-${tab}`} aria-busy={activityPending}>
            {tab==="solutions"&&profile.viewer.is_self&&<p className="profile-solutions-note">{t("匿名题解仅在你自己的主页可见。","Anonymous solutions appear here only when you view your own profile.")}</p>}
            {tab==="interviews"&&privateProfileOwner&&<p className="profile-interviews-note">{t("匿名面经仅在你自己的主页可见。","Anonymous experiences appear here only when you view your own profile.")}</p>}
            {unavailableInterviews ? <Empty title={t("面经分类暂不可用","Interview experiences are unavailable")}><p>{t("当前服务尚未提供个人面经分类，可以先查看全部帖子。","This service does not provide profile experiences yet. You can view all posts instead.")}</p><button type="button" className="button subtle" onClick={()=>chooseTab("posts")}>{t("查看全部帖子","View all posts")}</button></Empty> : activityPending ? (
              <Loading />
            ) : displayActivityError ? (
              <div className="feed-error">
                <SocialNotice error={displayActivityError} />
                <button
                  className="button subtle"
                  onClick={() => setRevision((v) => v + 1)}
                >
                  {t("重试", "Retry")}
                </button>
              </div>
            ) : connectionTab ? (
              connections.items.length ? (
                connections.items.map((member) => (
                  <div className="connection-card" key={member.id}>
                    <a href={profileHref(member)}>
                      <Avatar user={member} size={45} />
                    </a>
                    <div>
                      <a href={profileHref(member)}>
                        <b>{member.display_name}</b>
                        <span>@{member.username}</span>
                      </a>
                      <TestAccountBadge isTestUser={member.is_test_user} />
                      <AffiliationBadge affiliation={member.affiliation} institution={member.institution} />
                      {member.bio && <p>{member.bio}</p>}
                    </div>
                    {!member.viewer.is_self && (
                      <button
                        className={`button ${member.viewer.following ? "subtle" : ""}`}
                        disabled={busy}
                        aria-label={
                          member.viewer.following
                            ? t("取消关注", "Unfollow")
                            : t("关注", "Follow")
                        }
                        onClick={() =>
                          void follow(member.username, !member.viewer.following)
                        }
                      >
                        {member.viewer.following
                          ? t("已关注", "Following")
                          : t("关注", "Follow")}
                      </button>
                    )}
                  </div>
                ))
              ) : (
                <Empty
                  title={
                    tab === "following"
                      ? t("还没有关注的人", "No following yet")
                      : t("还没有粉丝", "No followers yet")
                  }
                  icon={<Users size={30} />}
                />
              )
            ) : activity.items.length ? (
              activity.items.map((item, index) => (
                <div
                  key={`${activityKey}:${activity.generation}:${item.kind}-${item.reply?.id || item.post.id}-${index}`}
                >
                  {item.reply ? (
                    <article className="profile-reply-card" data-reading-post={item.post.id} data-reading-reply={item.reply.id} data-reading-private-profile={profile.viewer.is_self&&item.post.anonymous===true?"true":undefined} data-reading-resumed={reading.resumedReply===item.reply.id||undefined}>
                      <div className="activity-label">
                        <MessageCircle size={13} />
                        {t("回复了", "Replied to")}
                        <a data-reading-link="title" href={postHref(item.post.id,profile.viewer.is_self&&item.post.anonymous===true?undefined:returnTo,item.reply.id)}>{item.post.title}</a>
                        <DateLabel value={item.created_at} />
                      </div>
                      <CommunityMarkdown body={item.reply.body} />
                    </article>
                  ) : (
                    <PostCard
                      post={item.post}
                      returnTo={returnTo}
                      privateProfile={profile.viewer.is_self&&item.post.anonymous===true}
                      resumed={reading.resumedPost===item.post.id}
                      readInPlace={tab==="solutions"}
                      activity={
                        tab === "likes" ? (
                          <>
                            <Heart size={13} />
                            {t("点赞了这个帖子", "Liked this post")}
                            <DateLabel value={item.created_at} />
                          </>
                        ) : tab === "reposts" ? (
                          <>
                            <Repeat2 size={13} />
                            {t("转发了这个帖子", "Reposted this post")}
                            <DateLabel value={item.created_at} />
                          </>
                        ) : undefined
                      }
                    />
                  )}
                </div>
              ))
            ) : (
              <Empty title={tab==="solutions"?t("还没有题解","No solutions yet"):tab==="interviews"?t("还没有面经","No interview experiences yet"):t("还没有相关动态", "No activity here yet")}>
                <p>{tab==="solutions"?t("发布的题解会按题目关联展示在这里。","Published solutions appear here with their practice modules."):tab==="interviews"?t("发布的面试复盘会汇总在这里，方便继续阅读与交流。","Published interview reflections appear here to revisit and discuss."):t("发布、回复或互动后，记录会显示在这里。","Posts and interactions will appear here as they happen.")}</p>
                {tab==="solutions"&&profile.viewer.is_self&&<a className="button subtle" href={solutionsHref()}>{t("去写一篇题解","Share a solution")}</a>}
                {tab==="interviews"&&privateProfileOwner&&<a className="button subtle" href="#/interviews">{t("前往面经广场","Explore experiences")}</a>}
              </Empty>
            )}
          </section>
          {!unavailableInterviews&&!activityPending&&!displayActivityError&&<Pagination key={activityKey} offset={offset} total={connectionTab?connections.total:activity.total} count={connectionTab?connections.items.length:activity.items.length} label={t('个人主页列表分页','Profile list pages')}
            onChange={next=>{if(next===offset)return;revealIntent.current=undefined;const target=document.getElementById('profile-community-panel');if(target){target.focus({preventScroll:true});revealReadingTarget(target);}setActivityLoading(true);location.hash=profileFeedHref(username,{tab,page:next/PAGE_SIZE+1});}}/>}
          </section>
          </div>
        </div>
      ) : null}
    </main>
  );
}
