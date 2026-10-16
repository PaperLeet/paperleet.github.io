import AffiliationBadge from "./AffiliationBadge";
import { resolveAffiliation } from './affiliationNames';
import InterviewMeta from './InterviewMeta';
import {InlinePostBody} from './InlineCommentBody';
import {useReadingOrigin,useReadingPosition} from './useReadingPosition';
import ReadingReturnNotice from './ReadingReturnNotice';
import {readInterviewRoute} from './interviewNavigation';
import { PublishingRules } from './PublishingRules';
import { affiliationRequired, canPublishCommunity } from './communityPublishing';
import PublishingAffiliationNotice from './PublishingAffiliationNotice';
import { useEffect, useLayoutEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";
import CommunityMarkdown from "./CommunityMarkdown";
import DiscussionReading from './DiscussionReading';
import {useDiscussionReplies,ReplyListControls,MoreReplies,ReplyReadingReturn,ReplyReadingNotice} from './useDiscussionReplies';
import CommunityAuthor,{CommunityAuthorAvatar} from './CommunityAuthor';
import PublishingIdentity from './PublishingIdentity';
import {quoteReplyDraft,type ReplyQuote} from './replyQuotes';
import './reply-quotes.css';
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
  ChevronLeft,
  ChevronRight,
  Code2,
  Coffee,
  Globe2,
  Heart,
  HelpCircle,
  MessageCircle,
  MessagesSquare,
  Plus,
  RefreshCw,
  Repeat2,
  Quote,
  Search,
  Settings2,
  Trash2,
  UserPlus,
  Users,
  X,
} from "lucide-react";
import { useI18n } from "./i18n";
import ProfileProgress from "./ProfileProgress";
import ProfileAchievements from "./ProfileAchievements";
import { practiceHref, topicHref, topicTitle, type ProblemTopic } from "./communityTopics";
import { signInHref } from "./accountNavigation";
import { COMMUNITY_PAGE_SIZE as PAGE_SIZE, communityFeedHref, communityPostHref, communityPostReturn, communityReplyTarget, emptyCommunityFeed, readCommunityFeed, type CommunityFeedState } from './communityNavigation';
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
import { withConfirmedReply, type Counts, type Post, type Reply, type PostDetail } from './communityDiscussion';
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
import { canEditReply } from './replyEdits';
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
  counts: Counts & { posts: number; followers: number; following: number; likes_received: number; reposts_received: number };
  viewer: { following: boolean; is_self: boolean };
};
type Connection = PublicUser & {
  viewer: { following: boolean; is_self: boolean };
};
type Activity = { kind: string; created_at: string; post: Post; reply?: Reply };
type Page<T> = { items: T[]; total: number; limit?: number; offset?: number };
const CATEGORIES = [
  { id: "papers", zh: "论文讨论", en: "Papers", icon: BookOpen },
  { id: "implementation", zh: "代码实现", en: "Implementation", icon: Code2 },
  { id: "questions", zh: "问题求助", en: "Questions", icon: HelpCircle },
  {
    id: "career",
    zh: "学术与职业",
    en: "Research & careers",
    icon: BriefcaseBusiness,
  },
  { id: "lounge", zh: "闲聊", en: "Lounge", icon: Coffee },
];
const postHref = communityPostHref;
const postPath = (id: string) => `community/posts/${encodeURIComponent(id)}`;
function TopicContext({ topic, discussion = false }: {
  topic: ProblemTopic;
  discussion?: boolean;
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
          <a href={topicHref(topic.id)}>{t("查看题目讨论", "All module discussions")}</a>
        )}
        {!topic.retired && (
          <a className="button subtle" href={practiceHref(topic.id)}>
            <Code2 size={15} />{t("返回练习", "Back to practice")}
          </a>
        )}
      </div>
    </div>
  );
}
function Category({ value }: { value: string }) {
  const { t } = useI18n();
  const c = CATEGORIES.find((c) => c.id === value);
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
function Pagination({
  offset,
  total,
  onChange,
}: {
  offset: number;
  total: number;
  onChange: (offset: number) => void;
}) {
  const { t } = useI18n();
  return total > PAGE_SIZE ? (
    <div className="community-pagination">
      <span>
        {offset + 1}–{Math.min(total, offset + PAGE_SIZE)} / {total}
      </span>
      <button
        disabled={!offset}
        aria-label={t("上一页", "Previous page")}
        onClick={() => onChange(Math.max(0, offset - PAGE_SIZE))}
      >
        <ChevronLeft size={17} />
      </button>
      <button
        disabled={offset + PAGE_SIZE >= total}
        aria-label={t("下一页", "Next page")}
        onClick={() => onChange(offset + PAGE_SIZE)}
      >
        <ChevronRight size={17} />
      </button>
    </div>
  ) : null;
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
function PostCard({ post, activity, returnTo, resumed, showActivity }: { post: Post; activity?: ReactNode; returnTo?: string; resumed?:boolean; showActivity?:boolean }) {
  const { locale } = useI18n();
  return (
    <article className="community-post-card" data-reading-post={post.id} data-reading-resumed={resumed||undefined}>
      {activity && <div className="activity-label">{activity}</div>}
      <div className="post-card-layout">
        <CommunityAuthorAvatar user={post.author}/>
        <div className="post-card-main">
          <div className="post-card-meta">
            <CommunityAuthor user={post.author} avatar={false}/>
            <span>·</span>
            <DateLabel value={post.created_at} />
            <Category value={post.category} />
          </div>
          <a data-reading-link="title" className="post-title" href={postHref(post.id, returnTo)}>
            {post.title}
          </a>
          {post.problem && (
            <a className="post-problem-link" href={topicHref(post.problem.id)}>
              <Code2 size={13} />{topicTitle(post.problem, locale === "en")}
            </a>
          )}
          {post.interview && <InterviewMeta details={post.interview} compact/>}
          <InlinePostBody body={post.body} limit={190}/>
          <PaperPreviewCard url={post.paper_url} paper={post.paper} compact />
          <CommunityCardActions post={post} returnTo={returnTo} showActivity={showActivity}/>
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
  onDraftChange,
  onClose,
}: {
  category: string;
  topic?: ProblemTopic;
  owner: string;
  scope: string;
  papersAvailable: boolean;
  onDraftChange: (available: boolean) => void;
  onClose: () => void;
}) {
  const { t, locale } = useI18n();
  const {user,capabilities,loading:accountLoading,refresh:refreshAccount}=useAccount();
  const canPublish=canPublishCommunity(user,capabilities)&&!accountLoading;
  const anonymousAvailable=capabilities?.community_anonymous_posts===true;
  const initialCategory = category || (topic ? "questions" : "papers");
  const [draft, setDraft] = useState<PostDraft>(() => readPostDraft(tabDraftStorage(), owner, scope) || { title: "", body: "", category: initialCategory });
  const currentDraft = useRef(draft);
  const currentActor=useRef(user?.id);currentActor.current=user?.id;
  const writingRoute=location.href;
  const [publication,setPublication] = useState(() => readPublication(tabDraftStorage(),owner,'post',scope));
  const publishing = useRef(false);
  const { title, body, category: board } = draft;
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
  const pendingMatches=!!publication&&samePublication(publication.receipt,postPublication(draft,topic?.id));
  const outlineCategory=(Object.hasOwn(communityOutlines,board)?board:'questions') as CommunityOutlineId;
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
    previous ||= publication&&samePublication(publication.receipt,postPublication(submitted,topic?.id))?publication.receipt:undefined;
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
      const pending = preparePublication(tabDraftStorage(),owner,'post',scope,previous?.payload || postPublication(submitted,topic?.id),previous);
      setPublication(pending);
      const post = await socialApi<Post>(pending.receipt.payload.anonymous===true?"community/posts/anonymous":"community/posts", {
        ...pending.receipt.payload,
        request_id: pending.receipt.requestId,
        expected_user_id: owner,
      });
      clearPublication(tabDraftStorage(),pending.receipt);
      if (samePublication(pending.receipt,postPublication(submitted,topic?.id))) clearPublishedDraft(tabDraftStorage(), owner, scope, submitted);
      if (mounted.current && location.hash === submittedRoute) location.hash = postHref(post.id, submittedRoute);
    } catch (e) {
      if (mounted.current) setError(errorMessage(e));
      if (affiliationRequired(e)) void refreshAccount().catch(() => {});
    } finally {
      publishing.current = false;
      if (mounted.current) setBusy(false);
    }
  }
  return (
    <section className="social-panel community-composer">
      <div className="panel-title">
        <h2>{board === 'papers' ? t('发起论文讨论', 'Discuss a paper') : t("发布帖子", "Start a discussion")}</h2>
        <button
          className="icon-button"
          aria-label={t("关闭发帖", "Close composer")}
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
      <PublicationRecovery pending={publication} payload={postPublication(draft,topic?.id)} busy={busy} disabled={!canPublish} onRetry={() => void publish(publication?.receipt)} />
      <form
        className="social-form"
        onSubmit={(e) => {
          e.preventDefault();
          void publish();
        }}
      >
        <PublishingIdentity anonymous={!!draft.anonymous} onChange={anonymous=>edit({anonymous})} disabled={busy}/>
        <label>
          {t("版块", "Category")}
          <select value={board} disabled={busy} onChange={(e) => edit({ category: e.target.value })}>
            {CATEGORIES.map((c) => (
              <option key={c.id} value={c.id}>
                {t(c.zh, c.en)}
              </option>
            ))}
          </select>
        </label>
        {board === 'papers' && <PaperLinkField value={draft.paperUrl || ''} onChange={paperUrl => edit({ paperUrl })} owner={owner} available={papersAvailable} required={papersAvailable} disabled={busy} onUseTitle={paperTitle => edit({ title: paperTitle.slice(0, 160) })} />}
        <label>
          {t("标题", "Title")}
          <input
            required
            maxLength={160}
            minLength={3}
            value={title}
            disabled={busy}
            placeholder={board === 'papers' ? t('这篇论文的哪个观点值得讨论？', 'What would you like to discuss about this paper?') : t("你想讨论什么？", "What would you like to discuss?")}
            onChange={(e) => edit({ title: e.target.value })}
          />
        </label>
        <details className="community-writing-guide" onToggle={event=>setWritingOpen(event.currentTarget.open)}><summary ref={writingSummary}><span><BookOpen size={15}/>{t('用提纲组织内容','Organize your post with prompts')} · {communityOutlines[outlineCategory].name[locale==='en'?'en':'zh']}</span><ChevronDown size={14}/></summary>{writingOpen&&<CommunityWritingGuide category={outlineCategory} outline={communityOutlines[outlineCategory]} hasBody={!!body.trim()} hasContent={contentReady} busy={busy} canUndo={!!writingEdit.current&&writingEdit.current.after===body} editedAfterInsert={!!writingEdit.current&&writingEdit.current.after!==body} onInsert={insertOutline} onUndo={undoOutline} onRestoreFocus={()=>writingSummary.current?.focus({preventScroll:true})}/>}</details>
        {writingError&&<p className="community-writing-hint" role="alert">{writingError}</p>}
        {writingNotice&&<p className="community-writing-hint" role="status">{writingNotice}</p>}
        <MarkdownEditor submitShortcut value={body} onChange={body=>edit({body})} preview={preview} onPreviewChange={setPreview}
          textareaRef={editor} describedBy={body.trim()&&!contentReady&&!pendingMatches?writingHint:undefined} label={t('正文','Post body')} modeLabel={t('帖子编辑模式','Post editor mode')} maxLength={20000} busy={busy}
          placeholder={board === 'papers' ? t('写下你的问题、观点或复现结果，让大家有话可聊…', 'Share a question, an observation, or a reproduction result to start the discussion…') : t('分享背景、思考或代码片段…','Share context, ideas, or a code snippet…')}/>
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
          <button className="button" disabled={busy || !canPublish || !body.trim() || !contentReady&&!pendingMatches || !!draft.anonymous&&!anonymousAvailable}>
            {busy ? <span className="spinner" /> : <Plus size={15} />}
            {publication ? samePublication(publication.receipt,postPublication(draft,topic?.id)) ? t('重试发布','Retry publication') : t('作为新内容发布','Publish as new content') : t("发布", "Publish")}
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
};
export default function Community({ postId, route, ...scope }: CommunityProps) {
  const { user } = useAccount();
  return postId ? (
    <Discussion key={`${postId}:${user?.id || "guest"}`} id={postId} returnTo={communityPostReturn(route || '', postId)} replyTarget={communityReplyTarget(route || '', postId)} />
  ) : (
    <CommunityFeed key={`${scope.topicId || "all"}:${user?.id || "guest"}`} route={route} {...scope} />
  );
}
function CommunityFeed({ route, topicId, topic, catalogLoading, catalogError, reloadCatalog }: Omit<CommunityProps, "postId">) {
  const { t, locale } = useI18n();
  const { user, capabilities, loading: accountLoading } = useAccount();
  const draftScope = topicId || "general";
  const [draftAvailable, setDraftAvailable] = useState(() => !!(user && readPostDraft(tabDraftStorage(), user.id, draftScope)));
  const filters = readCommunityFeed(route || (topicId ? topicHref(topicId) : 'community'))?.state || emptyCommunityFeed();
  const { category, following, search, page, sort } = filters;
  const saved = filters.saved === true;
  const sortingAvailable = capabilities?.community_feed_sorting === true;
  const sortingUnavailable = !!sort && !accountLoading && !sortingAvailable;
  const savedAvailable = capabilities?.community_bookmarks === true;
  const savedGate = saved && savedAvailable;
  const waitingForSavedAccount = saved && accountLoading;
  const offset = (page - 1) * PAGE_SIZE;
  const returnTo = communityFeedHref(topicId, filters);
  const requestKey = JSON.stringify([topicId, user?.id, category, following, saved, savedGate, search, page, sort, sortingAvailable]);
  const [query, setQuery] = useState(search);
  const boards = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<Page<Post> & { requestKey?: string }>({ items: [], total: 0 }),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [compose, setCompose] = useState(false),
    [retry, setRetry] = useState(0);
  const pending = loading || data.requestKey !== requestKey;
  const reading=useReadingPosition(accountLoading?undefined:user?'account:'+user.id:'guest',returnTo,!pending&&!error&&!accountLoading&&!((following||saved)&&!user)&&!(saved&&!savedAvailable)&&!sortingUnavailable);
  function choose(change: Partial<CommunityFeedState>) {
    location.hash = communityFeedHref(topicId, { ...filters, page: 1, ...change });
  }
  useEffect(() => { setQuery(search); }, [search]);
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
    const controller = new AbortController();
    setLoading(true);
    setError("");
    const params = new URLSearchParams({
      category,
      q: search,
      ...(sort ? {sort} : {}),
      following: following ? "true" : "false",
      ...(saved ? {saved: 'true'} : {}),
      offset: String(offset),
      limit: String(PAGE_SIZE),
      ...(topicId ? { problem_id: topicId } : {}),
    });
    if (waitingForSavedAccount || (sort && accountLoading)) return;
    if (sortingUnavailable) { setData({items:[],total:0,requestKey});setLoading(false);return; }
    if ((saved && !savedGate) || ((following || saved) && !user)) {
      setData({ items: [], total: 0, requestKey });
      setLoading(false);
      return;
    }
    socialApi<Page<Post> & {sort?:string}>(`community/posts?${params}`,undefined,controller.signal)
      .then((data) => {
        if (!live) return;
        if (!data || !Array.isArray(data.items) || !Number.isSafeInteger(data.total) || data.total < 0 || (sort && data.sort !== sort)) throw Error(t('暂时无法确认讨论列表的排序，请重试。','Could not confirm the discussion order. Please retry.'));
        const lastPage = Math.max(1, Math.ceil(data.total / PAGE_SIZE));
        if (page > lastPage) {
          redirected = true;
          location.replace(communityFeedHref(topicId, { category, following, ...(saved ? {saved:true} : {}), ...(sort ? {sort} : {}), search, page: lastPage }));
          return;
        }
        setData({ ...data, requestKey });
      })
      .catch((e) => {
        if (live) setError(errorMessage(e));
      })
      .finally(() => {
        if (live && !redirected) setLoading(false);
      });
    return () => {
      live = false; controller.abort();
    };
  }, [category, following, saved, savedGate, waitingForSavedAccount, search, offset, user?.id, retry, topicId, sort, accountLoading, sortingAvailable, sortingUnavailable]);
  return (
    <main ref={reading.container} className="social-page community-page">
      {topicId && (
        <a className="social-back" href="#/community">
          <ArrowLeft size={14} />{t("所有社区讨论", "All community discussions")}
        </a>
      )}
      <div className="community-layout">
        <aside className="community-sidebar">
          <div className="social-page-heading community-heading">
            <h1>{topicId ? t("题目讨论", "Module discussions") : category === 'papers' ? t('论文讨论', 'Paper discussions') : t("社区", "Community")}</h1>
          </div>
          <div ref={boards} className="community-board-scroll">
            <section className="social-panel board-list">
              <span className="sidebar-label">{t("版块", "Boards")}</span>
              <button
                className={!category ? "selected" : ""}
                aria-pressed={!category}
                onClick={() => choose({ category: '' })}
              >
                <MessagesSquare size={17} />
                {t("所有讨论", "All discussions")}
              </button>
              {CATEGORIES.map((c) => (
                <button
                  key={c.id}
                  className={category === c.id ? "selected" : ""}
                  aria-pressed={category === c.id}
                  onClick={() => choose({ category: c.id })}
                >
                  <c.icon size={17} />
                  {t(c.zh, c.en)}
                </button>
              ))}
            </section>
          </div>
          {user && (!topicId || topic) ? (
            <button className="button community-compose-button" onClick={() => setCompose(true)}>
              <Plus size={16} />
              {draftAvailable ? t("继续草稿", "Resume draft") : category === 'papers' ? t('讨论论文', 'Discuss a paper') : t("发布帖子", "New post")}
            </button>
          ) : !user ? (
            <a className="button community-compose-button" href={signInHref(returnTo)}>
              <Plus size={16} />
              {t("登录发帖", "Sign in to post")}
            </a>
          ) : null}
        </aside>
        <section className="community-feed">
          <div className="community-feed-toolbar">
            <div className="social-tabs compact">
              <button
                className={!following && !saved ? "selected" : ""}
                aria-pressed={!following && !saved}
                onClick={() => choose({ following: false, saved: false })}
              >
                {t("所有讨论", "All")}
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
              <form
                className="community-search"
                onSubmit={(e) => {
                  e.preventDefault();
                  choose({ search: query.trim() });
                }}
              >
                <Search size={14} />
                <input
                  aria-label={t("搜索讨论", "Search discussions")}
                  placeholder={t("搜索讨论…", "Search discussions…")}
                  value={query}
                  maxLength={200}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    if (!e.target.value) {
                      choose({ search: '' });
                    }
                  }}
                />
                <button type="submit" aria-label={t("搜索", "Search")}>
                  <ArrowRight size={14} />
                </button>
              </form>
              {sortingAvailable && <label className="community-sort" htmlFor="community-feed-sort"><Settings2 size={14} aria-hidden="true"/><span>{t('排序','Sort')}</span><select id="community-feed-sort" aria-label={t('排序','Sort')} value={sort || 'latest'} onChange={event=>choose({sort:event.target.value==='latest'?undefined:event.target.value as 'active'|'top'})}>
                <option value="latest">{saved?t('最近收藏','Recently saved'):t('最新发布','Newest posts')}</option><option value="active">{t('最近活跃','Recently active')}</option><option value="top">{t('最多赞','Most liked')}</option>
              </select></label>}
            </div>
          </div>
          {topicId && (
            catalogLoading ? <Loading /> : catalogError ? (
              <div className="feed-error">
                <SocialNotice error={catalogError} />
                <button className="button subtle" onClick={reloadCatalog}>
                  {t("重新加载题目信息", "Reload module details")}
                </button>
              </div>
            ) : <TopicContext topic={topic || { id: topicId, title: topicId, retired: true }} />
          )}
          {compose && user && (
            <Composer category={category} topic={topic} owner={user.id} scope={draftScope} papersAvailable={capabilities?.community_papers === true} onDraftChange={setDraftAvailable} onClose={() => setCompose(false)} />
          )}
          <ReadingReturnNotice notice={reading.notice} onRestore={reading.restore}/>
          <div className="social-panel feed-items" data-reading-list="" tabIndex={-1} role="region" aria-label={t('讨论列表','Discussion feed')}>
            {sortingUnavailable ? <Empty title={t('当前服务暂不支持此排序','This order is currently unavailable')}><p>{t('可切回默认排序继续浏览，原有搜索和版块会保留。','Switch to the default order to keep browsing with your search and board filters.')}</p><button type="button" className="button subtle" onClick={()=>choose({sort:undefined})}>{t('使用默认排序','Use default order')}</button></Empty> : error ? (
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
                      : topicId ? t("还没有这道题的讨论", "No discussions for this module yet") : t("这里还没有帖子", "No discussions yet")
                }
              >
                <p>
                  {search
                    ? t("换个关键词试试。", "Try a different search term.")
                    : saved ? t('在帖子中点击收藏，之后可以在这里继续阅读。','Save a post to return to it here later.') : following
                      ? t(
                          "关注其他研究者，查看他们发布的帖子。",
                          "Follow other members to see their posts here.",
                        )
                      : t(
                          topicId ? "可以分享你对公式、实现或测试结果的疑问。" : "分享一篇论文或一个实现问题，开启第一个讨论。",
                          topicId ? "Ask about the formulas, implementation, or test results." : "Share a paper or an implementation question to start the first discussion.",
                        )}
                </p>
                {(category || following || search) && <button className="button subtle" onClick={() => choose(emptyCommunityFeed())}>{t('清除筛选', 'Clear filters')}</button>}
              </Empty>
            ) : (
              data.items.map((post) => <PostCard key={post.id} post={post} returnTo={returnTo} showActivity={sort==='active'} resumed={reading.resumedPost===post.id}/>)
            )}
          </div>
          {!pending && !error && !sortingUnavailable && (
            <Pagination
              offset={offset}
              total={data.total}
              onChange={(offset) => choose({ page: Math.floor(offset / PAGE_SIZE) + 1 })}
            />
          )}
        </section>
      </div>
    </main>
  );
}
function Discussion({ id, returnTo, replyTarget }: { id: string; returnTo?: string; replyTarget?:string }) {
  const { t,locale } = useI18n();
  const { user, capabilities, refresh: refreshAccount, loading:accountLoading } = useAccount();
  const canPublish=canPublishCommunity(user,capabilities)&&!accountLoading;
  useReadingOrigin(accountLoading?undefined:user?'account:'+user.id:'guest',id,returnTo);
  const [editingPost,setEditingPost]=useState(false), [postEdited,setPostEdited]=useState(false);
  const postEditButton=useRef<HTMLButtonElement>(null);
  const [editingReply,setEditingReply]=useState<string>(), [replyEdited,setReplyEdited]=useState<string>();
  function closeReplyEditing(identifier:string) {
    setEditingReply(undefined);
    requestAnimationFrame(()=>document.getElementById(`edit-reply-${identifier}`)?.focus());
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
  const quoteEdit=useRef<(ReplyQuote&{selection:{start:number;end:number;scroll:number}})>(undefined);
  const [quoteNotice,setQuoteNotice]=useState('');
  const [replyFocus,setReplyFocus]=useState<{href:string;source:string;opener:Element|null;selection?:{start:number;end:number;scroll:number}}>();
  useLayoutEffect(()=>{
    if(!replyFocus||replyPreview)return;setReplyFocus(undefined);
    if(replyFocus.href!==location.href||replyFocus.source!==draft.body||!mounted.current||busy)return;
    if(document.activeElement!==replyFocus.opener&&!(document.activeElement===document.body&&!replyFocus.opener?.isConnected))return;
    const field=replyField.current;if(!field)return;
    field.focus({preventScroll:true});field.scrollIntoView({block:'center'});
    field.setSelectionRange(replyFocus.selection?.start??draft.body.length,replyFocus.selection?.end??draft.body.length);
    field.scrollTop=replyFocus.selection?.scroll??field.scrollHeight;
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
  function focusReply(selection?:{start:number;end:number;scroll:number}){
    if(!user||busy||loading||editingReply)return;
    setReplyPreview(false);setReplyFocus({href:location.href,source:currentDraft.current.body,opener:document.activeElement,selection});
  }
  function quoteReply(row:Reply){
    if(!user||busy||loading||editingReply||!post?.replies.some(reply=>reply.id===row.id))return;
    const previous=quoteEdit.current;
    if(previous?.replyId===row.id&&sameReplyDraft(previous.after,currentDraft.current)){setQuoteNotice(t('该引用已在草稿中，可继续编辑。','This excerpt is already in your draft. You can keep editing.'));focusReply();return;}
    const result=quoteReplyDraft(currentDraft.current,id,row,authorName(row.author)||t('用户','User'),locale==='en');
    if(!result.ok){setQuoteNotice(result.reason==='length'?t('引用会超过回复长度上限，现有草稿已保留。请精简后再引用。','This quote would exceed the reply limit. Your draft is kept; shorten it before quoting.'):t('暂时无法生成引用，仍可直接回复。','This reply could not be quoted. You can still reply directly.'));return;}
    const field=replyField.current;
    quoteEdit.current={...result.quote,selection:{start:field?.selectionStart??body.length,end:field?.selectionEnd??body.length,scroll:field?.scrollTop||0}};
    editReply(result.quote.after);setQuoteNotice(t('引用摘要已加入草稿，可继续编辑后发布。','A source-linked excerpt is in your draft. Edit it before posting.'));focusReply();
  }
  function undoQuote(){
    const operation=quoteEdit.current;
    if(!operation||busy||loading||!sameReplyDraft(operation.after,currentDraft.current))return;
    editReply(operation.before);quoteEdit.current=undefined;focusReply(operation.selection);
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
      location.hash = signInHref(postHref(id, returnTo, replyTarget));
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
          location.hash = returnTo || (post?.problem ? topicHref(post.problem.id) : '#/community');
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
      <div className="discussion-navigation"><a className="social-back" href={returnTo || (post?.problem ? topicHref(post.problem.id) : "#/community")}>
        <ArrowLeft size={14} />
        {returnTo&&readInterviewRoute(returnTo) ? t('返回面经','Back to interviews') : returnTo ? t('返回讨论列表', 'Back to discussions') : post?.problem ? t("返回题目讨论", "Back to module discussions") : t("返回社区", "Back to community")}
      </a><DiscussionLink postId={id}/><button className="button subtle" type="button" disabled={loading || busy} onClick={() => void refresh()}>
        <RefreshCw size={14} />{loading ? t('正在加载讨论…','Loading discussion…') : error ? t('重试加载讨论','Retry loading discussion') : t('刷新讨论','Refresh discussion')}
      </button></div>
      {error && <SocialNotice error={error} />}
      {loading && !post ? (
        <Loading />
      ) : post ? (
        <>
          {post.problem && <TopicContext topic={post.problem} discussion />}
          {postEdited&&<p role="status">{t('帖子修改已保存。','Post changes saved.')}</p>}
          {editingPost&&canEditPost(post,owner)&&<PostEditor post={post} owner={owner} disabled={busy||loading} onRefresh={refresh} onClose={closePostEditing} onSaved={confirmed=>{
            setPost(previous=>previous&&previous.id===confirmed.id?{...confirmed,replies:previous.replies}:previous);
            closePostEditing();setPostEdited(true);
          }}/>}
          <article className="social-panel discussion-post">
            <div className="discussion-heading">
              <Category value={post.category} />
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
            {post.interview && <InterviewMeta details={post.interview} returnTo={returnTo}/>}
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
                enabled={capabilities?.community_bookmarks===true} returnTo={postHref(id,returnTo,replyTarget)}
                disabled={busy||loading||editingPost||!!editingReply} onBusyChange={setBusy} onAccountRefresh={refreshAccount}
                onChange={bookmarked=>setPost(previous=>previous&&previous.id===id?{...previous,viewer:{...previous.viewer,bookmarked}}:previous)}/>
              <button type="button" onClick={()=>{const heading=document.getElementById('discussion-replies-heading');heading?.focus({preventScroll:true});heading?.scrollIntoView({block:'start'});}}><MessageCircle size={15}/>{post.counts.replies} {t('条回复','replies')}</button>
              {user?<button type="button" className="discussion-compose-action" disabled={repliesLocked} onClick={()=>focusReply()}><MessageCircle size={15}/>{t('写回复','Write a reply')}</button>:<a className="discussion-compose-action" href={signInHref(postHref(id,returnTo,replyTarget))}>{t('登录回复','Sign in to reply')}</a>}
            </div>
          </article>
          <section ref={replyList.root} tabIndex={-1} aria-labelledby="discussion-replies-heading" className="social-panel discussion-replies">
            {replyTarget&&!post.replies.some(row=>row.id===replyTarget)&&<p role="status">{t('链接指向的回复已删除或暂不可用，其他讨论内容仍可阅读。','The linked reply is deleted or unavailable. The rest of the discussion is still readable.')} <a href={postHref(id,returnTo)}>{t('查看完整讨论','View full discussion')}</a></p>}
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
                    <DiscussionLink key={r.id} postId={id} replyId={r.id} returnTo={returnTo}/>
                    {r.updated_at&&r.updated_at!==r.created_at&&<span>{t('已编辑','Edited')} <DateLabel value={r.updated_at}/></span>}
                    {canEditReply(r,owner)&&<button id={`edit-reply-${r.id}`} type="button" className="text-button" disabled={busy||loading||!!editingReply} onClick={()=>{setEditingReply(r.id);setReplyEdited(undefined);}}>{t('编辑回复','Edit reply')}</button>}
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
                  {editingReply===r.id&&canEditReply(r,owner)&&<ReplyEditor reply={r} owner={owner} disabled={busy||loading} onRefresh={refresh} onClose={()=>closeReplyEditing(r.id)} onSaved={confirmed=>{
                    setPost(previous=>previous&&previous.id===confirmed.post_id?{...previous,replies:previous.replies.map(row=>row.id===confirmed.id?confirmed:row)}:previous);
                    closeReplyEditing(r.id);setReplyEdited(r.id);
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
                <MarkdownEditor submitShortcut value={body} onChange={body=>editReply({body})} preview={replyPreview} onPreviewChange={setReplyPreview}
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
              <JoinPrompt returnTo={postHref(id, returnTo, replyTarget)} />
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
export function UserProfile({ username }: { username: string }) {
  const {user}=useAccount();
  return <ProfilePage key={`${username}:${user?.id||'guest'}`} username={username}/>;
}
function ProfilePage({ username }: { username: string }) {
  const { user } = useAccount();
  // Account changes must remove a previous viewer's private progress immediately.
  return <UserProfileContents key={`${username}:${user?.id || "guest"}`} username={username} />;
}
function UserProfileContents({ username }: { username: string }) {
  const { t } = useI18n();
  const { user, capabilities } = useAccount();
  const [profile, setProfile] = useState<Profile>(),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [tab, setTab] = useState("posts"),
    [offset, setOffset] = useState(0),
    [activity, setActivity] = useState<Page<Activity>>({ items: [], total: 0 }),
    [connections, setConnections] = useState<Page<Connection>>({
      items: [],
      total: 0,
    }),
    [activityLoading, setActivityLoading] = useState(true),
    [activityError, setActivityError] = useState(""),
    [busy, setBusy] = useState(false),
    [revision, setRevision] = useState(0),
    [activityReveal, setActivityReveal] = useState(0);
  const communityContent = useRef<HTMLElement>(null);
  const path = `community/profiles/${encodeURIComponent(username)}`;
  const connectionTab = tab === "following" || tab === "followers";
  useEffect(() => {
    setProfile(undefined);
    setTab("posts");
    setOffset(0);
  }, [username]);
  useEffect(() => {
    let live = true;
    setLoading(true);
    setError("");
    socialApi<Profile>(path)
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
      live = false;
    };
  }, [username, user?.id, revision]);
  useEffect(() => {
    let live = true;
    setActivityLoading(true);
    setActivityError("");
    const params = new URLSearchParams({
      offset: String(offset),
      limit: String(PAGE_SIZE),
      ...(connectionTab ? { kind: tab } : { tab }),
    });
    const promise = connectionTab
      ? socialApi<Page<Connection>>(`${path}/connections?${params}`).then(
          (data) => {
            if (live) setConnections(data);
          },
        )
      : socialApi<Page<Activity>>(`${path}/activity?${params}`).then((data) => {
          if (live) setActivity(data);
        });
    promise
      .catch((e) => {
        if (live) {
          setActivity({ items: [], total: 0 });
          setConnections({ items: [], total: 0 });
          setActivityError(errorMessage(e));
        }
      })
      .finally(() => {
        if (live) setActivityLoading(false);
      });
    return () => {
      live = false;
    };
  }, [username, tab, offset, user?.id, revision]);
  async function follow(target: string, active: boolean) {
    if (!user) {
      location.hash = signInHref(`u/${username}`);
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
  const tabs = [
    ["posts", t("帖子", "Posts")],
    ["replies", t("回复", "Replies")],
    ["likes", t("点赞", "Likes")],
    ["reposts", t("转发", "Reposts")],
    ["following", t("关注", "Following")],
    ["followers", t("粉丝", "Followers")],
  ] as const;
  const chooseTab = (value: string, reveal = false) => {
    if (value !== tab || offset !== 0) setActivityLoading(true);
    setTab(value);
    setOffset(0);
    if (reveal) setActivityReveal((value) => value + 1);
  };
  useLayoutEffect(() => {
    if (!activityReveal) return;
    const section = communityContent.current;
    const selected = section?.querySelector<HTMLButtonElement>('[role="tab"][aria-selected="true"]');
    // Commit the selection before moving focus; repeated shortcuts reveal the same tab too.
    selected?.focus({ preventScroll: true });
    selected?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "auto" });
    section?.scrollIntoView({ block: "start", inline: "nearest", behavior: "auto" });
  }, [activityReveal]);
  return (
    <main className="social-page profile-page">
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
                <span>{profile.counts[key]}</span>
              </button>
            ))}
          </div>
          <section className="social-panel profile-activity" id="profile-community-panel" role="tabpanel" aria-labelledby={`profile-tab-${tab}`} aria-busy={activityLoading}>
            {activityLoading ? (
              <Loading />
            ) : activityError ? (
              <div className="feed-error">
                <SocialNotice error={activityError} />
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
                  key={`${item.kind}-${item.reply?.id || item.post.id}-${index}`}
                >
                  {item.reply ? (
                    <article className="profile-reply-card">
                      <div className="activity-label">
                        <MessageCircle size={13} />
                        {t("回复了", "Replied to")}
                        <a href={postHref(item.post.id)}>{item.post.title}</a>
                        <DateLabel value={item.created_at} />
                      </div>
                      <CommunityMarkdown body={item.reply.body} />
                    </article>
                  ) : (
                    <PostCard
                      post={item.post}
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
              <Empty title={t("还没有相关动态", "No activity here yet")}>
                <p>
                  {t(
                    "发布、回复或互动后，记录会显示在这里。",
                    "Posts and interactions will appear here as they happen.",
                  )}
                </p>
              </Empty>
            )}
          </section>
          <Pagination
            offset={offset}
            total={connectionTab ? connections.total : activity.total}
            onChange={setOffset}
          />
          </section>
          </div>
        </div>
      ) : null}
    </main>
  );
}
