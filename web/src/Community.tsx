import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  BriefcaseBusiness,
  Check,
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
  Repeat2,
  Search,
  Settings2,
  Trash2,
  UserPlus,
  Users,
  X,
} from "lucide-react";
import { useI18n } from "./i18n";
import ProfileProgress from "./ProfileProgress";
import { practiceHref, topicHref, topicTitle, type ProblemTopic } from "./communityTopics";
import { signInHref } from "./accountNavigation";
import { COMMUNITY_PAGE_SIZE as PAGE_SIZE, communityFeedHref, communityPostHref, communityPostReturn, emptyCommunityFeed, readCommunityFeed, type CommunityFeedState } from './communityNavigation';
import { clearPublishedDraft, hasDraft, readPostDraft, tabDraftStorage, writePostDraft, type PostDraft } from "./communityDrafts";
import { clearPublishedReplyDraft, hasReplyDraft, readReplyDraft, sameReplyDraft, writeReplyDraft, type ReplyDraft } from "./replyDrafts";
import {
  Avatar,
  DateLabel,
  InstitutionBadge,
  PersonLink,
  SocialNotice,
  TestAccountBadge,
  errorMessage,
  profileHref,
  socialApi,
  useAccount,
} from "./social";
import type { PublicUser } from "./social";

type Counts = { likes: number; reposts: number; replies: number };
type Post = {
  id: string;
  title: string;
  body: string;
  category: string;
  problem_id: string | null;
  problem: ProblemTopic | null;
  created_at: string;
  updated_at: string;
  author: PublicUser;
  counts: Counts;
  viewer: { liked: boolean; reposted: boolean; can_delete: boolean };
};
type Reply = {
  id: string;
  post_id: string;
  parent_id: string | null;
  body: string;
  created_at: string;
  author: PublicUser;
  can_delete: boolean;
};
type PostDetail = Post & { replies: Reply[] };
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
        <span className="social-eyebrow">{t("题目讨论", "Module discussion")}</span>
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
function CommunityMarkdown({ body }: { body: string }) {
  return (
    <div className="prose community-prose">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer nofollow">
              {children}
            </a>
          ),
          img: ({ alt }) => (
            <span className="community-image-alt">
              {alt ? `[${alt}]` : "[image]"}
            </span>
          ),
        }}
      >
        {body}
      </Markdown>
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
function PostCard({ post, activity, returnTo }: { post: Post; activity?: ReactNode; returnTo?: string }) {
  const { locale } = useI18n();
  return (
    <article className="community-post-card">
      {activity && <div className="activity-label">{activity}</div>}
      <div className="post-card-layout">
        <a
          href={profileHref(post.author)}
          aria-label={post.author.display_name}
        >
          <Avatar user={post.author} size={36} />
        </a>
        <div className="post-card-main">
          <div className="post-card-meta">
            <a className="post-author-name" href={profileHref(post.author)}>
              {post.author.display_name}
            </a>
            <TestAccountBadge isTestUser={post.author.is_test_user} />
            <span>·</span>
            <DateLabel value={post.created_at} />
            <Category value={post.category} />
          </div>
          <a className="post-title" href={postHref(post.id, returnTo)}>
            {post.title}
          </a>
          {post.problem && (
            <a className="post-problem-link" href={topicHref(post.problem.id)}>
              <Code2 size={13} />{topicTitle(post.problem, locale === "en")}
            </a>
          )}
          <p className="post-excerpt">
            {post.body.replace(/[#*`>\[\]]/g, "").slice(0, 190)}
          </p>
          <div className="post-card-counts">
            <span>
              <MessageCircle size={13} />
              {post.counts.replies}
            </span>
            <span>
              <Heart size={13} />
              {post.counts.likes}
            </span>
            <span>
              <Repeat2 size={13} />
              {post.counts.reposts}
            </span>
          </div>
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
  onDraftChange,
  onClose,
}: {
  category: string;
  topic?: ProblemTopic;
  owner: string;
  scope: string;
  onDraftChange: (available: boolean) => void;
  onClose: () => void;
}) {
  const { t, locale } = useI18n();
  const initialCategory = category || (topic ? "questions" : "papers");
  const [draft, setDraft] = useState<PostDraft>(() => readPostDraft(tabDraftStorage(), owner, scope) || { title: "", body: "", category: initialCategory });
  const currentDraft = useRef(draft);
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
    setSaved(writePostDraft(tabDraftStorage(), owner, scope, next));
    onDraftChange(hasDraft(next));
  }
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const draftExists = hasDraft(draft);
  const draftMessage = saved === false
    ? t("当前标签页无法保存草稿，离开前请复制内容。", "This tab cannot save your draft. Copy the text before leaving.")
    : draftExists
      ? t("未发布草稿已保存在本标签页，仅当前账号可恢复。", "Unpublished draft saved in this tab, separately for your account.")
      : t("草稿仅保存在本标签页，发布后才会公开。", "Drafts stay in this tab and become public only when published.");
  function discardDraft() {
    edit({ title: "", body: "", category: initialCategory });
    setDiscardRequested(false);
    setPreview(false);
  }
  async function publish() {
    if (busy) return;
    const submitted = { ...currentDraft.current };
    const submittedRoute = location.hash;
    setBusy(true);
    setError("");
    try {
      const post = await socialApi<Post>("community/posts", {
        title: submitted.title.trim(),
        body: submitted.body.trim(),
        category: submitted.category,
        expected_user_id: owner,
        ...(topic ? { problem_id: topic.id } : {}),
      });
      clearPublishedDraft(tabDraftStorage(), owner, scope, submitted);
      if (mounted.current && location.hash === submittedRoute) location.hash = postHref(post.id, submittedRoute);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="social-panel community-composer">
      <div className="panel-title">
        <h2>{t("发布帖子", "Start a discussion")}</h2>
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
          <span>{t("丢弃后无法恢复，确定清空标题和正文？", "Discard this draft? Its title and body cannot be recovered.")}</span>
          <button type="button" disabled={busy} onClick={discardDraft}>{t("确认丢弃", "Discard")}</button>
          <button type="button" onClick={() => setDiscardRequested(false)}>{t("保留草稿", "Keep draft")}</button>
        </div>
      )}
      <form
        className="social-form"
        onSubmit={(e) => {
          e.preventDefault();
          void publish();
        }}
      >
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
        <label>
          {t("标题", "Title")}
          <input
            required
            maxLength={160}
            minLength={3}
            value={title}
            disabled={busy}
            placeholder={t("你想讨论什么？", "What would you like to discuss?")}
            onChange={(e) => edit({ title: e.target.value })}
          />
        </label>
        <div className="editor-toggle">
          <button
            className={!preview ? "selected" : ""}
            type="button"
            onClick={() => setPreview(false)}
          >
            {t("编写", "Write")}
          </button>
          <button
            className={preview ? "selected" : ""}
            type="button"
            onClick={() => setPreview(true)}
          >
            {t("预览", "Preview")}
          </button>
          <small>Markdown</small>
        </div>
        {preview ? (
          <div className="compose-preview">
            <CommunityMarkdown
              body={
                body ||
                t("写下内容后在这里预览。", "Your preview will appear here.")
              }
            />
          </div>
        ) : (
          <label className="body-label">
            <span className="sr-only">{t("正文", "Post body")}</span>
            <textarea
              required
              rows={10}
              maxLength={20000}
              value={body}
              disabled={busy}
              placeholder={t(
                "分享背景、思考或代码片段…",
                "Share context, ideas, or a code snippet…",
              )}
              onChange={(e) => edit({ body: e.target.value })}
            />
          </label>
        )}
        <div className="form-footer">
          <small>
            {t(
              "帖子以你的公开昵称发布。",
              "Posts are published under your public display name.",
            )}
            {topic && " " + t("仅发布你填写的内容，练习草稿与判题报告不会自动附带。", "Only the text you enter is published; practice drafts and reports are not attached.")}
          </small>
          <button className="button" disabled={busy || !body.trim()}>
            {busy ? <span className="spinner" /> : <Plus size={15} />}
            {t("发布", "Publish")}
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
    <Discussion key={`${postId}:${user?.id || "guest"}`} id={postId} returnTo={communityPostReturn(route || '', postId)} />
  ) : (
    <CommunityFeed key={`${scope.topicId || "all"}:${user?.id || "guest"}`} route={route} {...scope} />
  );
}
function CommunityFeed({ route, topicId, topic, catalogLoading, catalogError, reloadCatalog }: Omit<CommunityProps, "postId">) {
  const { t, locale } = useI18n();
  const { user } = useAccount();
  const draftScope = topicId || "general";
  const [draftAvailable, setDraftAvailable] = useState(() => !!(user && readPostDraft(tabDraftStorage(), user.id, draftScope)));
  const filters = readCommunityFeed(route || (topicId ? topicHref(topicId) : 'community'))?.state || emptyCommunityFeed();
  const { category, following, search, page } = filters;
  const offset = (page - 1) * PAGE_SIZE;
  const returnTo = communityFeedHref(topicId, filters);
  const requestKey = JSON.stringify([topicId, user?.id, category, following, search, page]);
  const [query, setQuery] = useState(search);
  const boards = useRef<HTMLElement>(null);
  const [data, setData] = useState<Page<Post> & { requestKey?: string }>({ items: [], total: 0 }),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [compose, setCompose] = useState(false),
    [retry, setRetry] = useState(0);
  const pending = loading || data.requestKey !== requestKey;
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
    setLoading(true);
    setError("");
    const params = new URLSearchParams({
      category,
      q: search,
      following: following ? "true" : "false",
      offset: String(offset),
      limit: String(PAGE_SIZE),
      ...(topicId ? { problem_id: topicId } : {}),
    });
    if (following && !user) {
      setData({ items: [], total: 0, requestKey });
      setLoading(false);
      return;
    }
    socialApi<Page<Post>>(`community/posts?${params}`)
      .then((data) => {
        if (!live) return;
        const lastPage = Math.max(1, Math.ceil(data.total / PAGE_SIZE));
        if (page > lastPage) {
          redirected = true;
          location.replace(communityFeedHref(topicId, { category, following, search, page: lastPage }));
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
      live = false;
    };
  }, [category, following, search, offset, user?.id, retry, topicId]);
  return (
    <main className="social-page community-page">
      {topicId && (
        <a className="social-back" href="#/community">
          <ArrowLeft size={14} />{t("所有社区讨论", "All community discussions")}
        </a>
      )}
      <div className="social-page-heading community-heading">
        <h1>{topicId ? t("题目讨论", "Module discussions") : t("社区", "Community")}</h1>
        {user && (!topicId || topic) ? (
          <button className="button" onClick={() => setCompose(true)}>
            <Plus size={16} />
            {draftAvailable ? t("继续草稿", "Resume draft") : t("发布帖子", "New post")}
          </button>
        ) : !user ? (
          <a className="button" href={signInHref(returnTo)}>
            <Plus size={16} />
            {t("登录发帖", "Sign in to post")}
          </a>
        ) : null}
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
      <div className="community-layout">
        <aside ref={boards} className="community-sidebar">
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
        </aside>
        <section className="community-feed">
          {compose && user && (
            <Composer category={category} topic={topic} owner={user.id} scope={draftScope} onDraftChange={setDraftAvailable} onClose={() => setCompose(false)} />
          )}
          <div className="community-feed-toolbar">
            <div className="social-tabs compact">
              <button
                className={!following ? "selected" : ""}
                aria-pressed={!following}
                onClick={() => choose({ following: false })}
              >
                {t("最新", "Latest")}
              </button>
              <button
                className={following ? "selected" : ""}
                aria-pressed={following}
                onClick={() => choose({ following: true })}
              >
                <Users size={14} />
                {t("关注", "Following")}
              </button>
            </div>
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
          </div>
          {!pending && !error && !(following && !user) && <p className="community-result-summary" role="status">
            {data.total ? `${offset + 1}–${Math.min(offset + PAGE_SIZE, data.total)} / ${data.total}` : '0'} {t('条讨论', 'discussions')}
          </p>}
          <div className="social-panel feed-items">
            {error ? (
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
            ) : following && !user ? (
              <Empty
                title={t(
                  "关注你感兴趣的人",
                  "Follow people you want to learn with",
                )}
                icon={<Users size={31} />}
              >
                <JoinPrompt returnTo={returnTo} />
              </Empty>
            ) : !data.items.length ? (
              <Empty
                title={
                  search
                    ? t("没有找到相关讨论", "No matching discussions")
                    : following
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
                    : following
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
              data.items.map((post) => <PostCard key={post.id} post={post} returnTo={returnTo} />)
            )}
          </div>
          {!pending && !error && (
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
function Discussion({ id, returnTo }: { id: string; returnTo?: string }) {
  const { t } = useI18n();
  const { user } = useAccount();
  const [post, setPost] = useState<PostDetail>(),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [deleteTarget, setDeleteTarget] = useState<string>();
  // Community keys this component by both post and account before it renders.
  const owner = user?.id || "";
  const [draft, setDraft] = useState<ReplyDraft>(() => readReplyDraft(tabDraftStorage(), owner, id) || { body: "" });
  const currentDraft = useRef(draft);
  const [draftSaved, setDraftSaved] = useState<boolean | null>(() => hasReplyDraft(draft) ? true : null);
  const [discardReply, setDiscardReply] = useState(false);
  const [replyPreview, setReplyPreview] = useState(false);
  const [replyNotice, setReplyNotice] = useState("");
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  const { body } = draft;
  const target = draft.parentId ? post?.replies.find((reply) => reply.id === draft.parentId) : undefined;
  const missingTarget = !!draft.parentId && !!post && !target;
  function editReply(change: Partial<ReplyDraft>) {
    const next = { ...currentDraft.current, ...change };
    currentDraft.current = next;
    setDraft(next);
    setDraftSaved(writeReplyDraft(tabDraftStorage(), owner, id, next));
    setReplyNotice("");
  }
  async function refresh() {
    const data = await socialApi<PostDetail>(postPath(id));
    if (mounted.current) setPost(data);
  }
  useEffect(() => {
    let live = true;
    setLoading(true);
    setError("");
    socialApi<PostDetail>(postPath(id))
      .then((data) => {
        if (live) setPost(data);
      })
      .catch((e) => {
        if (live) setError(errorMessage(e));
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [id, user?.id]);
  async function react(kind: "like" | "repost") {
    if (!user) {
      location.hash = signInHref(postHref(id, returnTo));
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
  async function reply() {
    if (!user || busy || !currentDraft.current.body.trim() || missingTarget) return;
    const submitted = { ...currentDraft.current };
    const submittedRoute = location.hash;
    setBusy(true);
    setError("");
    setReplyNotice("");
    try {
      await socialApi(`${postPath(id)}/replies`, {
        body: submitted.body.trim(),
        expected_user_id: owner,
        ...(submitted.parentId ? { parent_id: submitted.parentId } : {}),
      });
      clearPublishedReplyDraft(tabDraftStorage(), owner, id, submitted);
      if (!mounted.current || location.hash !== submittedRoute) return;
      if (sameReplyDraft(currentDraft.current, submitted)) {
        currentDraft.current = { body: "" };
        setDraft(currentDraft.current);
        setDraftSaved(null);
        setDiscardReply(false);
        setReplyPreview(false);
      }
      setReplyNotice(t("回复已发布。", "Reply posted."));
      try { await refresh(); }
      catch {
        if (mounted.current) setError(t("回复已发布，但刷新讨论失败。请刷新页面查看。", "Your reply was posted, but the discussion could not refresh. Refresh the page to view it."));
      }
    } catch (e) {
      if (mounted.current) setError(errorMessage(e));
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  async function remove() {
    if (!deleteTarget) return;
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
      <a className="social-back" href={returnTo || (post?.problem ? topicHref(post.problem.id) : "#/community")}>
        <ArrowLeft size={14} />
        {returnTo ? t('返回讨论列表', 'Back to discussions') : post?.problem ? t("返回题目讨论", "Back to module discussions") : t("返回社区", "Back to community")}
      </a>
      {error && <SocialNotice error={error} />}
      {loading ? (
        <Loading />
      ) : post ? (
        <>
          {post.problem && <TopicContext topic={post.problem} discussion />}
          <article className="social-panel discussion-post">
            <div className="discussion-heading">
              <Category value={post.category} />
              <h1>{post.title}</h1>
              <div className="discussion-author">
                <PersonLink user={post.author} />
                <InstitutionBadge institution={post.author.institution} />
                <DateLabel value={post.created_at} />
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
            <CommunityMarkdown body={post.body} />
            <div className="discussion-actions">
              <button
                className={post.viewer.liked ? "active" : ""}
                disabled={busy}
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
                disabled={busy}
                aria-pressed={post.viewer.reposted}
                onClick={() => void react("repost")}
              >
                <Repeat2 size={16} />
                {post.viewer.reposted
                  ? t("已转发", "Reposted")
                  : t("转发", "Repost")}
                <span>{post.counts.reposts}</span>
              </button>
              <span>
                <MessageCircle size={15} />
                {post.counts.replies} {t("条回复", "replies")}
              </span>
            </div>
          </article>
          <section className="social-panel discussion-replies">
            <div className="panel-title">
              <h2>
                {t("回复", "Replies")} <span>{post.counts.replies}</span>
              </h2>
            </div>
            {post.replies.length ? (
              post.replies.map((r) => (
                <article
                  key={r.id}
                  className={`community-reply ${r.parent_id ? "nested-reply" : ""}`}
                  id={`reply-${r.id}`}
                >
                  <div className="reply-heading">
                    <PersonLink user={r.author} compact />
                    <DateLabel value={r.created_at} />
                    {r.parent_id && (
                      <button
                        className="reply-context"
                        onClick={() =>
                          document
                            .getElementById(`reply-${r.parent_id}`)
                            ?.scrollIntoView({
                              behavior: "smooth",
                              block: "center",
                            })
                        }
                      >
                        {t("回复", "Replying to")}{" "}
                        {post.replies.find(
                          (parent) => parent.id === r.parent_id,
                        )?.author.display_name ||
                          t("已删除的回复", "a deleted reply")}
                      </button>
                    )}
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
                  {user && (
                    <button
                      className="reply-link"
                      disabled={busy}
                      onClick={() => {
                        editReply({ parentId: r.id });
                        setReplyPreview(false);
                        requestAnimationFrame(() => document.getElementById("reply-body")?.focus());
                      }}
                    >
                      <MessageCircle size={12} />
                      {t("回复", "Reply")}
                    </button>
                  )}
                </article>
              ))
            ) : (
              <p className="no-replies">
                {t(
                  "还没有回复，分享你的想法吧。",
                  "No replies yet. Share your thoughts.",
                )}
              </p>
            )}
          </section>
          {user ? (
            <section className="social-panel reply-composer">
              <div className="panel-title">
                <h3>{t("参与讨论", "Join the discussion")}</h3>
                <PersonLink user={user} compact />
              </div>
              {draft.parentId && (
                <div className="reply-target">
                  <span>
                    {t("回复", "Replying to")} {target?.author.display_name || t("已删除或暂不可用的回复", "a deleted or unavailable reply")}
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
              {replyNotice && <p role="status">{replyNotice}</p>}
              <form
                className="social-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  void reply();
                }}
              >
                <div className="editor-toggle" role="group" aria-label={t("回复编辑模式", "Reply editor mode")}>
                  <button type="button" className={!replyPreview ? "selected" : ""} aria-pressed={!replyPreview} disabled={busy} onClick={() => setReplyPreview(false)}>{t("编写", "Write")}</button>
                  <button type="button" className={replyPreview ? "selected" : ""} aria-pressed={replyPreview} disabled={busy} onClick={() => setReplyPreview(true)}>{t("预览", "Preview")}</button>
                  <small>Markdown</small>
                </div>
                {replyPreview ? <div className="compose-preview"><CommunityMarkdown body={body || t("写下内容后在这里预览。", "Your preview will appear here.")} /></div> : <label>
                  <span className="sr-only">{t("回复内容", "Your reply")}</span>
                  <textarea
                    id="reply-body"
                    required
                    rows={5}
                    maxLength={10000}
                    disabled={busy}
                    value={body}
                    placeholder={t(
                      "写下你的回复…（支持 Markdown）",
                      "Write a reply… (Markdown supported)",
                    )}
                    onChange={(e) => editReply({ body: e.target.value })}
                  />
                </label>}
                <div className="form-footer">
                  <small>
                    {t(
                      "回复会公开展示在你的个人主页。",
                      "Replies appear publicly on your profile.",
                    )}
                  </small>
                  <button className="button" disabled={busy || !body.trim() || missingTarget}>
                    <MessageCircle size={14} />
                    {t("发布回复", "Post reply")}
                  </button>
                </div>
              </form>
            </section>
          ) : (
            <section className="social-panel">
              <JoinPrompt returnTo={postHref(id, returnTo)} />
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
  const { user } = useAccount();
  // Account changes must remove a previous viewer's private progress immediately.
  return <UserProfileContents key={`${username}:${user?.id || "guest"}`} username={username} />;
}
function UserProfileContents({ username }: { username: string }) {
  const { t } = useI18n();
  const { user } = useAccount();
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
    [revision, setRevision] = useState(0);
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
  const chooseTab = (value: string) => {
    setTab(value);
    setOffset(0);
  };
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
                  <InstitutionBadge institution={profile.institution} />
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
            </div>
            <div className="profile-stats">
              <button onClick={() => chooseTab("following")}>
                <b>{profile.counts.following}</b>
                {t("关注", "Following")}
              </button>
              <button onClick={() => chooseTab("followers")}>
                <b>{profile.counts.followers}</b>
                {t("粉丝", "Followers")}
              </button>
            </div>
          </section>
          <section className="social-panel profile-community-summary">
            <h2>{t("社区贡献", "Community")}</h2>
            <dl>
              <div><dt><MessagesSquare size={16} />{t("帖子", "Posts")}</dt><dd><button aria-label={t("查看帖子", "View posts")} onClick={() => chooseTab("posts")}>{profile.counts.posts}</button></dd></div>
              <div><dt><MessageCircle size={16} />{t("回复", "Replies")}</dt><dd><button aria-label={t("查看回复", "View replies")} onClick={() => chooseTab("replies")}>{profile.counts.replies}</button></dd></div>
              <div><dt><Heart size={16} />{t("获赞", "Likes received")}</dt><dd>{profile.counts.likes_received}</dd></div>
              <div><dt><Repeat2 size={16} />{t("获转发", "Reposts received")}</dt><dd>{profile.counts.reposts_received}</dd></div>
            </dl>
            {profile.viewer.is_self && <a href="#/community">{t("加入讨论", "Join a discussion")}<ArrowRight size={14} /></a>}
          </section>
          </aside>
          <div className="profile-main">
          <ProfileProgress username={username} isSelf={profile.viewer.is_self} />
          <section className="profile-community-content" aria-labelledby="profile-community-title">
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
                      <InstitutionBadge institution={member.institution} />
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
