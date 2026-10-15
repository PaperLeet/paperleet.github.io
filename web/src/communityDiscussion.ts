import type { ProblemTopic } from './communityTopics';
import type { PublicUser } from './social';

export type Counts = { likes: number; reposts: number; replies: number };
export type Post = {
  id: string; title: string; body: string; category: string;
  anonymous?: boolean;
  interview?: import('./interviewModels').InterviewDetails;
  paper_url?: string | null; paper?: import('./communityPapers').PaperPreview | null;
  problem_id: string | null; problem: ProblemTopic | null;
  created_at: string; updated_at: string; last_reply_at?: string | null; edit_version?: string; author: PublicUser; counts: Counts;
  viewer: { liked: boolean; reposted: boolean; can_delete: boolean; can_edit?: boolean; bookmarked?: boolean };
};
export type Reply = {
  id: string; post_id: string; parent_id: string | null; body: string;
  created_at: string; updated_at?:string; edit_version?:string; author: PublicUser; can_delete: boolean; can_edit?:boolean;
};
export type PostDetail = Post & { replies: Reply[] };

/** A successful POST confirms this row even if a later GET cannot refresh. */
export function withConfirmedReply(post: PostDetail | undefined, reply: Reply): PostDetail | undefined {
  if (!post || reply.post_id !== post.id || post.replies.some(row => row.id === reply.id)) return post;
  const replies = [...post.replies, reply];
  return {...post,replies,counts:{...post.counts,replies:Math.max(post.counts.replies + 1,replies.length)}};
}
