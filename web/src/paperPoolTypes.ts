import type { PublicUser } from './social';

export type PaperPoolVote = 'approve' | 'needs_changes';
export type PoolProblem = {
  title: string;
  title_en?: string;
  statement: string;
  statement_en?: string;
  starter_code: string;
  functions?: {name: string; signature: string; returns?: string}[];
  difficulty?: string;
};
export type PoolProposal = {
  id: string;
  arxiv_id: string;
  arxiv_url: string;
  prompt?: string;
  status: string;
  reason?: string;
  created_at: string;
  updated_at: string;
  author?: PublicUser;
  version: number | null;
  opened_at?: string | null;
  problem: PoolProblem | null;
  review_summary: {
    approve: number; needs_changes: number; total: number;
    minimum_reviews: number; approval_ratio: number; review_hours: number;
    review_required?: boolean; eligible_at?: string | null;
  };
  reviews?: {vote: PaperPoolVote; reason: string; version: number; author: PublicUser; created_at: string; updated_at: string}[];
  viewer: {
    can_review: boolean; has_attempted: boolean;
    vote?: PaperPoolVote | {vote: PaperPoolVote; reason?: string} | null;
    reason?: string | null; eligibility_reason?: string | null; can_manage?: boolean;
  };
};
export type PaperPoolCapabilities = {
  can_submit: boolean;
  automation_enabled: boolean;
  generation: {ready: boolean; reason?: string};
  attempts?: {ready: boolean; reason?: string};
  reason?: string;
};
export type PoolList = {items: PoolProposal[]; total?: number; capabilities: PaperPoolCapabilities};
export type PoolDuplicate =
  | {kind: 'proposal'; proposal: PoolProposal}
  | {kind: 'catalog'; problems: {id: string; title: string; title_en?: string}[]};
export type PoolSubmission = {proposal: PoolProposal | null; duplicate?: PoolDuplicate};
export type PoolAttempt = {
  id: string; version: number;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  error?: string;
  report?: {passed: boolean; error?: string; tests: {id: string; passed: boolean; message?: string}[]};
};
export type PoolAttemptResponse = {attempt: PoolAttempt; proposal: PoolProposal};
