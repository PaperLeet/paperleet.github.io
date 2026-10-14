import type { Problem } from './types';

export type ProblemTopic = {
  id: string;
  number?: number | null;
  title: string;
  title_en?: string | null;
  retired?: boolean;
};

export const topicHref = (id: string) => `#/community/problem/${encodeURIComponent(id)}`;
export const practiceHref = (id: string) => `#/${encodeURIComponent(id)}`;

// App routes have already been decoded by decodeProblemHash.
export function topicIdFromRoute(route: string): string | undefined {
  const prefix = 'community/problem/';
  if (!route.startsWith(prefix)) return undefined;
  const id = route.slice(prefix.length);
  return /^[a-z0-9_-]+\/[a-z0-9_-]+$/.test(id) ? id : undefined;
}

export function topicFromProblem(problem: Problem): ProblemTopic {
  return { id: problem.id, number: problem.number, title: problem.title, title_en: problem.translations?.en?.title };
}

export function topicTitle(topic: ProblemTopic, english: boolean): string {
  return `${topic.number ? `#${topic.number} · ` : ''}${english && topic.title_en ? topic.title_en : topic.title}`;
}
