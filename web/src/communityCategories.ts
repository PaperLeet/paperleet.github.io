export const communityCategories = [
  { id: 'papers', zh: '论文讨论', en: 'Papers' },
  { id: 'implementation', zh: '代码实现', en: 'Implementation' },
  { id: 'questions', zh: '问题求助', en: 'Questions' },
  { id: 'admissions', zh: '求学', en: 'School' },
  { id: 'career', zh: '求职', en: 'Job search' },
  { id: 'lounge', zh: '闲聊', en: 'Lounge' },
] as const;

export type CommunityCategory = string;
export const isCommunityCategory = (value: unknown): value is CommunityCategory =>
  typeof value === 'string' && /^[a-z][a-z0-9_-]{0,39}$/.test(value);

export function communityCategoryHint(category: string): readonly [string, string] | undefined {
  if (category === 'admissions') return ['MS / PhD 申请、选校、导师与录取。', 'MS / PhD applications, programs, supervisors and offers.'];
  if (category === 'career') return ['实习、招聘、面试与职业发展。', 'Internships, hiring, interviews and career development.'];
}

export const validCommunityTags = (value:unknown):value is string[] => Array.isArray(value) && value.length<=5 && value.every(isCommunityCategory) && new Set(value).size===value.length;
