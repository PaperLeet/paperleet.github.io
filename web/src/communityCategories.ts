export const communityCategories = [
  { id: 'papers', zh: '论文讨论', en: 'Papers' },
  { id: 'implementation', zh: '代码实现', en: 'Implementation' },
  { id: 'questions', zh: '问题求助', en: 'Questions' },
  { id: 'admissions', zh: '求学', en: 'Graduate admissions' },
  { id: 'career', zh: '求职', en: 'Job search' },
  { id: 'lounge', zh: '闲聊', en: 'Lounge' },
] as const;

export type CommunityCategory = typeof communityCategories[number]['id'];
const categoryIds = new Set<string>(communityCategories.map(category => category.id));
export const isCommunityCategory = (value: unknown): value is CommunityCategory =>
  typeof value === 'string' && categoryIds.has(value);

export function communityCategoryHint(category: string): readonly [string, string] | undefined {
  if (category === 'admissions') return ['MS / PhD 申请、选校、导师与录取。', 'MS / PhD applications, programs, supervisors and offers.'];
  if (category === 'career') return ['实习、招聘、面试与职业发展。', 'Internships, hiring, interviews and career development.'];
}
