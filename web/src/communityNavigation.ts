import { topicHref } from './communityTopics';

export type CommunityFeedState = { category: string; following: boolean; search: string; page: number };
export const COMMUNITY_PAGE_SIZE = 20;
export const emptyCommunityFeed = (): CommunityFeedState => ({ category: '', following: false, search: '', page: 1 });
const categories = ['', 'papers', 'implementation', 'questions', 'career', 'lounge'];
const moduleId = /^[a-z0-9_-]+\/[a-z0-9_-]+$/;
const maxPage = Math.floor(Number.MAX_SAFE_INTEGER / COMMUNITY_PAGE_SIZE);

function parts(route: string) {
  const raw = route.replace(/^#\/?/, '');
  const query = raw.indexOf('?');
  try {
    return { path: decodeURIComponent(query < 0 ? raw : raw.slice(0, query)), params: new URLSearchParams(query < 0 ? '' : raw.slice(query + 1)) };
  } catch { return undefined; }
}
function knownParameters(params: URLSearchParams, allowed: string[]) {
  return [...params.keys()].every(key => allowed.includes(key) && params.getAll(key).length === 1);
}

/** Parse only local feed routes and supported filters; nested return routes are forbidden. */
export function readCommunityFeed(route: string): { topicId?: string; state: CommunityFeedState } | undefined {
  const value = parts(route);
  if (!value || !knownParameters(value.params, ['category', 'following', 'q', 'page'])) return;
  let topicId: string | undefined;
  if (value.path !== 'community') {
    if (!value.path.startsWith('community/problem/')) return;
    topicId = value.path.slice('community/problem/'.length);
    if (!moduleId.test(topicId)) return;
  }
  const category = value.params.get('category') || '';
  const following = value.params.get('following') || '';
  const search = (value.params.get('q') || '').trim();
  const page = value.params.get('page') || '1';
  if (!categories.includes(category) || !['', '1', 'true', '0', 'false'].includes(following)
    || Array.from(search).length > 200 || !/^[1-9]\d*$/.test(page) || !Number.isSafeInteger(Number(page)) || Number(page) > maxPage) return;
  return { topicId, state: { category, following: following === '1' || following === 'true', search, page: Number(page) } };
}

export function communityFeedHref(topicId: string | undefined, state: CommunityFeedState): string {
  if (topicId && !moduleId.test(topicId)) throw Error('Invalid community module');
  const base = topicId ? topicHref(topicId) : '#/community';
  const params = new URLSearchParams();
  if (state.category) params.set('category', state.category);
  if (state.following) params.set('following', '1');
  if (state.search.trim()) params.set('q', state.search.trim());
  if (state.page > 1) params.set('page', String(state.page));
  const href = base + (params.size ? `?${params}` : '');
  if (!readCommunityFeed(href)) throw Error('Invalid community filters');
  return href;
}

export function communityPostHref(id: string, returnTo?: string) {
  const base = `#/community/${encodeURIComponent(id)}`;
  const feed = returnTo ? readCommunityFeed(returnTo) : undefined;
  if (!feed) return base;
  const from = communityFeedHref(feed.topicId, feed.state).slice(2);
  return `${base}?${new URLSearchParams({ from })}`;
}

export function communityPostReturn(route: string, id: string): string | undefined {
  const value = parts(route);
  if (!value || value.path !== `community/${id}` || !knownParameters(value.params, ['from'])) return;
  const feed = readCommunityFeed(value.params.get('from') || '');
  return feed ? communityFeedHref(feed.topicId, feed.state) : undefined;
}

/** Account continuations accept only canonical community filters and feed-only return targets. */
export function communityContinuation(route: string): string | undefined {
  const feed = readCommunityFeed(route);
  if (feed) return communityFeedHref(feed.topicId, feed.state).slice(2);
  const value = parts(route);
  const id = value?.path.match(/^community\/([a-f0-9]{32})$/)?.[1];
  if (!id) return;
  const back = communityPostReturn(route, id);
  return back ? communityPostHref(id, back).slice(2) : undefined;
}
