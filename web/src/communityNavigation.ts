import {publicHref} from './appLocation';
import {readProfileRoute,profileFeedHref} from './profileNavigation';
import { topicHref } from './communityTopics';
import {readInterviewRoute,interviewHref} from './interviewNavigation';
import { isCommunityCategory } from './communityCategories';
import {readReportHistory,reportHistoryHref} from './reportHistoryNavigation';

export type CommunityFeedState = { category: string; tag?:string; following: boolean; saved?: boolean; solutions?: boolean; sort?: 'active' | 'top'; days?:'7'|'30'; search: string; page: number };
export const solutionsHref = (topicId?:string) => communityFeedHref(topicId,{...emptyCommunityFeed(),solutions:true});
export const COMMUNITY_PAGE_SIZE = 20;
export const emptyCommunityFeed = (): CommunityFeedState => ({ category: '', following: false, search: '', page: 1 });
const moduleId = /^[a-z0-9_-]+\/[a-z0-9_-]+$/;
export const MAX_COMMUNITY_PAGE = Math.floor(Number.MAX_SAFE_INTEGER / COMMUNITY_PAGE_SIZE);

function parts(route: string) {
  const raw = route.replace(/^#\/?/, '');
  const query = raw.indexOf('?');
  try {
    return { path: decodeURIComponent(query < 0 ? raw : raw.slice(0, query)), params: new URLSearchParams(query < 0 ? '' : raw.slice(query + 1)) };
  } catch { return undefined; }
}
const validFlag=(value:string)=>['', '1', 'true', '0', 'false'].includes(value);
function knownParameters(params: URLSearchParams, allowed: string[]) {
  return [...params.keys()].every(key => allowed.includes(key) && params.getAll(key).length === 1);
}

/** Parse only local feed routes and supported filters; nested return routes are forbidden. */
export function readCommunityFeed(route: string): { topicId?: string; state: CommunityFeedState } | undefined {
  const value = parts(route);
  if (!value || !knownParameters(value.params, ['category', 'tag', 'following', 'saved', 'solutions', 'sort', 'days', 'q', 'page'])) return;
  let topicId: string | undefined;
  if (value.path !== 'community') {
    if (!value.path.startsWith('community/problem/')) return;
    topicId = value.path.slice('community/problem/'.length);
    if (!moduleId.test(topicId)) return;
  }
  const category = value.params.get('category') || '';
  const tag=value.params.get('tag')||'';
  const following = value.params.get('following') || '';
  const saved = value.params.get('saved') || '';
  const solutions = value.params.get('solutions') || '';
  const sort = value.params.get('sort') || 'latest';
  const days = value.params.get('days') ?? 'all';
  const search = (value.params.get('q') || '').trim();
  const page = value.params.get('page') || '1';
  if ((tag!==''&&!isCommunityCategory(tag)) || (category !== '' && !isCommunityCategory(category)) || !validFlag(following)
    || !validFlag(saved) || !validFlag(solutions) || !['latest','active','top'].includes(sort) || !['all','7','30'].includes(days)
    || Array.from(search).length > 200 || !/^[1-9]\d*$/.test(page) || !Number.isSafeInteger(Number(page)) || Number(page) > MAX_COMMUNITY_PAGE) return;
  return { topicId, state: { category, ...(tag?{tag}:{}), following: following === '1' || following === 'true',
    ...(solutions === '1' || solutions === 'true' ? {solutions:true} : {}),
    ...(saved === '1' || saved === 'true' ? {saved: true} : {}), ...(sort === 'active' || sort === 'top' ? {sort} : {}), ...(days==='7'||days==='30'?{days}:{}), search, page: Number(page) } };
}

export function communityFeedHref(topicId: string | undefined, state: CommunityFeedState): string {
  if (topicId && !moduleId.test(topicId)) throw Error('Invalid community module');
  const base = topicId ? topicHref(topicId) : '#/community';
  const params = new URLSearchParams();
  if (state.category) params.set('category', state.category);
  if (state.tag) params.set('tag',state.tag);
  if (state.following) params.set('following', '1');
  if (state.saved) params.set('saved', '1');
  if (state.solutions) params.set('solutions', '1');
  if (state.sort) params.set('sort', state.sort);
  if (state.days) params.set('days', state.days);
  if (state.search.trim()) params.set('q', state.search.trim());
  if (state.page > 1) params.set('page', String(state.page));
  const href = base + (params.size ? `?${params}` : '');
  if (!readCommunityFeed(href)) throw Error('Invalid community filters');
  return href;
}

export function communityPostHref(id: string, returnTo?: string, replyId?: string) {
  const base = `#/community/${encodeURIComponent(id)}`;
  const feed = returnTo ? readCommunityFeed(returnTo) : undefined;
  const interviews = returnTo ? readInterviewRoute(returnTo) : undefined;
  const profile=returnTo?readProfileRoute(returnTo):undefined;
  const params = new URLSearchParams();
  if (feed) params.set('from', communityFeedHref(feed.topicId, feed.state).slice(2));
  else if(interviews) params.set('from',interviewHref(interviews).slice(2));
  else if(profile)params.set('from',profileFeedHref(profile.username,profile).slice(2));
  else if(returnTo&&readReportHistory(returnTo))params.set('from',returnTo.replace(/^#\/?/,''));
  if (replyId !== undefined) {
    if (!/^[a-f0-9]{32}$/.test(replyId)) throw Error('Invalid reply link');
    params.set('reply',replyId);
  }
  return base + (params.size ? `?${params}` : '');
}

function readCommunityPost(route: string, id: string): {returnTo?:string;replyId?:string}|undefined {
  const value = parts(route);
  if (!value || value.path !== `community/${id}` || !knownParameters(value.params, ['from','reply'])) return;
  const feed = value.params.has('from') ? readCommunityFeed(value.params.get('from') || '') : undefined;
  const interviews=value.params.has('from')?readInterviewRoute(value.params.get('from')||''):undefined;
  const profile=value.params.has('from')?readProfileRoute(value.params.get('from')||''):undefined;
  const reports=readReportHistory(value.params.get('from')||'');
  if(value.params.has('from')&&!feed&&!interviews&&!profile&&!reports)return;
  const replyId=value.params.get('reply')??undefined;
  if(replyId!==undefined&&!/^[a-f0-9]{32}$/.test(replyId))return;
  return {returnTo:feed?communityFeedHref(feed.topicId,feed.state):interviews?interviewHref(interviews):profile?profileFeedHref(profile.username,profile):reports?reportHistoryHref(reports.state,reports.page):undefined,replyId};
}
export function communityPostReturn(route: string, id: string): string | undefined {
  return readCommunityPost(route,id)?.returnTo;
}
export function communityReplyTarget(route:string,id:string):string|undefined {
  return readCommunityPost(route,id)?.replyId;
}
/** Share only the public thread/reply, never feed searches or current query data. */
export function communityShareUrl(pageUrl:string,id:string,replyId?:string):string {
  const page=new URL(pageUrl);
  if(!['http:','https:'].includes(page.protocol)||!/^[a-f0-9]{32}$/.test(id))throw Error('Invalid discussion share URL');
  return new URL(publicHref(communityPostHref(id,undefined,replyId))!,page.origin+page.pathname).href;
}

/** Account continuations accept only canonical community filters and public list return targets. */
export function communityContinuation(route: string): string | undefined {
  const reports=readReportHistory(route);if(reports)return reportHistoryHref(reports.state,reports.page).slice(2);
  const feed = readCommunityFeed(route);
  if (feed) return communityFeedHref(feed.topicId, feed.state).slice(2);
  const value = parts(route);
  const id = value?.path.match(/^community\/([a-f0-9]{32})$/)?.[1];
  if (!id) return;
  const post=readCommunityPost(route,id);
  return post ? communityPostHref(id,post.returnTo,post.replyId).slice(2) : undefined;
}
