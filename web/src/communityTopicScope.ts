import {communityFeedHref,readCommunityFeed} from './communityNavigation';

export const validCommunityTopic=(id:string)=>id.length<=200&&/^[a-z0-9_-]+\/[a-z0-9_-]+$/.test(id);

/** Change only the module scope; keep the current content filters and reset paging. */
export function communityTopicScopeHref(from:string,topicId:string):string|undefined{
 const feed=readCommunityFeed(from);
 if(!feed||topicId&&!validCommunityTopic(topicId))return;
 const same=(feed.topicId||'')===topicId;
 return communityFeedHref(topicId||undefined,{...feed.state,page:same?feed.state.page:1});
}

export function communityTopicKindHref(from:string,solutions:boolean):string|undefined{
 const feed=readCommunityFeed(from);
 if(!feed)return;
 return communityFeedHref(feed.topicId,{...feed.state,solutions,page:1});
}
