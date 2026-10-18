import AppLink from './AppLink';
import {useI18n} from './i18n';
import {useCommunityTaxonomy} from './useCommunityTaxonomy';
import {communityFeedHref,readCommunityFeed,emptyCommunityFeed} from './communityNavigation';
import type {Post} from './communityDiscussion';
import './community-taxonomy.css';
export default function PostTags({post,returnTo}:{post:Post;returnTo?:string}){
 const {t}=useI18n(),taxonomy=useCommunityTaxonomy();
 if(!post.tags?.length)return null;
 const feed=readCommunityFeed(returnTo||''),filters=feed?.state||emptyCommunityFeed();
 return <nav className="community-post-tags" aria-label={t('帖子标签','Post tags')}>{post.tags.map(item=>{const tag=taxonomy.data.tags.find(row=>row.id===item.id)||item;return <AppLink key={tag.id} href={communityFeedHref(feed?.topicId,{...filters,tag:tag.id,page:1})}>#{t(tag.zh,tag.en)}</AppLink>;})}</nav>;
}
