import type {PublicUser} from './social';
import type {Post} from './communityDiscussion';

export function isThreadAnonymous(user:PublicUser,postId:string){
  return user.anonymous===true&&user.id==='anonymous:'+postId&&user.username==='';
}
export function ownsCommunityPost(post:Post,owner:string,anonymous=false){
  return !!owner&&!!post.author&&(anonymous?post.anonymous===true&&isThreadAnonymous(post.author,post.id)&&post.viewer?.can_edit===true:!post.author.anonymous&&post.author.id===owner);
}
