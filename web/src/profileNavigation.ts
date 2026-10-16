/** Public profile context contains only a username, activity category and page. */
export type ProfileTab='posts'|'solutions'|'replies'|'likes'|'reposts'|'following'|'followers';
export type ProfileView={username:string;tab:ProfileTab;page:number};
const tabs=['posts','solutions','replies','likes','reposts','following','followers'];
const usernamePattern=/^[a-z0-9_]{3,24}$/;
const maxPage=Math.floor(Number.MAX_SAFE_INTEGER/20);
export function readProfileRoute(route:string):ProfileView|undefined{
 if(route.length>160)return;
 const raw=route.replace(/^#\/?/,''),separator=raw.indexOf('?');
 try{
  const path=decodeURIComponent(separator<0?raw:raw.slice(0,separator));
  const username=/^u\/([a-z0-9_]{3,24})$/.exec(path)?.[1];
  if(!username)return;
  const params=new URLSearchParams(separator<0?'':raw.slice(separator+1));
  if([...params.keys()].some(key=>!['tab','page'].includes(key)||params.getAll(key).length!==1))return;
  const tab=params.get('tab')??'posts',page=params.get('page')??'1';
  if(!tabs.includes(tab)||!/^[1-9]\d*$/.test(page)||!Number.isSafeInteger(Number(page))||Number(page)>maxPage)return;
  return {username,tab:tab as ProfileTab,page:Number(page)};
 }catch{return;}
}
export function profileFeedHref(username:string,state:{tab:ProfileTab;page:number}={tab:'posts',page:1}):string{
 if(!usernamePattern.test(username))throw Error('Invalid profile username');
 const query=new URLSearchParams();
 if(state.tab!=='posts')query.set('tab',state.tab);
 if(state.page!==1)query.set('page',String(state.page));
 const href=`#/u/${username}`+(query.size?'?'+query:'');
 if(!readProfileRoute(href))throw Error('Invalid profile activity context');
 return href;
}
/** Profile list changes keep their existing page position; entering a different profile starts at the top. */
export function sameProfilePage(before:string,after:string):boolean{
 const a=readProfileRoute(before),b=readProfileRoute(after);return !!a&&a.username===b?.username;
}
