import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/community',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__readingFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0')},plugins:[{name:'reading-account-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0reading-account-fixture';},load(id){if(id==='\0reading-account-fixture')return `import React from 'react';export const useAccount=()=>globalThis.__readingFixture.account;export const socialApi=(...args)=>globalThis.__readingFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=()=>React.createElement('span',null,'Avatar');export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;export const profileHref=user=>'#/profile/'+user.username;`;}}]});
let root,model;
let viewport=0,base=1500,scrolls=[];
Object.defineProperty(window,'scrollY',{get:()=>viewport,configurable:true});
window.scrollTo=(options,y)=>{viewport=typeof options==='object'?options.top:y;scrolls.push(viewport);};
dom.window.HTMLElement.prototype.scrollIntoView=function(){viewport=base-100;scrolls.push(viewport);};
dom.window.HTMLElement.prototype.getBoundingClientRect=function(){
 const notice=document.querySelector('.reading-return-notice');
 const noticeHeight=notice?(notice.querySelector('button')?76:42):0;
 const top=base+noticeHeight+(this.dataset?.readingLink==='read'?180:this.dataset?.readingLink==='practice'?240:0)-viewport;
 return {x:0,y:top,left:0,right:400,width:400,top,bottom:top+30,height:30,toJSON(){return this;}};
};
try{
 const {createRoot}=await import('react-dom/client');
 model=await server.ssrLoadModule('/src/readingReturn.ts');
 const {useReadingPosition,useReadingOrigin}=await server.ssrLoadModule('/src/useReadingPosition.ts');
 const {default:Notice}=await server.ssrLoadModule('/src/ReadingReturnNotice.tsx');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const nav=await server.ssrLoadModule('/src/communityNavigation.ts');
 const library=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const {usePracticeReadingOrigin}=await server.ssrLoadModule('/src/usePracticeReadingOrigin.ts');
 const interviews=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const owner='account:alice',id='a'.repeat(32),other='b'.repeat(32);
 const community=nav.communityFeedHref('architecture/attention',{category:'questions',following:false,saved:true,search:'mask & 梯度 😀',page:3});
 const activeCommunity=nav.communityFeedHref('architecture/attention',{category:'questions',following:false,saved:true,sort:'active',search:'mask & 梯度 😀',page:3});
 const interview=interviews.interviewHref({...interviews.emptyInterviewFilters(),company:'google',role:'llm',stage:'technical',days:'90',saved:true,q:'attention',concept:'architecture/attention',page:2});
 const wait=()=>new Promise(resolve=>setTimeout(resolve,5));
 const navigate=async href=>act(async()=>{location.hash=href;viewport=0;await wait();});
 function LocaleControl(){const {locale,setLocale}=useI18n();return React.createElement('button',{id:'locale',onClick:()=>setLocale(locale==='en'?'zh':'en')},'Locale');}
 let lastHook;
 function List({scope=community,account=owner,ready=true,items=[id,other],prevent=false,practiceIds=['architecture/attention','rl/gae']}){
  const reading=useReadingPosition(account===null?undefined:account,scope,ready);lastHook=reading;
  return React.createElement('main',{ref:reading.container},React.createElement(LocaleControl),React.createElement('input',{id:'search','aria-label':'Search'}),React.createElement('p',{'data-reading-list':'',tabIndex:-1},'List results'),React.createElement(Notice,{notice:reading.notice,onRestore:reading.restore}),items.map(post=>React.createElement('article',{key:post,'data-reading-post':post,'data-reading-resumed':reading.resumedPost===post||undefined},['title','read'].map(link=>React.createElement('a',{key:link,'data-reading-link':link,href:nav.communityPostHref(post,scope),onClick:prevent?event=>event.preventDefault():undefined},link+' '+post)),practiceIds.map(practice=>React.createElement('a',{key:practice,'data-reading-link':'practice','data-reading-practice':practice,href:library.withLibraryContext(practice,scope)},practice)))));
 }
 function Practice({scope=interview,account=owner,problem='architecture/attention',number=1}){usePracticeReadingOrigin(account===null?undefined:account,{id:problem,number},scope);return React.createElement('textarea',{id:'learner-code',defaultValue:'Private learner code'});}
 function Thread({scope=community,account=owner,post=id}){useReadingOrigin(account,post,scope);return React.createElement('div',null,'Full post');}
 const render=async(Component,props={})=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Component,props)));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 // Prevent only the test document's native navigation after observers ran; the
 // controller below drives the same hashchange traversal as the real router.
 const activate=async(element,options={})=>act(async()=>{
  assert(element,'Missing activation target');
  if(options.keyboard)element.focus();
  document.addEventListener('click',event=>event.preventDefault(),{once:true});
  element.dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true,cancelable:true,button:0,detail:options.keyboard?0:1,...options}));
 });
 const anchor=(post=id,link='title')=>document.querySelector(`[data-reading-post="${post}"] [data-reading-link="${link}"]`);
 async function depart(scope=community,{keyboard=true,link='title'}={}){
  model.discardReadingReturn();await navigate(scope);base=1500;viewport=1250;await render(List,{scope});await activate(anchor(id,link),{keyboard});
  const record=model.currentReadingReturn();assert(record);assert.equal(record.href,scope);assert.equal(record.phase,'opening');assert.equal(record.focus,keyboard);
  await stop();await navigate(nav.communityPostHref(id,scope));await render(Thread,{scope});assert.equal(model.currentReadingReturn().phase,'reading');await stop();return record;
 }
 async function returnLoading(scope=community,props={}){await navigate(scope);base=1800;await render(List,{scope,ready:false,...props});assert.equal(scrolls.length,0,'A pending response cannot restore against missing or stale rows');}
 for(const scope of [community,activeCommunity,interview,'#/u/alice?tab=solutions&page=2','#/u/alice?tab=interviews&page=2']){
  for(const link of ['title','read']){
   await depart(scope,{link});scrolls=[];await returnLoading(scope);assert.equal(model.currentReadingReturn().phase,'returning');
   await render(List,{scope,ready:true});assert.equal(scrolls.length,1);assert.equal(viewport,1592,'Measure after the status notice is committed, retaining the exact viewport offset');
   assert.equal(document.activeElement,anchor(id,link),'Keyboard return restores the exact originating action');assert.equal(anchor(id,link).getBoundingClientRect().top,link==='read'?430:250);
   assert(document.querySelector('[data-reading-resumed="true"]'));assert(document.querySelector('[role="status"]').textContent.includes('Back at'));assert.equal(model.currentReadingReturn(),undefined);assert.equal(sessionStorage.getItem(model.READING_RETURN_KEY),null);
   await render(List,{scope,ready:true});assert.equal(scrolls.length,1,'Routine rerenders never repeat the return');await act(async()=>document.querySelector('#locale').click());assert(document.querySelector('[role="status"]').textContent.includes('已回到'));assert.equal(scrolls.length,1,'Language changes keep the reading position');await act(async()=>document.querySelector('#locale').click());await stop();
  }
 }
 await depart(community,{keyboard:false});scrolls=[];await returnLoading();await render(List);assert.equal(document.activeElement,document.body,'Pointer returns preserve focus rather than forcing a link');await stop();
 // Scrolling, typing and focus while data is loading opt out of automatic movement.
 for(const intent of ['wheel','touchmove','pointerdown','keydown','focusin','scroll']){
  await depart();scrolls=[];await returnLoading();
  await act(async()=>{if(intent==='focusin')document.querySelector('#search').focus();else{if(intent==='scroll')viewport=400;window.dispatchEvent(new dom.window.Event(intent));}});
  await render(List);assert.equal(scrolls.length,0,intent+' keeps the current viewport');assert(document.querySelector('.reading-return-notice button'));
  await activate(document.querySelector('.reading-return-notice button'));assert.equal(scrolls.length,1);assert.equal(viewport,1592,'An explicit return accounts for the disappearing manual control');assert.equal(document.activeElement,anchor());await stop();
 }
 await depart();scrolls=[];await returnLoading();await act(async()=>document.querySelector('#search').focus());await render(List);
 await act(async()=>{document.querySelector('.reading-return-notice button').click();document.querySelector('#search').focus();});assert.equal(scrolls.length,0,'A later outside focus cancels even an explicitly queued return');assert.equal(document.activeElement,document.querySelector('#search'));assert(document.querySelector('.reading-return-notice button'));await stop();
 // A queued manual action cannot cross a synchronous owner or route replacement.
 await depart();scrolls=[];await returnLoading();await act(async()=>document.querySelector('#search').focus());await render(List);
 await act(async()=>{document.querySelector('.reading-return-notice button').click();root.render(React.createElement(LocaleProvider,null,React.createElement(List,{account:'account:bob'})));});assert.equal(scrolls.length,0,'Owner changes invalidate a queued layout restoration before passive effects');assert(!document.querySelector('.reading-return-notice'));await stop();
 await depart();scrolls=[];await returnLoading();await act(async()=>document.querySelector('#search').focus());await render(List);const queuedRoute=nav.communityFeedHref(undefined,nav.emptyCommunityFeed());
 await act(async()=>{document.querySelector('.reading-return-notice button').click();location.hash=queuedRoute;root.render(React.createElement(LocaleProvider,null,React.createElement(List,{scope:queuedRoute})));await wait();});assert.equal(scrolls.length,0,'A changed route cannot apply an older queued reading anchor');assert(!document.querySelector('.reading-return-notice'));await stop();
 // Empty or changed lists announce the missing post, with keyboard focus on the list.
 await depart();scrolls=[];await returnLoading();await render(List,{items:[other]});assert(document.querySelector('[role="status"]').textContent.includes('no longer on this page'));assert.equal(document.activeElement,document.querySelector('[data-reading-list]'));assert(!document.querySelector('[data-reading-resumed]'));assert.equal(model.currentReadingReturn(),undefined);await stop();
 await depart();scrolls=[];await returnLoading();await render(List,{items:[]});assert(document.querySelector('[role="status"]').textContent.includes('no longer on this page'));await stop();
 // Account, readiness and filter changes invalidate all stale callbacks synchronously.
 await depart();scrolls=[];await returnLoading();const staleRestore=lastHook.restore;await render(List,{account:'account:bob'});assert.equal(scrolls.length,0);assert(!document.querySelector('.reading-return-notice'));assert.equal(model.currentReadingReturn(),undefined);await act(async()=>staleRestore(document.querySelector('#search')));assert.equal(scrolls.length,0);await stop();
 await depart();scrolls=[];await returnLoading(community,{account:null});await act(async()=>document.querySelector('#search').focus());await render(List,{account:owner});assert.equal(scrolls.length,0,'Resolving an account cannot erase an interaction made while waiting');assert(document.querySelector('.reading-return-notice button'));await stop();
 await depart();scrolls=[];await returnLoading();const changed=nav.communityFeedHref(undefined,nav.emptyCommunityFeed());await navigate(changed);await render(List,{scope:changed});assert.equal(scrolls.length,0);assert(!document.querySelector('.reading-return-notice'));assert.equal(model.currentReadingReturn(),undefined);await stop();
 // Direct views and unrelated routes never inherit a prior reading position.
 model.discardReadingReturn();await navigate(community);await render(List);
 for(const options of [{ctrlKey:true},{metaKey:true},{shiftKey:true},{altKey:true},{button:1}]){await activate(anchor(),options);assert.equal(model.currentReadingReturn(),undefined,'Modified/new-tab activation does not arm a return');}
 await render(List,{prevent:true});await activate(anchor());assert.equal(model.currentReadingReturn(),undefined,'A cancelled navigation is not recorded');
 await render(List,{ready:false});await activate(anchor());assert.equal(model.currentReadingReturn(),undefined,'Stale visible cards cannot record a pending or failed response');await stop();
 await depart();await navigate('#/settings');await navigate(community);scrolls=[];await render(List);assert.equal(scrolls.length,0);assert.equal(model.currentReadingReturn(),undefined);await stop();
 // A reply link inside the originating thread retains the same list return.
 await depart();await navigate(nav.communityPostHref(id,community,other));assert.equal(model.currentReadingReturn().phase,'reading');scrolls=[];await returnLoading();await render(List);assert.equal(scrolls.length,1);await stop();
 // Exercise the real feed components and their API readiness gates as well.
 const {default:Hub}=await server.ssrLoadModule('/src/Interviews.tsx'),{default:Community}=await server.ssrLoadModule('/src/Community.tsx');
 const requests=[];globalThis.__readingFixture={account:{user:{id:'alice',username:'alice',display_name:'Alice'},loading:false,capabilities:{community_bookmarks:true}},request(path,body,signal){assert.equal(body,undefined,'Reading returns never mutate posts, saved state or account data');return new Promise(resolve=>requests.push({path,resolve,signal}));}};
 const take=fragment=>{const index=requests.findIndex(request=>request.path.includes(fragment));assert(index>=0,'Missing request '+fragment);return requests.splice(index,1)[0];};
 const reply=async(request,value)=>act(async()=>request.resolve(value));
 const post={id,title:'A real feed reflection',body:'I clarified the requirements and checked the edge cases.',category:'career',created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',author:{id:'author',username:'author',display_name:'Author',avatar_color:'#48766b'},counts:{replies:0,likes:0,reposts:0},viewer:{bookmarked:true},interview:{company:'google',role:'llm',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:['architecture/attention']}};
 const summary={reports:0,contributors:0,days:'90',basis:'published-community-reports',generated_at:'2026-10-15',companies:[],questions:[]};
 for(const scope of [interview,community]){
  const isInterview=scope===interview,Component=isInterview?Hub:Community,props={route:scope.slice(2),...(isInterview?{problems:[{id:'architecture/attention',number:1,title:'Attention',starter:'',module:{optional:false}}],progressReady:false}:{topicId:'architecture/attention',topic:{id:'architecture/attention',title:'Attention'}})};
  const feedFragment=isInterview?'community/interviews?':'community/posts?';
  const resolveFeed=async()=>{if(isInterview)await reply(take('/summary'),summary);await reply(take(feedFragment),{items:[post],total:100,limit:20,offset:isInterview?20:40,owner_id:'alice'});};
  model.discardReadingReturn();await navigate(scope);base=1500;viewport=1250;await render(Component,props);await resolveFeed();await activate(anchor(),{keyboard:true});assert.equal(model.currentReadingReturn().href,scope);await stop();
  const thread=nav.communityPostHref(id,scope);await navigate(thread);await render(Community,{postId:id,route:thread.slice(2)});assert.equal(model.currentReadingReturn().phase,'reading');await stop();
  await navigate(scope);base=1800;scrolls=[];await render(Component,props);assert.equal(scrolls.length,0,'Actual feeds wait for their matching current-owner response');await resolveFeed();assert.equal(scrolls.length,1);assert.equal(viewport,1592);assert.equal(document.activeElement,anchor());assert(document.querySelector('[data-reading-resumed="true"]'));
  if(isInterview){
   base=1500;viewport=1250;const link=document.querySelector('[data-reading-practice="architecture/attention"]');assert(link);await activate(link,{keyboard:true});const origin=model.currentReadingReturn();assert.equal(origin.link,'practice');await stop();await navigate(library.withLibraryContext('architecture/attention',scope));await render(Practice,{scope});assert.equal(model.currentReadingReturn().phase,'reading');await stop();await navigate(scope);base=1800;scrolls=[];await render(Component,props);await resolveFeed();assert.equal(scrolls.length,1);const restored=document.querySelector('[data-reading-practice="architecture/attention"]');assert.equal(document.activeElement,restored);assert.equal(restored.getBoundingClientRect().top,origin.top);
  }
  await stop();requests.length=0;
 }

 // Actual practice navigation keeps the origin chip across next-module navigation and delayed lists.
 const practiceAnchor=problem=>document.querySelector('[data-reading-post="'+id+'"] [data-reading-practice="'+problem+'"]');
 async function departPractice({keyboard=true,problem='architecture/attention'}={}){
  model.discardReadingReturn();await navigate(interview);base=1500;viewport=1250;await render(List,{scope:interview});await activate(practiceAnchor(problem),{keyboard});
  const row=model.currentReadingReturn();assert.equal(row.link,'practice');assert.equal(row.practice,problem);assert.equal(row.phase,'opening');await stop();
  await navigate(library.withLibraryContext(problem,interview));await render(Practice,{problem,number:problem==='rl/gae'?2:1});assert.equal(model.currentReadingReturn().phase,'reading');assert.equal(document.querySelector('#learner-code').value,'Private learner code');await stop();return row;
 }
 for(const keyboard of [true,false]){
  await departPractice({keyboard});await navigate(library.withLibraryContext('rl/gae',interview));await render(Practice,{problem:'rl/gae',number:2});assert.equal(model.currentReadingReturn().practice,'architecture/attention');assert.equal(model.currentReadingReturn().phase,'reading');await stop();scrolls=[];await returnLoading(interview);await render(List,{scope:interview});
  assert.equal(scrolls.length,1);assert.equal(viewport,1592);assert.equal(practiceAnchor('architecture/attention').getBoundingClientRect().top,490);assert.equal(document.activeElement,keyboard?practiceAnchor('architecture/attention'):document.body);await stop();
 }
 await departPractice({problem:'rl/gae'});scrolls=[];await returnLoading(interview);await render(List,{scope:interview});assert.equal(document.activeElement,practiceAnchor('rl/gae'),'Return to the exact second originating practice chip');await stop();
 // New user intent retains the current viewport and offers an explicit return instead.
 await departPractice();scrolls=[];await returnLoading(interview);await act(async()=>document.querySelector('#search').focus());await render(List,{scope:interview});assert.equal(scrolls.length,0);assert(document.querySelector('.reading-return-notice button'));await activate(document.querySelector('.reading-return-notice button'));assert.equal(document.activeElement,practiceAnchor('architecture/attention'));assert.equal(scrolls.length,1);await stop();
 // A removed/hidden practice chip falls back to the original card title, never a different practice.
 await departPractice();scrolls=[];await returnLoading(interview);await render(List,{scope:interview,practiceIds:['rl/gae']});assert.equal(document.activeElement,anchor(id,'title'));assert.equal(scrolls.length,1);await stop();
 await departPractice();scrolls=[];await returnLoading(interview);await render(List,{scope:interview,items:[other]});assert(document.querySelector('.reading-return-notice').textContent.includes('no longer on this page'));assert.equal(document.activeElement,document.querySelector('[data-reading-list]'));await stop();
 // Thread -> practice -> next -> list keeps the original thread action as the return target.
 await depart(interview,{link:'read'});await navigate(library.withLibraryContext('architecture/attention',interview));await render(Practice);assert.equal(model.currentReadingReturn().link,'read');await stop();await navigate(library.withLibraryContext('2',interview));await render(Practice,{problem:'rl/gae',number:2});await stop();scrolls=[];await returnLoading(interview);await render(List,{scope:interview});assert.equal(document.activeElement,anchor(id,'read'));await stop();
 // Late effects from an outgoing module cannot erase a completed return or a newer practice journey.
 await departPractice();await navigate(interview);model.resumePracticeReadingOrigin(owner,{id:'architecture/attention',number:1},interview,library.withLibraryContext('architecture/attention',interview));assert.equal(model.currentReadingReturn().phase,'returning');model.discardReadingReturn();
 await departPractice();await navigate(library.withLibraryContext('rl/gae',interview));model.resumePracticeReadingOrigin(owner,{id:'architecture/attention',number:1},interview,library.withLibraryContext('architecture/attention',interview));assert.equal(model.currentReadingReturn().phase,'reading');model.discardReadingReturn();
 // Reload uses persisted, bounded metadata and a real catalog identity; no code is persisted.
 const practiceRecord=await departPractice();const practiceRaw=sessionStorage.getItem(model.READING_RETURN_KEY);assert(!practiceRaw.includes('Private learner code'));assert.deepEqual(Object.keys(JSON.parse(practiceRaw)).sort(),['version','id','owner','href','post','link','practice','top','focus','at','phase'].sort());model.discardReadingReturn();sessionStorage.setItem(model.READING_RETURN_KEY,practiceRaw);
 const practiceFresh=await server.ssrLoadModule('/src/readingReturn.ts?fresh=practice');practiceFresh.resumePracticeReadingOrigin(owner,{id:'architecture/attention',number:1},interview);assert.equal(practiceFresh.currentReadingReturn().phase,'reading');await navigate(library.withLibraryContext(2,interview));assert.equal(practiceFresh.currentReadingReturn().phase,'reading');await navigate(interview);assert.equal(practiceFresh.currentReadingReturn().phase,'returning');practiceFresh.discardReadingReturn();
 for(const [label,account,entry] of [['wrong-owner','account:bob',{id:'architecture/attention',number:1}],['wrong-entry',owner,{id:'rl/gae',number:2}]]){
  await navigate(library.withLibraryContext('architecture/attention',interview));sessionStorage.setItem(model.READING_RETURN_KEY,practiceRaw);const fresh=await server.ssrLoadModule('/src/readingReturn.ts?fresh='+label);fresh.resumePracticeReadingOrigin(account,entry,interview);assert.equal(fresh.currentReadingReturn(),undefined);fresh.discardReadingReturn();
 }
 // Unrelated scopes, malformed continuations, paper pages and spoofed target metadata cannot arm or extend a return.
 for(const route of ['#/papers/attention?from='+encodeURIComponent(interview.slice(2)),library.withLibraryContext('architecture/attention',interviews.interviewHref({...interviews.emptyInterviewFilters(),company:'amazon'})),library.withLibraryContext('architecture/attention',interview)+'&extra=1','#/account']){
  await departPractice();await navigate(route);assert.equal(model.currentReadingReturn(),undefined);await stop();
 }
 model.discardReadingReturn();await navigate(interview);await render(List,{scope:interview});const spoof=practiceAnchor('architecture/attention');spoof.dataset.readingPractice='rl/gae';await activate(spoof);assert.equal(model.currentReadingReturn(),undefined);await stop();
 for(const patch of [{link:'practice',practice:undefined},{link:'practice',practice:'javascript:alert(1)'},{link:'title',practice:'architecture/attention'},{link:'practice',practice:'architecture/attention',href:community}]){
  model.rememberReadingReturn({...practiceRecord,...patch});assert.equal(model.currentReadingReturn(),undefined);
 }

 // Full thread -> practice -> canonical thread preserves its earlier list only in this owner/tab.
 const sourcePost=nav.communityPostHref(id);
 for(const source of [community,activeCommunity,interview]){
  await depart(source);const record=model.currentReadingReturn();
  for(const path of ['architecture/attention','1','papers/attention']){
   await navigate(library.withLibraryContext(path,sourcePost));
   assert.equal(model.currentReadingReturn()?.id,record.id);assert.equal(model.currentReadingReturn()?.phase,'reading');
   if(!path.startsWith('papers/')){await render(Practice,{scope:sourcePost});assert.equal(model.currentReadingReturn()?.id,record.id);await stop();}
   await navigate(sourcePost);assert.equal(model.sourcePostReadingReturn(owner,id,sourcePost),source);assert.equal(model.sourcePostReadingReturn('account:bob',id,sourcePost),undefined);
   await render(Thread,{scope:source});assert.equal(model.currentReadingReturn()?.id,record.id);await stop();
  }
  scrolls=[];await navigate(source);await render(List,{scope:source});assert.equal(model.currentReadingReturn(),undefined);assert(document.activeElement===anchor());await stop();
 }
 await depart(community);await navigate(library.withLibraryContext(1,sourcePost));model.resumePracticeReadingOrigin(owner,{id:'rl/gae',number:2},sourcePost);assert.equal(model.currentReadingReturn(),undefined,'The actual catalog entry must match the source-post practice route');await stop();
 await depart(community);await navigate(library.withLibraryContext(1,sourcePost));model.resumePracticeReadingOrigin('account:bob',{id:'architecture/attention',number:1},sourcePost);assert.equal(model.currentReadingReturn(),undefined,'Changing owners drops the old private list origin');await stop();
 const sourceRecord=await depart(community);await navigate(library.withLibraryContext(1,sourcePost));const sourceRaw=sessionStorage.getItem(model.READING_RETURN_KEY);model.discardReadingReturn();sessionStorage.setItem(model.READING_RETURN_KEY,sourceRaw);
 const sourceReload=await server.ssrLoadModule('/src/readingReturn.ts?fresh=source-post-practice');sourceReload.resumePracticeReadingOrigin(owner,{id:'architecture/attention',number:1},sourcePost);assert.equal(sourceReload.currentReadingReturn()?.id,sourceRecord.id);await navigate(sourcePost);assert.equal(sourceReload.sourcePostReadingReturn(owner,id,sourcePost),community);await navigate(community);assert.equal(sourceReload.currentReadingReturn()?.phase,'returning');sourceReload.discardReadingReturn();
 // A queued hashchange from an older document route must not erase a newly armed source-post journey.
 for(const scope of [community,activeCommunity,interview,'#/u/alice?tab=solutions&page=2'])for(const phase of ['opening','reading']){
  model.discardReadingReturn();window.history.replaceState(null,'',scope);model.rememberReadingReturn({owner,href:scope,post:id,link:'title',top:240,focus:true});
  const practice=library.withLibraryContext(1,nav.communityPostHref(id));window.history.replaceState(null,'',practice);
  if(phase==='reading')model.resumePracticeReadingOrigin(owner,{id:'architecture/attention',number:1},nav.communityPostHref(id),practice);
  const newer=model.currentReadingReturn(),stored=sessionStorage.getItem(model.READING_RETURN_KEY),position=viewport;assert(newer);assert.equal(newer.phase,phase);
  window.dispatchEvent(new dom.window.HashChangeEvent('hashchange',{oldURL:new URL(library.withLibraryContext(999,nav.communityPostHref(other)),location.href).href,newURL:new URL(library.withLibraryContext('papers/attention',nav.communityPostHref(other)),location.href).href}));
  assert.deepEqual(model.currentReadingReturn(),newer,'Queued stale hashchange must keep the newer journey');assert.equal(sessionStorage.getItem(model.READING_RETURN_KEY),stored);assert.equal(location.hash,practice);assert.equal(viewport,position);
  model.resumePracticeReadingOrigin(owner,{id:'architecture/attention',number:1},nav.communityPostHref(id),practice);assert.equal(model.currentReadingReturn().phase,'reading');
  window.history.replaceState(null,'',scope);window.dispatchEvent(new dom.window.HashChangeEvent('hashchange',{oldURL:new URL(practice,location.href).href,newURL:new URL(scope,location.href).href}));assert.equal(model.currentReadingReturn().phase,'returning','An event matching the current URL still returns to the exact source list');
  window.history.replaceState(null,'','#/settings');window.dispatchEvent(new dom.window.HashChangeEvent('hashchange',{oldURL:new URL(scope,location.href).href,newURL:location.href}));assert.equal(model.currentReadingReturn(),undefined,'Actual navigation away still retires a return');assert.equal(sessionStorage.getItem(model.READING_RETURN_KEY),null);
 }
 // Storage denial leaves an entirely local return available in the current tab.
 const storage=globalThis.sessionStorage;Object.defineProperty(globalThis,'sessionStorage',{configurable:true,value:{getItem(){throw Error('Blocked')},setItem(){throw Error('Blocked')},removeItem(){throw Error('Blocked')}}});
 await depart();scrolls=[];await returnLoading();await render(List);assert.equal(scrolls.length,1);await stop();Object.defineProperty(globalThis,'sessionStorage',{configurable:true,value:storage});
 // Reload the model module to exercise only persisted data, with no memory cache.
 const persisted=await depart();const raw=sessionStorage.getItem(model.READING_RETURN_KEY);model.discardReadingReturn();sessionStorage.setItem(model.READING_RETURN_KEY,raw);
 const fresh=await server.ssrLoadModule('/src/readingReturn.ts?fresh=reload');fresh.resumeReadingOrigin(owner,id,community);assert.equal(fresh.currentReadingReturn().phase,'reading');await navigate(community);assert.equal(fresh.currentReadingReturn().phase,'returning','Reloading the actual thread preserves its scoped return');fresh.discardReadingReturn();
 sessionStorage.setItem(model.READING_RETURN_KEY,JSON.stringify({...persisted,phase:'returning'}));const unresolved=await server.ssrLoadModule('/src/readingReturn.ts?fresh=unresolved');unresolved.keepReadingOwner(undefined);await navigate('#/settings');assert.equal(unresolved.currentReadingReturn(),undefined,'Navigation during account resolution invalidates a persisted return before any late identity can apply it');unresolved.discardReadingReturn();
 sessionStorage.setItem(model.READING_RETURN_KEY,'{broken');const corrupt=await server.ssrLoadModule('/src/readingReturn.ts?fresh=corrupt');assert.equal(corrupt.currentReadingReturn(),undefined);corrupt.discardReadingReturn();
 for(const patch of [{href:'https://evil.test'},{owner:'account:../escape'},{post:'x'},{top:Infinity},{top:100001},{focus:'yes'},{body:'Never persist draft text'}]){
  model.rememberReadingReturn({...persisted,...patch});assert.equal(model.currentReadingReturn(),undefined,'Invalid persisted metadata is rejected');
 }
 sessionStorage.setItem(model.READING_RETURN_KEY,JSON.stringify({...persisted,phase:'reading',at:Date.now()-7*60*60*1000}));const expired=await server.ssrLoadModule('/src/readingReturn.ts?fresh=expired');assert.equal(expired.currentReadingReturn(),undefined);expired.discardReadingReturn();
 await navigate(community);await render(List);await activate(anchor(),{keyboard:true});const stored=JSON.parse(sessionStorage.getItem(model.READING_RETURN_KEY));assert.deepEqual(Object.keys(stored).sort(),['version','id','owner','href','post','link','top','focus','at','phase'].sort());await stop();model.discardReadingReturn();
 console.log('PASS reading returns: exact community/interview/search/saved/page context, pointer and keyboard actions, delayed data, notice-aware layout, manual recovery without stolen focus, missing/empty lists, account/filter isolation, modified/prevented navigation, reply routes, bounded reloadable metadata, superseded queued-hash rejection with current-navigation retirement, blocked storage and expired/corrupt data; direct practice, next-module/numeric traversal, exact chip/title fallback, private-code-free reload, real catalog binding and account/scope isolation');
}finally{
 if(root)await act(async()=>root.unmount());model?.discardReadingReturn();await server.close();dom.window.close();for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
