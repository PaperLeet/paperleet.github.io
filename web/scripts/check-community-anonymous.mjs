import assert from 'node:assert/strict';
import React,{act} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/community?category=career',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,requestAnimationFrame:dom.window.requestAnimationFrame.bind(dom.window),IS_REACT_ACT_ENVIRONMENT:true,__anonFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');let errors=[];dom.window.addEventListener('error',event=>errors.push(event.error));
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'anonymous-account-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0anonymous-social';},load(id){if(id==='\0anonymous-social')return `import React from 'react';export const useAccount=()=>globalThis.__anonFixture.account;export const socialApi=(...args)=>globalThis.__anonFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=({user})=>React.createElement('span',{'data-avatar-url':user.avatar_url||'',className:'fixture-avatar'},user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('a',{className:'person-link',href:profileHref(user)},user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Community,UserProfile}=await server.ssrLoadModule('/src/Community.tsx');
 const {default:InterviewCard}=await server.ssrLoadModule('/src/InterviewReportCard.tsx');
 const {default:Composer}=await server.ssrLoadModule('/src/InterviewComposer.tsx');
 const {default:Author,CommunityAuthorAvatar}=await server.ssrLoadModule('/src/CommunityAuthor.tsx');
 const drafts=await server.ssrLoadModule('/src/communityDrafts.ts'),receipts=await server.ssrLoadModule('/src/publicationRequests.ts'),interviews=await server.ssrLoadModule('/src/interviewDrafts.ts');
 const {canEditPost}=await server.ssrLoadModule('/src/communityEdits.ts'),{canEditReply}=await server.ssrLoadModule('/src/replyEdits.ts');
 const affiliation={kind:'institution',domain:'research.edu',name:'测试研究院',name_en:'Research Institute'};
 const alice={can_publish_community:true,id:'alice',username:'alice',display_name:'Private Alice',avatar_color:'#123456',avatar_url:'/api/account/avatars/alice/personal.png',bio:'Private biography',created_at:'2020-01-01',affiliation};
 const postId='a'.repeat(32),anonymous={...alice,id:'anonymous:'+postId,username:'',display_name:'Anonymous',anonymous:true,avatar_url:null,avatar_color:'#6b7280',bio:'',created_at:''};
 const named={id:'b'.repeat(32),title:'Named report',body:'A public reflection',category:'career',problem:null,problem_id:null,created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',author:alice,counts:{likes:0,reposts:0,replies:1},viewer:{can_edit:true},edit_version:'e'.repeat(64)};
 const reply={id:'c'.repeat(32),post_id:postId,parent_id:null,body:'Anonymous follow-up',created_at:named.created_at,author:anonymous,can_edit:true,edit_version:'d'.repeat(64)};
 const post={...named,id:postId,anonymous:true,author:anonymous,replies:[reply]};
 let mode='read',lost=false,attempts=[],created=0,published=[],holdProfile=false,held=[];
 const account={user:alice,loading:false,capabilities:{community_anonymous_posts:true},refresh:async()=>{}};
 globalThis.__anonFixture={account,request:async(path,payload)=>{
  if(payload){assert.equal(path,'community/posts/anonymous','Anonymous intent must never fall back to the legacy named endpoint');assert.equal(payload.anonymous,true);assert.equal(payload.expected_user_id,alice.id);attempts.push(payload);if(attempts.length===1)created++;if(lost){lost=false;throw Error('Response lost after publication');}return {...post,...(payload.interview?{interview:payload.interview}:{})};}
  if(path.startsWith('community/profiles/')){if(holdProfile)return new Promise(resolve=>held.push(resolve));return path.includes('/activity')?{items:[{kind:'posts',post}],total:1}:{...alice,counts:{posts:1,replies:1,likes:0,reposts:0,followers:0,following:0,likes_received:0,reposts_received:0},viewer:{is_self:true}};}
  return path.startsWith('community/posts?')?{items:[mode==='named'?named:post],total:1}:post;
 }};
 const render=async(Component,props={})=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Component,props)));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const click=async node=>{assert(node);await act(async()=>node.click());};
 const submit=async selector=>act(async()=>document.querySelector(selector).dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
 const draft={title:'Private reflection',body:'Actual interview reflections',category:'career'};
 // Real card/discussion identities retain affiliation but expose no anonymous profile links.
 await render(Community,{route:'community?category=career'});assert(document.querySelector('.post-card-meta').textContent.includes('Research Institute'));assert(document.querySelector('.post-card-meta').textContent.includes('Anonymous'));assert(!document.querySelector('.community-post-card a[href^="#/u/"]'));await stop();
 mode='named';await render(Community,{route:'community?category=career'});assert(document.querySelector('.community-post-card a[href="#/u/alice"]'));assert(document.querySelector('.post-card-meta').textContent.includes('Research Institute'));await stop();mode='read';
 await render(Community,{postId,route:'community/'+postId});assert.equal(document.querySelectorAll('.discussion-post .institution-badge').length,1);assert(document.querySelector('.community-reply .institution-badge'));assert(!document.querySelector('.discussion-post a[href^="#/u/"]'));assert(!document.querySelector('.community-reply a[href^="#/u/"]'));assert(document.querySelector('#edit-reply-'+reply.id));assert(!document.querySelector('.reply-composer a[href="#/u/alice"]'),'The anonymous author sees their anonymous reply identity');await stop();
 const unsafe={...alice,anonymous:true};const html=renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(React.Fragment,null,React.createElement(Author,{user:unsafe}),React.createElement(CommunityAuthorAvatar,{user:unsafe}))));assert(!html.includes('Private Alice')&&!html.includes('personal.png')&&!html.includes('#/u/alice'));assert(html.includes('Research Institute'));
 assert(canEditPost(post,'alice'));assert(!canEditPost({...post,viewer:{can_edit:false}},'bob'));assert(canEditReply(reply,'alice'));assert(!canEditReply({...reply,can_edit:false},'bob'));
 // Draft/receipt identity is durable, owner-scoped, strict and part of retry equality.
 assert(drafts.writePostDraft(sessionStorage,alice.id,'general',draft));assert(drafts.writePostDraft(sessionStorage,alice.id,'general',{...draft,anonymous:true}));assert(!drafts.clearPublishedDraft(sessionStorage,alice.id,'general',draft));assert.equal(drafts.readPostDraft(sessionStorage,alice.id,'general').anonymous,true);assert.equal(drafts.readPostDraft(sessionStorage,'bob','general'),undefined);
 const payload=receipts.postPublication({...draft,anonymous:true});const pending=receipts.preparePublication(sessionStorage,alice.id,'post','general',payload);assert.equal(pending.receipt.payload.anonymous,true);assert(!receipts.samePublication(pending.receipt,receipts.postPublication(draft)));assert.equal(receipts.readPublication(sessionStorage,alice.id,'post','general').receipt.requestId,pending.receipt.requestId);receipts.clearPublication(sessionStorage,pending.receipt);
 // Losing an anonymous response retries the exact identity and request ID; no named fallback.
 lost=true;await render(Community,{route:'community?category=career'});await click(document.querySelector('.community-compose-button'));assert(document.querySelector('.publishing-identity input').checked);assert(document.querySelector('.publishing-identity').textContent.includes('Research Institute'));const source=document.querySelector('.community-composer textarea');source.setSelectionRange(2,8);await submit('.community-composer form');assert(document.body.textContent.includes('Response lost'));assert.equal(drafts.readPostDraft(sessionStorage,alice.id,'general').anonymous,true);assert.equal(document.querySelector('.community-composer textarea'),source);await submit('.community-composer form');assert.equal(attempts.length,2);assert.equal(attempts[0].request_id,attempts[1].request_id);assert.equal(created,1);assert.equal(drafts.readPostDraft(sessionStorage,alice.id,'general'),undefined);await stop();
 // A restored anonymous draft is blocked on older services until explicitly changed.
 attempts=[];account.capabilities={};drafts.writePostDraft(sessionStorage,alice.id,'general',{...draft,anonymous:true});await render(Community,{route:'community?category=career'});await click(document.querySelector('.community-compose-button'));assert(document.querySelector('.community-composer .form-footer>.button').disabled);await submit('.community-composer form');assert.equal(attempts.length,0);assert(document.body.textContent.includes('Anonymous posting is unavailable'));await click(document.querySelector('.publishing-identity input'));assert(!document.querySelector('.community-composer .form-footer>.button').disabled);assert(!drafts.readPostDraft(sessionStorage,alice.id,'general').anonymous);await stop();
 // Anonymous interview confirmation accepts only the server-authorized thread identity.
 account.capabilities={community_anonymous_posts:true};attempts=[];const interview={company:'google',role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:[]};interviews.writeInterviewDraft(sessionStorage,alice.id,{title:'An experience',body:'Actual experience with a coding round.',interview,anonymous:true});const card=renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(InterviewCard,{post:{...post,interview},owner:alice.id,canSave:false,returnTo:'#/interviews',savedView:false,onReload(){}})));assert(card.includes('Research Institute')&&card.includes('Anonymous')&&!card.includes('#/u/alice'));await render(Composer,{owner:alice.id,company:'google',problems:[],onClose(){},onPublished:value=>published.push(value)});await submit('.interview-composer form');assert.equal(published.length,1);assert.equal(published[0].author.id,'anonymous:'+postId);assert.equal(interviews.readInterviewDraft(sessionStorage,alice.id),undefined);await stop();
 // Only the public profile explains missing affiliation; existing institution/domain labels stay intact.
 for(const fields of [
  {affiliation,institution:null,label:'Research Institute'},
  {affiliation:null,institution:{id:'brown',name:'布朗大学',name_en:'Brown University'},label:'Brown University'},
  {affiliation:{kind:'email_domain',name:'brown.edu',domain:'brown.edu'},institution:null,label:'Brown University'},
  {affiliation:{kind:'email_domain',name:'research-lab.ac.uk',domain:'research-lab.ac.uk'},institution:null,label:'Research LAB'},
  {affiliation:null,institution:null,label:'Affiliation not verified'},
  {affiliation:undefined,institution:undefined,label:'Affiliation not verified'},
 ]){
  Object.assign(alice,{affiliation:fields.affiliation,institution:fields.institution,email:'fixture@private.example'});
  await render(UserProfile,{username:'alice'});
  const facts=document.querySelector('.profile-facts');assert(facts.textContent.includes(fields.label));
  assert.equal(!!facts.querySelector('.profile-affiliation-empty'),fields.label==='Affiliation not verified');
  assert(!facts.textContent.includes('unverified')&&!facts.querySelector('.lucide-shield-check'),'Missing affiliation never displays a verified badge');
  assert(!document.body.innerHTML.includes(alice.email),'Public profile rendering must not expose a full mailbox');
  assert(!document.querySelector('.community-post-card .profile-affiliation-empty'),'Activity cards do not gain an empty affiliation placeholder');
  await stop();
 }
 localStorage.setItem('paperleet.locale','zh');await render(UserProfile,{username:'alice'});assert(document.querySelector('.profile-affiliation-empty').textContent.includes('机构未认证'));await stop();localStorage.setItem('paperleet.locale','en');
 alice.is_test_user=true;await render(UserProfile,{username:'alice'});assert(!document.querySelector('.profile-affiliation-empty'),'Explicit test accounts retain their test badge without an affiliation-verification prompt');await stop();delete alice.is_test_user;
 Object.assign(alice,{affiliation});delete alice.institution;delete alice.email;
 const withoutAffiliation=renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(Author,{user:{...anonymous,affiliation:undefined,institution:undefined}})));
 assert(!withoutAffiliation.includes('Affiliation not verified')&&!withoutAffiliation.includes('机构未认证'),'Anonymous cards do not disclose or invent missing affiliation status');
 // Public profile rendering must drop owner-only anonymous history synchronously on account change.
 await render(UserProfile,{username:'alice'});assert(document.querySelector('.community-post-card'));holdProfile=true;account.user=undefined;await render(UserProfile,{username:'alice'});assert(!document.querySelector('.community-post-card'),'Previous owner-only history cannot survive the guest render while requests are pending');await stop();held.forEach(resolve=>resolve({items:[],total:0}));
 account.user=alice;holdProfile=false;localStorage.setItem('paperleet.locale','zh');post.replies=[reply,{...reply,id:'d'.repeat(32),parent_id:reply.id}];await render(Community,{postId,route:'community/'+postId});assert(document.querySelector('.anonymous-author').textContent.includes('匿名用户'));assert(document.querySelector('.reply-context').textContent.includes('匿名用户'));assert(!document.querySelector('.reply-context').textContent.includes('Anonymous'));await stop();
 assert.deepEqual(errors,[]);console.log('PASS anonymous community: real affiliation cards/replies, neutral identity and no profile/avatar leaks, owner edit controls, durable identity-bound drafts and retries, no named fallback on old services, anonymous interview confirmation and owner-only profile isolation');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,{...descriptor});else delete globalThis[key];}}
