import assert from 'node:assert/strict';
import React,{act} from 'react';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/interviews',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});dom.window.scrollTo=()=>{};dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,location:dom.window.location,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,IS_REACT_ACT_ENVIRONMENT:true,__reportProgress:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});localStorage.setItem('paperleet.locale','en');
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-report-progress-');
const server=await createServer({cacheDir,define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_API_ORIGIN':JSON.stringify('')},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'report-progress-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0report-progress-social';},load(id){if(id==='\0report-progress-social')return `import React from 'react';export const useAccount=()=>globalThis.__reportProgress.account;export const socialApi=(...args)=>globalThis.__reportProgress.request(...args);export class SocialError extends Error{};export const errorMessage=e=>e.message;export const Avatar=()=>React.createElement('span',{'aria-hidden':true});export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {PracticeProgressProvider:Provider,usePracticeProgress}=await server.ssrLoadModule('/src/PracticeProgressContext.tsx');
 const {default:Card}=await server.ssrLoadModule('/src/InterviewReportCard.tsx');
 const {default:Interviews}=await server.ssrLoadModule('/src/Interviews.tsx');
 const {default:Community}=await server.ssrLoadModule('/src/Community.tsx');
 const {default:Meta}=await server.ssrLoadModule('/src/InterviewMeta.tsx');
 const {topicFromProblem}=await server.ssrLoadModule('/src/communityTopics.ts');
 const {browserLearningPassed}=await server.ssrLoadModule('/src/types.ts');
 const {interviewHref,emptyInterviewFilters}=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const {libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const {useReadingPosition}=await server.ssrLoadModule('/src/useReadingPosition.ts');
 const {usePracticeReadingOrigin}=await server.ssrLoadModule('/src/usePracticeReadingOrigin.ts');
 const reading=await server.ssrLoadModule('/src/readingReturn.ts');
 const id='a'.repeat(32),alice={id:'alice',username:'alice',display_name:'Alice',avatar_color:'#48766b',can_publish_community:true,affiliation:{kind:'institution',domain:'research.edu',name:'测试研究院',name_en:'Research Institute'}};
 const account={user:alice,health:{mode:'hosted'},loading:false,capabilities:{community_bookmarks:true},refresh:async()=>{}};
 const modules=['architecture/attention','rl/gae','diffusion/ddpm','optimization/adam','finetuning/lora'].map((id,index)=>({id,number:index+1,title:'知识点 '+index,translations:{en:{title:'Concept '+index}},starter:'def solution(): pass',module:{optional:index===3},tests:{T0:1}}));
 const pass=problem=>({problem:problem.id,summary:{T0:[1,1]},tests:[{id:'works',layer:'T0',passed:true}]});
 const initialSaved=()=>({[modules[0].id]:{acceptedReport:pass(modules[0]),report:{...pass(modules[0]),tests:[{id:'bad',layer:'T0',passed:false}]}},[modules[1].id]:{code:'PRIVATE_OLDER_DRAFT',updated:'2026-10-17T01:00:00Z'},[modules[2].id]:{code:'PRIVATE_NEWER_DRAFT',updated:'2026-10-17T02:00:00Z'},[modules[3].id]:{[browserLearningPassed]:true},[modules[4].id]:{report:{...pass(modules[4]),problem:'wrong/module'}}});
 const post={id,title:'A useful interview experience',body:'Public experience body.',category:'career',author:alice,created_at:'2026-10-17T00:00:00Z',updated_at:'2026-10-17T00:00:00Z',problem_id:null,problem:null,counts:{likes:2,replies:0,reposts:0},viewer:{liked:false,bookmarked:false},interview:{company:'amazon',role:'ml',stage:'technical',outcome:'unknown',interview_month:'2026-10',location:'Remote',questions:modules.map(row=>row.id)}};
 const scope=interviewHref({...emptyInterviewFilters(),company:'amazon',role:'ml',stage:'technical',q:'Attention & 多模态',page:2,saved:true});
 let owner='user:alice',saved=initialSaved(),ready=true,failed=false,retries=0,available=modules,questions=post.interview.questions,surface='cards',provider=true,catalog=modules,lastContext;
 const requests=[];globalThis.__reportProgress={account,request:async(path,payload)=>{
  requests.push({path,payload});assert.equal(payload,undefined,'Displaying practice progress never publishes, judges or writes to an account');
  if(path.includes('/summary?'))return {reports:1,contributors:1,days:'365',basis:'published-community-reports',generated_at:'2026-10-17T00:00:00Z',companies:[{company:'amazon',reports:1,contributors:1}],questions:[]};
  if(path.startsWith('community/interviews?'))return {items:[post],total:21,limit:20,offset:20,...(new URLSearchParams(path.split('?')[1]).get('saved')==='true'?{owner_id:account.user.id}:{})};
  if(path==='community/posts/'+id)return {...post,replies:[]};
  throw Error('Unexpected progress request: '+path);
 }};
 const h=React.createElement;
 function Probe(){const context=usePracticeProgress();lastContext=context;return h('output',{id:'progress-probe'},JSON.stringify(context&&{owner:context.owner,ready:context.ready,failed:context.failed,values:modules.map(row=>context.read(row.id))}));}
 function Locale(){const {locale,setLocale}=useI18n();return h('button',{id:'locale',onClick:()=>setLocale(locale==='en'?'zh':'en')},'Locale');}
 const card=()=>h(Card,{post:{...post,interview:{...post.interview,questions}},owner:account.user?.id||'',canSave:false,returnTo:scope,savedView:false,problems:available,catalogError:available.length<modules.length?'Catalog unavailable':undefined,onReload(){}});
 function ReadingList(){const position=useReadingPosition(account.user?'account:'+account.user.id:'guest',scope,true);return h('main',{ref:position.container},h('p',{'data-reading-list':'',tabIndex:-1},'Experiences'),h(Card,{post,owner:account.user?.id||'',canSave:false,returnTo:scope,savedView:false,problems:available,onReload(){},resumed:position.resumedPost===id,resumedPractice:position.resumedPractice}));}
 function Workspace(){const target=modules.find(row=>location.hash.split('?')[0]==='#/'+row.id)||modules[0];usePracticeReadingOrigin(account.user?'account:'+account.user.id:'guest',target,scope,location.hash);return h('a',{id:'back-to-experience',href:scope},'Back to experience');}
 function Journey(){const [route,setRoute]=React.useState(location.hash);React.useEffect(()=>{const update=()=>setRoute(location.hash);window.addEventListener('hashchange',update);return()=>window.removeEventListener('hashchange',update);},[]);return route===scope?h(ReadingList):h(Workspace);}
 const render=async()=>act(async()=>{root??=createRoot(document.getElementById('root'));const content=h(React.Fragment,null,h(Locale),h(Probe),h('textarea',{'aria-label':'Private implementation',defaultValue:'PRIVATE_EDITOR_CODE'}),surface==='cards'?h(React.Fragment,null,card(),card()):surface==='meta'?h(Meta,{details:{...post.interview,questions},topics:available.map(topicFromProblem),returnTo:'#/community/'+id}):surface==='feed'?h(Interviews,{route:scope,problems:available,saved,progressReady:ready}):surface==='discussion'?h(Community,{postId:id,route:'#/community/'+id,topics:available.map(topicFromProblem)}):h(Journey));root.render(h(LocaleProvider,null,provider?h(Provider,{owner,problems:catalog,saved,ready,failed,retry:()=>{retries++;}},content):content));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const area=()=>document.querySelector('.interview-report-practice'),next=()=>area()?.querySelector('.interview-report-practice-next'),count=()=>area()?.querySelector('.interview-report-progress-count');
 const named=text=>[...document.querySelectorAll('button')].find(node=>node.textContent===text);
 const click=async node=>{assert(node);await act(async()=>{node.focus();node.click();});await render();};
 const settle=async predicate=>{for(let i=0;i<80&&!predicate();i++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(predicate(),'Reading return did not settle');};
 location.hash=scope;const storage=JSON.stringify({...sessionStorage});await render();
 assert.equal(count().querySelector('b').textContent,'2 / 5');assert.equal(area().querySelector('progress').value,2);assert.equal(area().querySelector('progress').max,5);assert.equal(next().dataset.readingPractice,modules[2].id);assert(next().textContent.includes('Resume practice'));assert.equal(libraryReturnHref(next().getAttribute('href')),scope);
 assert(area().querySelector('[data-practice-state="done"]'));assert(area().querySelector('[data-practice-state="started"]'));assert.equal(lastContext.read(modules[4].id).state,'started','A wrong-problem report is an attempt, never a pass');assert.equal(lastContext.read('unknown/module'),undefined);
 assert.equal(requests.length,0);assert.equal(JSON.stringify({...sessionStorage}),storage);assert([...document.querySelectorAll('.interview-report-practice,#progress-probe')].every(node=>!node.textContent.includes('PRIVATE_')));assert(!document.getElementById('progress-probe').textContent.includes('code'));
 const code=document.querySelector('[aria-label="Private implementation"]');code.setSelectionRange(2,10,'backward');code.scrollTop=31;code.scrollLeft=12;
 // A focused action keeps its concept while a background grade changes the recommendation.
 await act(async()=>next().focus());const focused=next(),target=focused.getAttribute('href');saved={...saved,[modules[2].id]:{acceptedReport:pass(modules[2])}};await render();assert(next()===focused);assert.equal(next().getAttribute('href'),target);assert(next().textContent.includes('Review this concept'));assert(document.activeElement===focused);
 await act(async()=>code.focus());await render();assert.equal(next().dataset.readingPractice,modules[1].id);assert.deepEqual([code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],[2,10,'backward',31,12]);
 // Completion is based on accepted results, including optional concepts, with no implicit grade.
 saved=Object.fromEntries(modules.map(row=>[row.id,{acceptedReport:pass(row)}]));await render();assert.equal(count().querySelector('b').textContent,'5 / 5');assert(next().textContent.includes('Review these concepts'));assert.equal(next().dataset.readingPractice,modules[0].id);
 saved={};await render();assert.equal(count().querySelector('b').textContent,'0 / 5');assert(next().textContent.includes('Start practice'));assert.equal(next().dataset.readingPractice,modules[0].id);
 // Partial catalogs count only known loaded concepts and never claim complete preparation.
 available=modules.slice(0,2);saved=Object.fromEntries(modules.map(row=>[row.id,{acceptedReport:pass(row)}]));await render();assert.equal(count().querySelector('b').textContent,'2 / 2');assert(count().textContent.includes('3 unavailable'));assert(!next());assert(!area().textContent.includes('Review these concepts'));
 available=modules;catalog=modules.slice(0,1);await render();assert.equal(count().querySelector('b').textContent,'1 / 1');assert(count().textContent.includes('4 unavailable'));catalog=modules;
 // Loading and failed progress retain public links without showing false zeroes or stale grades.
 ready=false;await render();assert(!count());assert(!next());assert(area().textContent.includes('Loading learning progress'));assert(area().querySelectorAll('[data-practice-state]').length===0);assert(area().querySelectorAll('.interview-report-topics a').length===2);
 failed=true;await render();await click(named('Retry learning progress'));assert.equal(retries,1);ready=true;failed=false;await render();assert(document.activeElement===next(),'An explicit successful retry returns focus to the next practice');
 ready=false;failed=true;await render();await click(named('Retry learning progress'));code.focus();ready=true;failed=false;await render();assert(document.activeElement===code,'A retry cannot steal newer editor focus');
 ready=false;failed=true;await render();await click(named('Retry learning progress'));location.hash='#/community';document.getElementById('locale').focus();ready=true;failed=false;await render();assert.equal(document.activeElement.id,'locale');location.hash=scope;
 // No account can see another owner's grades, retry callback or recommendation during a transition.
 saved=initialSaved();account.loading=true;await render();assert.equal(lastContext.owner,'unresolved');assert(!lastContext.read(modules[0].id));assert(!count());assert(!next());account.loading=false;account.user={...alice,id:'bob'};await render();assert(!lastContext.ready);assert(!count());assert(!next());assert.equal(lastContext.retry,undefined);
 owner='user:bob';saved={};await render();assert.equal(count().querySelector('b').textContent,'0 / 5');account.user=undefined;await render();assert(!lastContext.ready);owner='anonymous';await render();assert(lastContext.ready);account.health={mode:'local'};await render();assert(!lastContext.ready);owner='local';await render();assert(lastContext.ready);account.health=undefined;await render();assert(!lastContext.ready);
 // An anonymous report still shows the viewer's workspace, without exposing its author's name.
 account.user=alice;account.health={mode:'hosted'};owner='user:alice';saved=initialSaved();post.author={...alice,id:'anonymous:'+id,username:'',anonymous:true,display_name:'Private author name'};await render();assert.equal(count().querySelector('b').textContent,'2 / 5');assert(area().textContent.includes('My practice'));assert.equal(next().dataset.readingPractice,modules[2].id);assert(!document.body.textContent.includes('Private author name'));assert(document.body.textContent.includes('Research Institute'));post.author=alice;
 // A loading/account A-B-A transition must not revive an old focused recommendation.
 await act(async()=>next().focus());account.loading=true;await render();account.user={...alice,id:'bob'};account.loading=false;owner='user:bob';saved={};await render();account.user=alice;owner='user:alice';saved={[modules[0].id]:{code:'newest private draft',updated:'2026-10-17T04:00:00Z'}};await render();assert.equal(next().dataset.readingPractice,modules[0].id);
 // Without the provider, standalone readers keep their existing read-only behavior.
 await stop();provider=false;await render();assert(!area().querySelector('.interview-report-progress'));assert(!area().querySelector('[data-practice-state]'));assert.equal(requests.length,0);await stop();provider=true;account.user=alice;account.health={mode:'hosted'};owner='user:alice';saved=initialSaved();
 // Full metadata and the actual discussion/feed use the same already-loaded workspace.
 for(const view of ['meta','discussion','feed']){surface=view;location.hash=view==='discussion'?'#/community/'+id:scope;await render();assert(count(),view+' must expose report practice progress');assert.equal(count().querySelector('b').textContent,'2 / 5');assert.equal(next().dataset.readingPractice,modules[2].id);if(view==='meta'||view==='discussion')assert.equal(libraryReturnHref(next().getAttribute('href')),'#/community/'+id);await stop();}
 // A newer unfinished third concept returns to the exact filtered experience and expanded link.
 surface='journey';location.hash=scope;reading.discardReadingReturn();await new Promise(resolve=>setTimeout(resolve,10));await render();const beforeJourney=requests.length;await click(next());await settle(()=>document.getElementById('back-to-experience'));assert.equal(libraryReturnHref(location.hash),scope);assert.equal(reading.currentReadingReturn()?.phase,'reading');await click(document.getElementById('back-to-experience'));await settle(()=>area()?.querySelector('.interview-report-practice-toggle')?.getAttribute('aria-expanded')==='true'&&document.activeElement?.dataset.readingPractice===modules[2].id);assert.equal(location.hash,scope);assert.equal(reading.currentReadingReturn(),undefined);assert.equal(requests.length,beforeJourney);
 await click(named('Locale'));assert(count().textContent.includes('已通过'));assert(next().textContent.includes('继续练习'));assert(area().textContent.includes('知识点 2'));assert([...document.querySelectorAll('.interview-report-practice,#progress-probe')].every(node=>!node.textContent.includes('PRIVATE_')));assert.equal(JSON.stringify({...sessionStorage}),storage);
 console.log('PASS experience practice progress: real account-scoped workspace provider, accepted/browser/attempted grade semantics, latest unfinished/first new/all-done review, focused action identity, truthful partial catalogs, pending/error/retry focus guards, owner/guest/local isolation, provider-free readers, real report/feed/discussion integration and exact filtered third-concept practice return without extra sync, publication, judging or draft changes');
}finally{if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();for(const[key,value]of previous){if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}}
