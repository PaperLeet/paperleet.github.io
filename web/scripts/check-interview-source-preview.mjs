import assert from 'node:assert/strict';
import React,{act} from 'react';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/interviews?company=google&view=questions',pretendToBeVisual:true});
dom.window.scrollTo=()=>{};dom.window.HTMLElement.prototype.scrollIntoView=function(){};dom.window.matchMedia=()=>({matches:false});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__sourceFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});localStorage.setItem('paperleet.locale','en');
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-source-preview-');
const server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0')},plugins:[{name:'source-preview-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0source-preview-social';},load(id){if(id==='\0source-preview-social')return `import React from 'react';export const useAccount=()=>globalThis.__sourceFixture.account;export const socialApi=(...args)=>globalThis.__sourceFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const PersonLink=({user})=>React.createElement('a',{href:'#/u/'+user.username},user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;export const profileHref=user=>'#/u/'+user.username;`;}}]});
let root;
const originalTimeout=globalThis.setTimeout,originalClear=globalThis.clearTimeout,deadlines=new Map();
globalThis.setTimeout=(callback,delay,...args)=>{if(delay===15000){const token=Symbol('source deadline');deadlines.set(token,()=>{deadlines.delete(token);callback(...args);});return token;}return originalTimeout(callback,delay,...args);};
globalThis.clearTimeout=token=>typeof token==='symbol'?deadlines.delete(token):originalClear(token);
try{
 const {createRoot}=await import('react-dom/client'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Ranking}=await server.ssrLoadModule('/src/InterviewQuestionRanking.tsx');
 const {emptyInterviewFilters,interviewHref}=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const {libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const id='architecture/attention';const problem={id,number:1,title:'注意力',translations:{en:{title:'Attention'}},starter:'initial code',difficulty:'M',estimated_minutes:25,module:{goal:'练习',goal_en:'Practice'},tests:{T0:1}};
 const filters={...emptyInterviewFilters(),company:'google',role:'ml',stage:'technical',days:'90',sort:'helpful',outcome:'offer',q:'Attention',view:'questions',saved:true,study:'all',queue:'1'};location.hash=interviewHref(filters);
 const requests=[],pending=[],account={user:{id:'alice',username:'alice',display_name:'Alice'},loading:false,capabilities:{community_bookmarks:true,interview_outcome_filter:true}};
 let responder;
 globalThis.__sourceFixture={account,request(path,body,signal){assert.equal(body,undefined,'Reading sources cannot mutate reports or private grades');requests.push({path,signal});if(responder)return responder(path,signal);return new Promise((resolve,reject)=>pending.push({path,signal,resolve,reject}));}};
 const questions=[{problem_id:id,reports:4,contributors:2,companies:[{company:'google',reports:4,contributors:2,problem_id:id,latest:'2026-10-17'}]}];
 function Language(){const {locale,setLocale}=useI18n();return React.createElement('button',{id:'source-language',onClick:()=>setLocale(locale==='en'?'zh':'en')},'Language');}
 let component=Ranking,props={filters,questions,problems:[problem],saved:{},ready:true,owner:'alice',onClearSearch(){}};
 const render=async()=>act(async()=>{root??=createRoot(document.getElementById('root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Language),React.createElement('textarea',{id:'private-code',defaultValue:'private implementation'}),React.createElement(component,props)));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const click=async node=>{assert(node,'Missing source control');await act(async()=>{node.focus();node.click();});};
 const answer=async(request,value)=>act(async()=>request.resolve(value));
 const reject=async(request,message)=>act(async()=>request.reject(Error(message)));
 const take=fragment=>{const index=pending.findIndex(request=>request.path.includes(fragment));assert(index>=0,'Missing request '+fragment);return pending.splice(index,1)[0];};
 const pause=()=>new Promise(resolve=>originalTimeout(resolve,5));
 const navigate=async href=>act(async()=>{location.hash=href;await pause();});
 const activate=async link=>act(async()=>{assert(link);link.focus();document.addEventListener('click',event=>event.preventDefault(),{once:true});link.dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true,cancelable:true,detail:0}));});
 await render();
 assert.equal(requests.length,0,'Closed source previews issue no requests');
 const toggle=document.querySelector('.interview-source-preview-toggle');assert(toggle,'High-frequency concepts need an inline source preview');
 assert.equal(toggle.getAttribute('aria-expanded'),'false');const rankedPractice=document.querySelector('.interview-ranking-actions>a');assert(libraryReturnHref(rankedPractice.getAttribute('href')).includes('view=questions'));
 await act(async()=>toggle.click());assert.equal(toggle.getAttribute('aria-expanded'),'true');assert.equal(requests.length,1);
 const query=new URLSearchParams(requests[0].path.split('?')[1]);assert.equal(query.get('problem_id'),id);assert.equal(query.get('limit'),'3');assert.equal(query.get('offset'),'0');assert.equal(query.get('company'),'google');assert.equal(query.get('role'),'ml');assert.equal(query.get('stage'),'technical');assert.equal(query.get('outcome'),'offer');assert.equal(query.get('days'),'90');assert.equal(query.get('sort'),'helpful');assert(!query.has('saved')&&!query.has('q'),'Ranking search and private bookmarks cannot narrow public source evidence');
 const source=await server.ssrLoadModule('/src/interviewSourcePreviewModel.ts');
 const nav=await server.ssrLoadModule('/src/communityNavigation.ts');
 const model=await server.ssrLoadModule('/src/readingReturn.ts');
 const sourceHref=source.interviewSourceHref(filters,id),scope=interviewHref(filters),postId='a'.repeat(32);
 const author={id:'anonymous:'+postId,username:'',display_name:'Anonymous',anonymous:true,avatar_color:'#48766b',affiliation:{kind:'institution',domain:'research.edu',name:'研究机构',name_en:'Research Institute'}};
 const post={id:postId,title:'A real attention experience',body:'## Reasoning\n\n[Approach](https://example.test) with a clear explanation.\n\n![remote](https://example.test/image.png)',anonymous:true,category:'career',author,created_at:'2026-10-17',updated_at:'2026-10-17',counts:{likes:2,replies:0,reposts:0},viewer:{liked:false},interview:{company:'google',role:'ml',stage:'technical',outcome:'offer',interview_month:'',location:'',questions:[id]},replies:[]};
 const feed=(items=[post])=>({items,total:items.length?4:0,limit:3,offset:0,outcome:'offer'});
 assert(source.readInterviewSourcePreview(feed(),sourceHref));
 const badPost=patch=>({...post,...patch}),badDetails=patch=>badPost({interview:{...post.interview,...patch}});
 for(const raw of [null,{},feed([post,post]),{...feed(),limit:20},{...feed(),offset:1},{...feed(),total:0},feed([badPost({solution:true})]),feed([badDetails({questions:['rl/gae']})]),feed([badDetails({company:'amazon'})]),feed([badDetails({role:'llm'})]),feed([badDetails({stage:'onsite'})]),feed([badDetails({outcome:'rejected'})]),feed([badPost({author:{...author,id:'alice',username:'alice',anonymous:false}})]),feed([badPost({author:{...author,id:'anonymous:'+'b'.repeat(32)}})])])assert.equal(source.readInterviewSourcePreview(raw,sourceHref),undefined,'Unconfirmed source rows must never render');
 for(const href of ['https://evil.test','#/interviews?view=reports&company=unknown&concept=architecture%2Fattention','#/interviews?concept=architecture%2Fattention&saved=1',scope])assert.equal(source.interviewSourceRequest(href),undefined);
 const first=take('community/interviews?'),refresh=()=>document.querySelector('.interview-source-panel-heading button');
 const beforeBusy=requests.length;await act(async()=>{refresh().click();refresh().click();});assert.equal(requests.length,beforeBusy,'Busy refresh is atomic and read-only');
 const code=document.querySelector('#private-code');code.setSelectionRange(2,8,'backward');code.scrollTop=27;code.scrollLeft=14;sessionStorage.setItem('source-private-draft','Never publish');const storage=JSON.stringify({...sessionStorage});
 code.focus();await answer(first,feed());assert.equal(document.activeElement,code,'Async source data never takes newer focus');
 assert(document.querySelector('.interview-source-author').textContent.includes('Research Institute'));assert(!document.querySelector('.interview-source-author a'));assert(!document.querySelector('.interview-source-excerpt img'));assert(!document.querySelector('img[src="https://example.test/image.png"]'),'Remote body images cannot load from a plain source excerpt');
 assert.equal(document.querySelector('.interview-source-excerpt').textContent,'Approach with a clear explanation.');
 assert.equal(nav.communityPostReturn(document.querySelector('.interview-source-title').getAttribute('href'),postId),scope);
 assert.equal(document.querySelectorAll('.interview-source-item').length,1);assert.equal(document.querySelectorAll('[data-interview-facet]').length,3);
 const allSource=document.querySelector('.interview-source-toolbar a');assert.equal(allSource.getAttribute('href'),sourceHref);
 const countBeforeLocale=requests.length;await click(document.querySelector('#source-language'));assert(document.querySelector('.interview-source-author').textContent.includes('研究机构'));assert.equal(requests.length,countBeforeLocale);await click(document.querySelector('#source-language'));
 await click(refresh());const loading=take('community/interviews?');const stableRefresh=refresh();await reject(loading,'Network unavailable');assert.equal(refresh(),stableRefresh);assert(document.querySelector('.interview-source-error').textContent.includes('Network unavailable'));assert.equal(document.activeElement,stableRefresh);
 await act(async()=>{refresh().click();refresh().click();});assert.equal(pending.length,1,'Rapid retry cannot duplicate source reads');const malformed=take('community/interviews?');await answer(malformed,feed([badDetails({company:'amazon'})]));assert(document.querySelector('.interview-source-error'));assert(!document.querySelector('.interview-source-item'));
 await click(refresh());const timed=take('community/interviews?');assert.equal(deadlines.size,1);await act(async()=>[...deadlines.values()][0]());assert(timed.signal.aborted);assert(document.querySelector('.interview-source-error').textContent.includes('timed out'));await answer(timed,feed());assert(!document.querySelector('.interview-source-item'),'Timed-out results cannot replace the error');
 await click(refresh());const abandoned=take('community/interviews?');const inside=refresh();inside.focus();await act(async()=>inside.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true,isComposing:true})));assert.equal(toggle.getAttribute('aria-expanded'),'true');
 await act(async()=>inside.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true})));assert.equal(toggle.getAttribute('aria-expanded'),'false');assert.equal(document.activeElement,toggle);assert(abandoned.signal.aborted);assert(document.getElementById(toggle.getAttribute('aria-controls')).hidden);await answer(abandoned,feed());assert(!document.querySelector('.interview-source-item'));
 await click(toggle);const ownerRead=take('community/interviews?');props={...props,owner:'bob'};await render();assert(ownerRead.signal.aborted);assert.equal(document.querySelector('.interview-source-preview-toggle').getAttribute('aria-expanded'),'false');await answer(ownerRead,feed());assert(!document.querySelector('.interview-source-item'));
 props={...props,accountLoading:true};await render();await click(document.querySelector('.interview-source-preview-toggle'));const hydrationCount=requests.length;assert(document.querySelector('.interview-source-panel').textContent.includes('Checking your account'));props={...props,accountLoading:false};await render();assert.equal(requests.length,hydrationCount+1);const hydration=take('community/interviews?');await answer(hydration,feed([]));assert(document.querySelector('.interview-source-panel').textContent.includes('No experiences match'));
 assert.equal(JSON.stringify({...sessionStorage}),storage);assert.equal(code,document.querySelector('#private-code'));assert.deepEqual([code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],['private implementation',2,8,'backward',27,14]);
 await stop();
 console.log('PASS source preview: explicit three-report public query, strict scoped/anonymous protocol, real affiliation/facets/plain excerpts, native full-source links, no hidden request or writes, timeout/retry/duplicate/abort/owner/hydration guards, local Escape/IME and preserved private code/draft');

 // Actual hub -> full post -> practice -> source post -> ranked concept, including a row beyond the initial six.
 const {default:Hub}=await server.ssrLoadModule('/src/Interviews.tsx'),{default:Community}=await server.ssrLoadModule('/src/Community.tsx');
 const {usePracticeReadingOrigin}=await server.ssrLoadModule('/src/usePracticeReadingOrigin.ts');
 const {sourcePostReadingReturn}=model;
 const library=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const allProblems=[problem,...Array.from({length:6},(_,index)=>({...problem,id:'architecture/concept_'+index,number:index+2,title:'知识点 '+index,translations:{en:{title:'Attention concept '+index}}}))];
 const allQuestions=allProblems.map(row=>({...questions[0],problem_id:row.id}));const seventh=allProblems[6];
 const detail={...post,interview:{...post.interview,questions:allProblems.map(row=>row.id)}};
 const summary={reports:4,contributors:2,days:'90',basis:'published-community-reports',generated_at:'2026-10-17',companies:[],questions:allQuestions,outcome:'offer'};
 let summaryValue=summary,holdSummary=false;
 responder=(path,signal)=>{
  if(path.includes('/summary')&&!holdSummary)return Promise.resolve(summaryValue);
  if(path==='community/posts/'+postId)return Promise.resolve(detail);
  return new Promise((resolve,reject)=>pending.push({path,signal,resolve,reject}));
 };
 const hubProps=()=>({route:scope,problems:allProblems,saved:{},progressReady:true});
 const hub=async(extra={})=>{component=Hub;props={...hubProps(),...extra};await render();};
 const sourceRow=concept=>document.querySelector(`[aria-labelledby="interview-ranked-${encodeURIComponent(concept)}"]`);
 const sourceTitle=concept=>sourceRow(concept)?.querySelector('.interview-source-title');
 let viewport=0,scrolls=[];Object.defineProperty(window,'scrollY',{get:()=>viewport,configurable:true});window.scrollTo=options=>{viewport=options.top;scrolls.push(options.top);};
 dom.window.HTMLElement.prototype.getBoundingClientRect=function(){return {top:400-viewport,bottom:430-viewport,left:0,right:500,width:500,height:30};};
 await navigate(scope);await hub();assert.equal(document.querySelectorAll('.interview-source-preview-toggle').length,6);assert(!pending.length);
 await click(document.querySelector('.interview-ranking-more'));await click(sourceRow(seventh.id).querySelector('.interview-source-preview-toggle'));await answer(take('community/interviews?'),{...feed([detail])});
 viewport=150;await activate(sourceTitle(seventh.id));const remembered=model.currentReadingReturn();assert.equal(remembered.concept,seventh.id);assert.equal(remembered.href,scope);assert.equal(remembered.owner,'account:alice');assert(!sessionStorage.getItem(model.READING_RETURN_KEY).includes(detail.body));
 const threadHref=sourceTitle(seventh.id).getAttribute('href');await stop();await navigate(threadHref);component=Community;props={postId:postId,route:threadHref,topics:allProblems};await render();assert.equal(model.currentReadingReturn().phase,'reading');
 const sourcePost=nav.communityPostHref(postId),practice=document.querySelector('.discussion-page .interview-report-practice a');assert(practice);assert.equal(library.libraryReturnHref(practice.getAttribute('href')),sourcePost);const practiceHref=practice.getAttribute('href');
 await stop();await navigate(practiceHref);
 function Practice(){usePracticeReadingOrigin('account:alice',problem,sourcePost,practiceHref);return React.createElement('p',null,'Real module');}
 component=Practice;props={};await render();assert.equal(model.currentReadingReturn().concept,seventh.id);assert.equal(model.currentReadingReturn().phase,'reading');
 await stop();await navigate(sourcePost);assert.equal(sourcePostReadingReturn('account:alice',postId,sourcePost),scope);component=Community;props={postId:postId,route:sourcePost,topics:allProblems};await render();assert.equal(document.querySelector('.discussion-navigation a').getAttribute('href'),scope);
 await stop();await navigate(scope);viewport=0;scrolls=[];await hub();assert(sourceRow(seventh.id));assert.equal(sourceRow(seventh.id).querySelector('.interview-source-preview-toggle').getAttribute('aria-expanded'),'true');assert.equal(scrolls.length,0,'A source return waits for actual source data');
 await answer(take('community/interviews?'),feed([detail]));assert.equal(scrolls.length,1);assert.equal(viewport,150);assert.equal(document.activeElement,sourceTitle(seventh.id));assert.equal(model.currentReadingReturn(),undefined);assert(sourceRow(seventh.id),'The revealed seventh concept stays mounted after its return is consumed');
 // Exact concept binding prevents the same post in another open preview from becoming the return target.
 async function armReturn(){await stop();model.discardReadingReturn();await navigate(scope);model.rememberReadingReturn({owner:'account:alice',href:scope,post:postId,concept:seventh.id,link:'title',top:250,focus:true});await navigate(threadHref);await navigate(scope);viewport=0;scrolls=[];}
 await armReturn();await hub();const returningRead=take('community/interviews?');document.querySelector('#private-code').focus();await answer(returningRead,feed([detail]));assert.equal(scrolls.length,0);assert(document.querySelector('.reading-return-notice button'));
 await click(sourceRow(id).querySelector('.interview-source-preview-toggle'));await answer(take('community/interviews?'),feed([detail]));assert.equal(document.querySelectorAll('.interview-source-title').length,2);
 await click(document.querySelector('.reading-return-notice button'));assert.equal(document.activeElement,sourceTitle(seventh.id));assert.equal(scrolls.length,1);assert.equal(model.currentReadingReturn(),undefined);
 await armReturn();await hub();document.querySelector('#private-code').focus();await answer(take('community/interviews?'),feed([detail]));assert(document.querySelector('.reading-return-notice button'));await click(sourceRow(seventh.id).querySelector('.interview-source-panel-heading button'));const manualRefresh=take('community/interviews?');assert(!document.querySelector('.reading-return-notice'),'A manual return waits for the refreshed source cohort');assert(model.currentReadingReturn());await answer(manualRefresh,feed([detail]));assert.equal(scrolls.length,0);await click(document.querySelector('.reading-return-notice button'));assert.equal(document.activeElement,sourceTitle(seventh.id));
 for(const failure of ['empty','malformed','error','timeout']){
  await armReturn();await hub();const request=take('community/interviews?');
  if(failure==='error')await reject(request,'Unavailable');else if(failure==='timeout')await act(async()=>[...deadlines.values()][0]());else await answer(request,failure==='empty'?feed([]):{...feed(),limit:20});
  assert.equal(model.currentReadingReturn(),undefined,failure+' settles return readiness');assert(document.querySelector('.reading-return-notice').textContent.includes('no longer on this page'));assert.equal(document.activeElement,document.querySelector('.interview-question-ranking'));
  if(failure==='timeout')await answer(request,feed([detail]));
 }
 await armReturn();summaryValue={...summary,questions:allQuestions.slice(0,6)};await hub();assert(!sourceRow(seventh.id));assert(!pending.length);assert.equal(model.currentReadingReturn(),undefined);assert(document.querySelector('.reading-return-notice').textContent.includes('no longer on this page'));summaryValue=summary;
 await armReturn();await hub();const stale=take('community/interviews?');await click(sourceRow(seventh.id).querySelector('.interview-source-preview-toggle'));assert.equal(model.currentReadingReturn(),undefined,'Explicitly closing an automatic preview cancels only that return');assert(stale.signal.aborted);await answer(stale,feed([detail]));assert.equal(scrolls.length,0);
 await armReturn();await hub();const oldOwner=take('community/interviews?');account.user={...account.user,id:'bob',username:'bob'};await hub();assert(oldOwner.signal.aborted);await answer(oldOwner,feed([detail]));assert(!sourceTitle(seventh.id));assert.equal(model.currentReadingReturn(),undefined);assert.equal(scrolls.length,0);account.user={...account.user,id:'alice',username:'alice'};
 await armReturn();account.loading=true;await hub();assert(!pending.length);assert(model.currentReadingReturn());document.querySelector('#private-code').focus();account.loading=false;await hub();await answer(take('community/interviews?'),feed([detail]));assert.equal(scrolls.length,0);assert(document.querySelector('.reading-return-notice button'));await click(document.querySelector('.reading-return-notice button'));assert.equal(document.activeElement,sourceTitle(seventh.id));
 await armReturn();summaryValue={};await hub();assert.equal(model.currentReadingReturn(),undefined);assert(document.querySelector('.reading-return-notice').textContent.includes('no longer on this page'));assert(!pending.length);summaryValue=summary;
 // A saved learning-state filter waits for real progress before choosing a return target.
 const studyScope=interviewHref({...filters,study:'todo'});
 await stop();model.discardReadingReturn();await navigate(studyScope);model.rememberReadingReturn({owner:'account:alice',href:studyScope,post:postId,concept:seventh.id,link:'title',top:250,focus:true});const studyThread=nav.communityPostHref(postId,studyScope);await navigate(studyThread);await navigate(studyScope);scrolls=[];viewport=0;
 await hub({route:studyScope,progressReady:false});await answer(take('community/interviews?'),feed([detail]));assert(model.currentReadingReturn());assert.equal(scrolls.length,0);
 await hub({route:studyScope,progressReady:true});assert.equal(model.currentReadingReturn(),undefined);assert.equal(document.activeElement,sourceTitle(seventh.id));assert.equal(scrolls.length,1);
 await armReturn();holdSummary=true;await hub();const waitingSummary=take('/summary');await navigate('#/interviews?company=amazon&view=questions');await hub({route:location.hash});assert(waitingSummary.signal.aborted);await answer(waitingSummary,summary);assert.equal(model.currentReadingReturn(),undefined);assert.equal(scrolls.length,0);await stop();for(const request of pending.splice(0))await answer(request,summary);holdSummary=false;
 // Strict ephemeral metadata: questions require a bounded concept, reports/profiles reject it, and practice is not broadened.
 const row={owner:'account:alice',href:scope,post:postId,link:'title',concept:seventh.id,top:250,focus:true};
 for(const patch of [{concept:undefined},{concept:'invalid'},{concept:'architecture/../escape'},{href:interviewHref({...filters,concept:id})},{href:sourceHref},{href:'#/u/alice?tab=interviews'},{link:'practice',practice:id},{body:'Do not store'}]){model.discardReadingReturn();model.rememberReadingReturn({...row,...patch});assert.equal(model.currentReadingReturn(),undefined);}
 assert.equal(model.readingPracticeTarget(library.withLibraryContext(id,scope),scope),undefined);
 model.discardReadingReturn();
 console.log('PASS ranked source reading: actual hub/post/practice/return chain, seventh-concept reveal and exact duplicate-post row, applied ranking/query/queue scope, native keyboard focus and newer-intent manual return, empty/error/malformed/timeout/missing readiness, close cancellation, stale owner/route rejection and strict metadata with unchanged practice semantics');
}finally{if(root)await act(async()=>root.unmount());globalThis.setTimeout=originalTimeout;globalThis.clearTimeout=originalClear;await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();for(const[key,value]of previous)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}
