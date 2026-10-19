import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtempSync,readFileSync,rmSync} from 'node:fs';
import path from 'node:path';
import os from 'node:os';
const dom=new JSDOM('<div id="root"></div>',{url:'https://paperleet.test/#',pretendToBeVisual:true});
let scrolls=0;dom.window.matchMedia=()=>({matches:false});dom.window.HTMLElement.prototype.scrollIntoView=()=>scrolls++;
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,location:dom.window.location,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,HTMLElement:dom.window.HTMLElement,Element:dom.window.Element,ResizeObserver:class{observe(){}disconnect(){}},requestAnimationFrame:fn=>setTimeout(fn,0),cancelAnimationFrame:clearTimeout,scrollY:0,innerHeight:900,scrollTo:()=>scrolls++,IS_REACT_ACT_ENVIRONMENT:true,__libraryInterview:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
sessionStorage.setItem('private-fixture','private report draft');
const cache=mkdtempSync(path.join(os.tmpdir(),'paperleet-library-interview-'));
const server=await createServer({cacheDir:cache,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_API_ORIGIN':JSON.stringify('')},plugins:[{name:'library-interview-account',enforce:'pre',resolveId(source){if(source==='./social')return '\0library-interview-account';},load(id){if(id==='\0library-interview-account')return `export const useAccount=()=>globalThis.__libraryInterview.account;export const socialApi=(...args)=>globalThis.__libraryInterview.request(...args);export const errorMessage=e=>e.message;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client'),h=React.createElement;
 const {PaperLibrary}=await server.ssrLoadModule('/src/PaperViews.tsx');
 const {PracticeProgressProvider}=await server.ssrLoadModule('/src/PracticeProgressContext.tsx');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {libraryInterviewEvidence:project}=await server.ssrLoadModule('/src/libraryInterviewEvidence.ts');
 const {libraryHref,emptyLibraryFilters,libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const {readInterviewRoute}=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const catalog=JSON.parse(readFileSync('public/data/catalog.json','utf8')),p=catalog.problems.slice(0,5);
 const source=(company,index,reports,contributors)=>({company,problem_id:p[index].id,reports,contributors,latest:'2026-10-17T00:00:00Z'});
 const summary={reports:20,contributors:8,days:'365',outcome:'',basis:'published-community-reports',generated_at:'2026-10-17T00:00:00Z',companies:[{company:'google',reports:14,contributors:7},{company:'meta',reports:6,contributors:4}],questions:p.map((problem,index)=>({problem_id:problem.id,reports:index===0?6:index===1?3:2,contributors:index===0?3:index===1?2:1,companies:index===0?[source('google',index,4,2),source('meta',index,2,1)]:[source(index===4?'meta':'google',index,index===1?3:2,index===1?2:1)]}))};
 const before=JSON.stringify({summary,catalog});
 assert.equal(project(summary,catalog.problems).questions.length,5);assert.equal(project(summary,[p[0],p[1]],'meta').questions.length,1);assert.equal(project(summary,catalog.problems).companies.get('google'),4);
 for(const bad of [{...summary,days:'90'},{...summary,questions:[...summary.questions,summary.questions[0]]},{...summary,questions:[{...summary.questions[0],reports:21}]},{...summary,questions:[{...summary.questions[0],companies:[source('google',1,1,1)]}]},{...summary,companies:[...summary.companies,summary.companies[0]]}])assert.equal(project(bad,catalog.problems),undefined);
 const user={id:'library-alice',username:'alice',display_name:'Alice'},account={user,loading:false,health:{mode:'hosted'}},requests=[],pending=[];
 let data=summary,hold=true,owner='user:'+user.id,ready=true,failed=false,progressRetries=0,route='#';
 const pass=problem=>({problem:problem.id,summary:Object.fromEntries(Object.entries(problem.tests).map(([layer,count])=>[layer,[count,count]])),tests:Object.entries(problem.tests).flatMap(([layer,count])=>Array.from({length:count},(_,i)=>({id:layer+'-'+i,layer,passed:true})))});
 let saved={[p[0].id]:{acceptedReport:pass(p[0])},[p[1].id]:{code:'',updated:'2026-10-17T00:00:00Z'}};
 globalThis.__libraryInterview={account,request(url,body,signal){assert.equal(body,undefined);assert.equal(url,'community/interviews/summary?days=365');requests.push({url,signal});return hold?new Promise((resolve,reject)=>pending.push({resolve,reject,signal})):Promise.resolve(data);}};
 let setLocale;function Language(){const i=useI18n();setLocale=i.setLocale;return null;}
 function App(){return h(LocaleProvider,null,h(Language),h('textarea',{id:'private-code',defaultValue:'private implementation\n😀'}),h(PracticeProgressProvider,{owner,problems:catalog.problems,saved,ready,failed,retry:()=>progressRetries++},h(PaperLibrary,{...catalog,saved,progressReady:ready,route,onFilters:href=>{route=href;historyView(href);}})));}
 function historyView(href){location.hash=href.slice(1);void render();}
 root=createRoot(document.getElementById('root'));
 const render=()=>act(async()=>root.render(h(App)));
 const wait=async(fn)=>{const deadline=Date.now()+8000;while(!fn()&&Date.now()<deadline)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(fn());};
 const panel=()=>document.querySelector('.library-interview-guide'),select=()=>panel()?.querySelector('select'),next=()=>panel()?.querySelector('.library-interview-next'),button=text=>[...panel().querySelectorAll('button')].find(b=>b.textContent===text);
 const click=async node=>{assert(node);node.focus();await act(async()=>node.click());};
 await render();await wait(()=>select()&&pending.length);
 const code=document.getElementById('private-code');code.focus();code.setSelectionRange(2,11,'backward');code.scrollTop=72;code.scrollLeft=18;
 const first=pending.shift();assert.equal(requests.length,1);assert(panel().textContent.includes('Loading report evidence'));assert(!next());
 await act(async()=>first.resolve(summary));await wait(()=>next());assert(document.activeElement===code);assert.equal(panel().querySelectorAll('.library-interview-row').length,5);assert(panel().textContent.includes('Solved 1 / 5'));assert.equal(next().textContent,'Continue practice');assert(next().getAttribute('href').startsWith('#/'+p[1].number));assert(!panel().querySelector('a a'));
 const selector=select();selector.focus();await act(async()=>{selector.value='google';selector.dispatchEvent(new dom.window.Event('change',{bubbles:true}));});assert(select()===selector);assert(document.activeElement===selector);assert(panel().textContent.includes('4 linked exercises'));assert.equal(requests.length,1);assert.equal(location.hash,'');
 const details=panel().querySelector('details');await act(async()=>{details.open=true;details.dispatchEvent(new dom.window.Event('toggle'));});const rowIds=[...panel().querySelectorAll('.library-interview-exercise')].map(a=>a.getAttribute('href'));assert.equal(rowIds.length,4);
 const sourceLink=panel().querySelector('.library-interview-source'),filters=readInterviewRoute(sourceLink.getAttribute('href'));assert.equal(filters.company,'google');assert.equal(filters.concept,p[0].id);assert.equal(filters.days,'365');
 // Existing library search drives this same side panel without another statistics request.
 route=libraryHref(undefined,{...emptyLibraryFilters(),query:p[1].translations?.en?.title||p[1].title});location.hash=route.slice(1);await render();assert(select()===selector);assert.equal(select().value,'google');assert(next().getAttribute('href').startsWith('#/'+p[1].number));assert.equal(libraryReturnHref(next().getAttribute('href')),route);assert.equal(requests.length,1);
 const currentCode=document.getElementById('private-code');assert(currentCode===code);assert.deepEqual([code.value,code.selectionStart,code.selectionEnd,code.selectionDirection,code.scrollTop,code.scrollLeft],['private implementation\n😀',2,11,'backward',72,18]);
 route='#';location.hash='';await render();await act(async()=>setLocale('zh'));assert(select()===selector);assert(panel().textContent.includes('公司面试准备'));await act(async()=>setLocale('en'));assert.equal(requests.length,1);
 // Accepted grades and ongoing work determine the direct next exercise; all solved becomes review.
 saved=Object.fromEntries(p.map(problem=>[problem.id,{acceptedReport:pass(problem)}]));code.focus();await render();assert.equal(next().textContent,'Review practice');assert(panel().textContent.includes('Solved 4 / 4'));assert(document.activeElement===code);
 ready=false;await render();assert(!next());assert(!panel().textContent.includes('Solved 4'));route=libraryHref(undefined,{...emptyLibraryFilters(),state:'done'});location.hash=route.slice(1);await render();assert(panel().textContent.includes('Checking library filters'));assert(!panel().textContent.includes('No linked experiences'));assert(select().disabled);route='#';location.hash='';ready=true;failed=true;await render();assert(!next());assert(button('Retry progress'));await click(button('Retry progress'));assert.equal(progressRetries,1);assert(document.activeElement===select());code.focus();failed=false;await render();assert(document.activeElement===code);
 // Owner switches abort old transport and retire company state before the next workspace arrives.
 hold=true;account.loading=true;await render();assert(!next());assert(select().disabled);account.loading=false;account.user={...user,id:'library-bob'};await render();await wait(()=>pending.length);const bob=pending.shift();assert.equal(select().value,'');assert(!next());await act(async()=>bob.resolve(summary));assert(!next(),'A stale Alice progress owner cannot recommend on Bob');owner='user:library-bob';saved={};await render();assert.equal(next().textContent,'Start practice');assert(panel().textContent.includes('Solved 0 / 5'));
 account.user=user;await render();await wait(()=>pending.length);const retired=pending.shift();account.user={...user,id:'library-bob'};await render();await wait(()=>pending.length);const active=pending.shift();assert(retired.signal.aborted);await act(async()=>retired.resolve({...summary,questions:[]}));assert(!next());await act(async()=>active.resolve(summary));assert(next());
 // Broken cohort/module identity is refused; explicit retry moves focus synchronously, never on arrival.
 hold=false;data={...summary,days:'30'};account.user=user;owner='user:'+user.id;await render();await wait(()=>button('Retry report evidence'));assert(!next());hold=true;await click(button('Retry report evidence'));assert(document.activeElement===select());const recovery=pending.shift();code.focus();await act(async()=>recovery.resolve(summary));assert(document.activeElement===code);assert(next());
 hold=false;data={...summary,reports:0,contributors:0,companies:[],questions:[]};account.user={...user,id:'empty-owner'};owner='user:empty-owner';await render();await wait(()=>panel().textContent.includes('No linked experiences'));assert(!next());assert(!panel().textContent.includes('Solved 0 / 0'));
 // A narrow layout starts collapsed and does not request a report until explicitly opened.
 dom.window.matchMedia=()=>({matches:true});account.user=null;owner='anonymous';data=summary;hold=true;const count=requests.length;await render();assert.equal(requests.length,count);const toggle=panel().querySelector('h2 button'),content=document.getElementById(toggle.getAttribute('aria-controls'));assert.equal(toggle.getAttribute('aria-expanded'),'false');assert(content.hidden);await click(toggle);assert.equal(requests.length,count+1);const narrow=pending.shift();code.focus();await act(async()=>narrow.resolve(summary));assert(document.activeElement===code);assert(next());select().focus();await act(async()=>select().dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true})));assert(content.hidden);assert(document.activeElement===toggle);await click(toggle);assert(!content.hidden);assert.equal(requests.length,count+1);const narrowSelector=select();narrowSelector.focus();await act(async()=>toggle.click());assert(content.hidden);assert(document.activeElement===toggle);await click(toggle);assert(select()===narrowSelector);assert.equal(requests.length,count+1);
 // A real rejected transport has the same explicit recovery, without exposing an old owner's result.
 account.user={...user,id:'network-owner'};owner='user:network-owner';await render();await click(panel().querySelector('h2 button'));const network=pending.shift();await act(async()=>network.reject(Error('offline')));assert(button('Retry report evidence'));assert(!next());await click(button('Retry report evidence'));const networkRetry=pending.shift();code.focus();await act(async()=>networkRetry.resolve(summary));assert(document.activeElement===code);assert(next());
 assert.equal(JSON.stringify({summary,catalog}),before);assert.equal(scrolls,0);assert.equal(code.value,'private implementation\n😀');assert.equal(sessionStorage.getItem('private-fixture'),'private report draft');assert(Object.keys(sessionStorage).every(key=>['private-fixture','paperleet.library.positions.v2'].includes(key)));assert(requests.every(r=>r.url==='community/interviews/summary?days=365'));
 console.log('PASS library interview guide: actual deferred library sidebar, one scoped public evidence request, strict source/cohort validation, company selection and native disclosure, filter/return continuity, real owner-safe practice grades and next exercise, all-solved review, locale/focus/editor preservation, retry/background/ABA/account/pending/empty guards and no private writes');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();rmSync(cache,{recursive:true,force:true});dom.window.close();for(const [key,d]of previous)if(d)Object.defineProperty(globalThis,key,d);else delete globalThis[key];
}
