import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/interviews?view=questions&company=google&role=llm&days=90',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:false});dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const values={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__groupFixture:undefined};
const previous=new Map(Object.keys(values).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(values))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({cacheDir:'node_modules/.vite-practice-group-check',server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0')},plugins:[{name:'practice-group-account',enforce:'pre',resolveId(source){if(source==='./social')return '\0practice-group-account';},load(id){if(id==='\0practice-group-account')return `import React from 'react';export const useAccount=()=>globalThis.__groupFixture.account;export const socialApi=(...args)=>globalThis.__groupFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=()=>null;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Ranking}=await server.ssrLoadModule('/src/InterviewQuestionRanking.tsx');
 const {default:Hub}=await server.ssrLoadModule('/src/Interviews.tsx');
 const {ModuleNavigator}=await server.ssrLoadModule('/src/ModuleNavigator.tsx');
 const {default:Next}=await server.ssrLoadModule('/src/InterviewPracticeNextStep.tsx');
 const {interviewPracticeBatch}=await server.ssrLoadModule('/src/interviewPracticeBatch.ts');
 const {emptyInterviewFilters,interviewHref,readInterviewRoute}=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const {libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const {signInHref,accountReturnHref}=await server.ssrLoadModule('/src/accountNavigation.ts');
 const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 const problems=Array.from({length:8},(_,index)=>({id:'architecture/concept'+(index+1),number:index+1,title:'知识点 '+(index+1),translations:{en:{title:'Concept '+(index+1)}},paper_id:'paper'+index%3,starter:'initial code',difficulty:'M',estimated_minutes:20+index,module:{optional:index%2===0,goal:'练习',goal_en:'Practice'},tests:{T0:1}}));
 const questions=problems.map((problem,index)=>({problem_id:problem.id,reports:10-index,contributors:8-index,companies:[{company:'google',reports:10-index,contributors:8-index,latest:'2026-10-15'}]}));
 const originalEvidence=JSON.stringify(questions);
 const passed=problem=>({problem:problem.id,summary:{T0:[1,1]},tests:[{id:'passed',layer:'T0',passed:true}]});
 const saved={[problems[4].id]:{acceptedReport:passed(problems[4])},[problems[1].id]:{code:'Private draft never belongs in a batch URL',updated:'2026-10-15T11:00:00Z'}};
 const filters={...emptyInterviewFilters(),view:'questions',company:'google',role:'llm',stage:'technical',days:'90',q:'',study:'all'};
 const account={user:{id:'alice',username:'alice',display_name:'Alice'},loading:false,capabilities:{},refresh:async()=>{}};
 let requests=0,writes=0;
 globalThis.__groupFixture={account,request:async(path,body)=>{requests++;if(body){writes++;throw Error('Batch controls must not mutate the service');}assert(path.startsWith('community/interviews/summary?'));return {basis:'published-community-reports',reports:30,contributors:8,companies:[{company:'google',reports:30,contributors:8,latest:'2026-10-15'}],questions};}};
 const props={filters,questions,problems,saved,ready:true,onClearSearch(){}};
 const render=async(Component=Ranking,extra={})=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Component,{...props,...extra})));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const click=async node=>{assert(node,'Missing group control');await act(async()=>{node.focus();node.click();});};
 const group=()=>document.querySelector('.interview-practice-group');
 const selected=()=>[...group().querySelectorAll('[data-selected-problem]')].map(row=>Number(row.dataset.selectedProblem));
 const toggle=number=>[...document.querySelectorAll('.interview-ranking-row')].find(row=>row.querySelector('.interview-ranking-title>a')?.textContent==='Concept '+number)?.querySelector('.interview-group-toggle');
 const control=(number,action)=>group().querySelector(`[data-group-number="${number}"][data-group-action="${action}"]`);
 const clear=()=>group().querySelector('header>button');
 const start=()=>group().querySelector('footer>a');
 const scope=location.href,storage=JSON.stringify({...sessionStorage});
 await render();assert.deepEqual(selected(),[]);assert(!start());
 for(const number of [5,2,4])await click(toggle(number));assert.deepEqual(selected(),[5,2,4]);assert.equal(location.href,scope);assert.equal(requests,0);assert.equal(writes,0);assert.equal(JSON.stringify({...sessionStorage}),storage);assert.equal(JSON.stringify(questions),originalEvidence);assert.equal(group().querySelector('footer>small').textContent,'1 passed · 2 remaining · About 68 min for the whole batch');
 assert(start().textContent.includes('Resume this batch'));let captured=readInterviewRoute(libraryReturnHref(start().getAttribute('href')));assert.equal(captured.queue,'5,2,4');assert.equal(start().getAttribute('href').split('?')[0],'#/2','A completed module is skipped using real workspace evidence');assert.equal(captured.company,'google');assert.equal(captured.role,'llm');assert.equal(captured.stage,'technical');assert.equal(captured.days,'90');assert(!start().getAttribute('href').includes('Private draft'));
 await click(control(4,'up'));assert.deepEqual(selected(),[5,4,2]);assert.equal(document.activeElement,control(4,'up'));await click(control(4,'up'));assert.deepEqual(selected(),[4,5,2]);assert.equal(document.activeElement,control(4,'down'),'At an ordering boundary focus stays on an available control for the moved row');
 await click(control(5,'remove'));assert.deepEqual(selected(),[4,2]);assert.equal(document.activeElement,control(2,'remove'));await click(toggle(5));assert.deepEqual(selected(),[4,2,5]);assert.equal(readInterviewRoute(libraryReturnHref(document.querySelector('.interview-ranking-row:nth-child(2) .interview-ranking-title>a').getAttribute('href'))).queue,'4,2,5','Selected row links use the same explicit group');
 // Captured order drives the actual module picker and next-step component across papers.
 const href=start().getAttribute('href'),back=libraryReturnHref(href),current=problems[3];assert.equal(accountReturnHref(decodeProblemHash(signInHref(href))),href);
 await render(ModuleNavigator,{current,modules:[current],allModules:problems,saved,ready:true,returnTo:back});await click(document.querySelector('.module-picker-trigger'));assert.deepEqual([...document.querySelectorAll('.module-picker-links a')].map(row=>Number(decodeProblemHash(row.getAttribute('href')).split('?')[0])),[4,2,5]);
 await render(Next,{current,problems,saved,returnTo:back,onNotes(){},onLab(){}});assert(document.querySelector('.journey-next-link').getAttribute('href').startsWith('#/2?'));assert.equal(readInterviewRoute(libraryReturnHref(document.querySelector('.journey-next-link').getAttribute('href'))).queue,'4,2,5');
 const complete=Object.fromEntries(problems.map(problem=>[problem.id,{acceptedReport:passed(problem)}]));await render(Next,{current,problems,saved:complete,returnTo:back,onNotes(){},onLab(){}});assert.equal(document.querySelector('.journey-next-link').getAttribute('href'),back);
 await render(Ranking,{filters:readInterviewRoute(back)});assert.deepEqual(selected(),[4,2,5],'Returning restores the exact captured group');assert.equal(interviewPracticeBatch(back,problems,saved,current).next.number,2);
 await click(clear());assert.deepEqual(selected(),[]);assert.equal(document.activeElement,group().querySelector('h3'));await click(document.querySelector('.interview-ranking-more'));
 for(const number of [1,2,3,4,5,6])await click(toggle(number));assert.deepEqual(selected(),[1,2,3,4,5,6]);assert(toggle(7).disabled);await click(toggle(7));assert.deepEqual(selected(),[1,2,3,4,5,6]);assert(!toggle(6).disabled);await click(toggle(6));assert(!toggle(7).disabled);await click(toggle(7));assert.deepEqual(selected(),[1,2,3,4,5,7]);
 // Missing captured modules remain visible and removable rather than silently changing the group.
 await render(Ranking,{filters:{...filters,queue:'2,99999'}});assert.deepEqual(selected(),[2,99999]);assert(!start());assert(group().querySelector('footer>button').disabled);assert(group().textContent.includes('Some practice modules are unavailable'));await click(control(99999,'remove'));assert.deepEqual(selected(),[2]);assert.equal(document.activeElement,control(2,'remove'));assert.equal(readInterviewRoute(libraryReturnHref(start().getAttribute('href'))).queue,'2');
 let catalogRetries=0;await render(Ranking,{filters:{...filters,queue:'2,3'},problems:[problems[1]],onRetryCatalog:()=>catalogRetries++});assert(!start());await click(group().querySelector('.practice-group-unavailable button'));assert.equal(catalogRetries,1);await render(Ranking,{filters:{...filters,queue:'2,3'}});assert.deepEqual(selected(),[2,3]);assert(start(),'A recovered catalog keeps the selected order and enables practice');
 await render(Ranking,{filters:{...filters,queue:'1,2,3,4,5,6,7'}});assert.deepEqual(selected(),[]);assert(group().textContent.includes('This linked batch cannot be edited here'));await click(toggle(3));assert.deepEqual(selected(),[3]);assert(!group().textContent.includes('This linked batch cannot be edited here'));
 // Unresolved/failed progress never borrows a pass or skips the user's first selected module.
 await render(Ranking,{filters:{...filters,queue:'5,2'},ready:false});assert(!group().textContent.includes('passed ·'));assert(start().textContent.includes('Open this batch'));assert(start().getAttribute('href').startsWith('#/5?'));await render(Ranking,{filters:{...filters,queue:'5,2'},error:'unavailable'});assert(start().getAttribute('href').startsWith('#/5?'));
 await render(Ranking,{filters:{...filters,queue:'5,2'},saved:complete});assert(start().textContent.includes('Review this batch'));assert(start().getAttribute('href').startsWith('#/5?'));
 await render(Ranking,{filters:{...filters,study:'todo',queue:'5'},saved:complete});await click(clear());assert(group());assert.equal(document.activeElement,group().querySelector('h3'),'Clearing the last unavailable filtered choice leaves a stable focus target');
 // One action fills only the available slots, in the current filtered ranking order.
 const fill=()=>group().querySelector('.practice-group-fill button');
 const row=number=>group().querySelector(`[data-selected-problem="${number}"]`);
 await render(Ranking,{filters:{...filters,queue:'5,2'}});
 assert.equal(fill().textContent,'Add 4 more');assert(document.getElementById(fill().getAttribute('aria-describedby')).textContent.includes('keeping your existing choices'));const fillOpener=fill();
 await act(async()=>{fillOpener.focus();fillOpener.click();fillOpener.click();});
 assert.deepEqual(selected(),[5,2,1,3,4,6]);assert(document.activeElement===fillOpener);
 assert.equal(fill().getAttribute('aria-disabled'),'true');assert.equal(fill().disabled,false);
 assert.equal(fill().textContent,'Batch full');await click(fill());assert.deepEqual(selected(),[5,2,1,3,4,6]);
 assert(row(5).querySelector('[data-state=done]').textContent==='Passed');
 assert(row(2).querySelector('[data-state=started]').textContent==='In progress');
 assert(row(1).querySelector('[data-state=todo]').textContent==='Not started');
 assert(row(5).querySelector('.practice-group-module-meta').textContent.includes('About 24 min'));
 for(const number of selected()){
  const href=row(number).querySelector('.practice-group-title>a').getAttribute('href');
  assert.equal(href.split('?')[0],'#/'+number);
  assert.equal(readInterviewRoute(libraryReturnHref(href)).queue,'5,2,1,3,4,6');
  assert.equal(accountReturnHref(decodeProblemHash(signInHref(href))),href);
 }
 await click(control(1,'remove'));await click(fill());assert.deepEqual(selected(),[5,2,3,4,6,1]);
 await render(Ranking,{filters:{...filters,study:'todo'}});await click(fill());
 assert.deepEqual(selected(),[1,3,4,6,7,8],'Use the complete filtered ranking, including rows not yet revealed');
 await render(Ranking,{filters:{...filters,q:'Concept 7',queue:'2'}});
 assert.equal(fill().textContent,'Add 1 more');await click(fill());assert.deepEqual(selected(),[2,7]);
 assert.equal(fill().textContent,'Current list added');assert.equal(fill().getAttribute('aria-disabled'),'true');
 await render(Ranking,{filters:{...filters,q:'no matching concept',queue:'2'}});
 assert.equal(fill().textContent,'No concepts in this view');await click(fill());assert.deepEqual(selected(),[2]);
 await render(Ranking,{filters:{...filters,queue:'2,99999'}});await click(fill());
 assert.deepEqual(selected(),[2,99999,1,3,4,5]);assert(!start());
 assert(!row(2).querySelector('.practice-group-title>a'),'An incomplete captured batch cannot produce a partial navigation snapshot');
 await render(Ranking,{filters:{...filters,queue:'5,2'},ready:false});await click(fill());
 assert.deepEqual(selected(),[5,2,1,3,4,6]);assert(start().getAttribute('href').startsWith('#/5?'));
 assert.equal(group().querySelectorAll('[data-state=unknown]').length,6);assert(!group().querySelector('[data-state=done]'));
 await render(Ranking,{filters:{...filters,queue:'5,2'}});
 assert.deepEqual(selected(),[5,2,1,3,4,6]);assert.equal(group().querySelectorAll('[data-state=done]').length,1);
 // A retained state filter cannot assemble all rows while its progress is unresolved.
 for(const pending of [{ready:false},{error:'Progress unavailable'}]){
  await render(Ranking,{filters:{...filters,study:'todo',queue:'2'},...pending});
  assert.equal(fill().textContent,'Progress required');assert.equal(fill().getAttribute('aria-disabled'),'true');
  assert(document.getElementById(fill().getAttribute('aria-describedby')).textContent.includes('still add individual concepts'));
  await click(fill());assert.deepEqual(selected(),[2]);assert(document.activeElement===fill());
 }
 await render(Ranking,{filters:{...filters,study:'todo',queue:'2'}});
 assert.equal(fill().textContent,'Add 5 more');await click(fill());assert.deepEqual(selected(),[2,1,3,4,6,7]);
 // Curated preparation uses its own curriculum order without inventing report counts.
 const {interviewCatalog}=await server.ssrLoadModule('/src/interviewModels.ts');
 const curated=interviewCatalog.paths[0].modules.slice(0,3).map((id,index)=>({...problems[index],id,number:101+index}));
 await render(Ranking,{filters:{...filters,company:'',role:''},questions:[],problems:curated,saved:{},ready:false});
 assert.equal(fill().textContent,'Add the first 3');await click(fill());assert.deepEqual(selected(),[101,102,103]);
 assert.equal(requests,0);assert.equal(writes,0);assert.equal(JSON.stringify({...sessionStorage}),storage);
 assert.equal(JSON.stringify(questions),originalEvidence);assert.equal(location.href,scope);
 // Removing/clearing does not take focus after a newer user action or navigation.
 await render(Ranking,{filters:{...filters,queue:'2,3'}});await act(async()=>{control(2,'remove').focus();control(2,'remove').click();toggle(4).focus();});assert.equal(document.activeElement,toggle(4));await act(async()=>{clear().focus();clear().click();location.hash='#/settings';});assert.notEqual(document.activeElement,group().querySelector('h3'));location.hash=scope.split('#')[1];await stop();
 // A source/route or account change resets only local choices; real progress remains owner-scoped.
 await render(Ranking);await click(toggle(2));await render(Ranking,{filters:{...filters,company:'microsoft'}});assert.deepEqual(selected(),[]);await render(Ranking,{questions:[],filters:{...filters,queue:'2'}});assert.deepEqual(selected(),[2]);assert(document.body.textContent.includes('Curated practice'));assert.equal(JSON.stringify(questions),originalEvidence);await stop();
 await render(Hub,{route:interviewHref(filters),progressReady:true});for(let i=0;i<100&&!document.querySelector('.interview-group-toggle');i++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(document.querySelector('.interview-group-toggle'));await click(document.querySelector('.interview-group-toggle'));assert.equal(selected().length,1);account.user={...account.user,id:'bob',username:'bob'};await render(Hub,{route:interviewHref(filters),progressReady:false,saved:{}});assert.deepEqual(selected(),[]);assert(!group().textContent.includes('passed ·'));assert.equal(writes,0);await stop();
 localStorage.setItem('paperleet.locale','zh');await render(Ranking,{filters:{...filters,queue:'2,3'}});assert(group().textContent.includes('自选练习组'));await click(control(3,'up'));assert(group().querySelector('[role=status]').textContent.includes('移到第 1 位'));assert.equal(document.activeElement,control(3,'down'));await click(fill());assert.deepEqual(selected(),[3,2,1,4,5,6]);assert.equal(fill().textContent,'练习组已满');assert(group().querySelector('[role=status]').textContent.includes('按当前列表顺序加入 4 道'));assert.equal(group().querySelector('[data-state=done]').textContent,'已通过');assert(document.activeElement===fill());assert.equal(writes,0);await stop();
 // One-step undo restores the exact ordered public selection after every local edit.
 localStorage.setItem('paperleet.locale','en');
 const undo=()=>group().querySelector('.practice-group-undo');
 await render(Ranking,{filters:{...filters,queue:'2,3'}});assert(!undo());
 await click(control(2,'remove'));assert.deepEqual(selected(),[3]);await click(undo());assert.deepEqual(selected(),[2,3]);assert.equal(document.activeElement,group().querySelector('h3'));assert(!undo());assert(group().querySelector('.practice-group-status').textContent.includes('Last change undone'));
 assert.equal(readInterviewRoute(libraryReturnHref(start().getAttribute('href'))).queue,'2,3');
 await click(control(3,'up'));assert.deepEqual(selected(),[3,2]);await click(undo());assert.deepEqual(selected(),[2,3]);
 await click(toggle(1));assert.deepEqual(selected(),[2,3,1]);await click(toggle(1));assert.deepEqual(selected(),[2,3]);await click(undo());assert.deepEqual(selected(),[2,3,1],'Undo restores only the most recent toggle');assert(!undo());
 const clearTwice=clear();await act(async()=>{clearTwice.focus();clearTwice.click();clearTwice.click();});assert.deepEqual(selected(),[]);assert(undo());await click(undo());assert.deepEqual(selected(),[2,3,1],'A queued no-op clear does not overwrite the useful snapshot');
 await click(fill());assert.deepEqual(selected(),[2,3,1,4,5,6]);await click(fill());await click(undo());assert.deepEqual(selected(),[2,3,1],'A full batch no-op keeps the last useful change');
 await render(Ranking,{filters:{...filters,queue:''}});assert(!undo());await click(toggle(2));await click(undo());assert.deepEqual(selected(),[],'Adding the first concept can be undone back to an empty batch');assert(!start());
 await render(Ranking,{filters:{...filters,queue:'2,99999'}});await click(control(99999,'remove'));assert(start());await click(undo());assert.deepEqual(selected(),[2,99999]);assert(!start(),'Restoring a missing concept restores its explicit unavailable state');
 await render(Ranking,{filters:{...filters,queue:'2,3'}});await click(control(2,'remove'));
 for(const name of ['hidden','inert','aria-hidden']){group().setAttribute(name,name==='aria-hidden'?'true':'');await click(undo());assert.deepEqual(selected(),[3],'Inactive batches cannot undo');group().removeAttribute(name);}
 const retiredUndo=undo();location.hash='#/settings';await click(retiredUndo);assert.deepEqual(selected(),[3],'Undo cannot mutate a retired route');location.hash=scope.split('#')[1];
 await act(async()=>{undo().focus();undo().click();undo().click();toggle(4).focus();});assert.deepEqual(selected(),[2,3]);assert.equal(document.activeElement,toggle(4),'Undo yields to newer user focus');assert(!undo());
 await click(control(3,'remove'));assert(undo());await render(Ranking,{filters:{...filters,queue:'2,3'},ready:false,saved:{}});assert(undo(),'Progress refresh retains selection history');await click(undo());assert.deepEqual(selected(),[2,3]);assert(start().textContent.includes('Open this batch'));assert(!group().textContent.includes('passed ·'),'Undo never restores old progress');
 await click(control(3,'remove'));assert(undo());await render(Ranking,{filters:{...filters,queue:'2,3'},owner:'bob'});assert(!undo(),'Account changes retire the previous undo');assert.deepEqual(selected(),[2,3]);
 await click(control(3,'remove'));await render(Ranking,{filters:{...filters,queue:'2,3',q:'Concept'},owner:'bob'});assert(!undo(),'Filter changes retire the previous undo');
 await click(control(3,'remove'));await render(Ranking,{filters:{...filters,queue:'2,3',q:'Concept'},owner:'bob',questions:[]});assert(!undo(),'Switching between report and curated sources retires undo');
 assert.equal(location.href,scope);assert.equal(JSON.stringify({...sessionStorage}),storage);assert.equal(JSON.stringify(questions),originalEvidence);assert.equal(writes,0);await stop();
 console.log('PASS practice batch undo: exact ordered add/remove/reorder/clear/fill recovery, queued no-op and double-click bounds, first-add empty and missing-concept restoration, exact practice links, stable or newer focus, hidden/inert/stale-route and account/filter/source retirement without route/storage/service/evidence mutations');
 console.log('PASS custom interview batches: filtered one-action fill with queued-click bounds, individual exact-batch links and real per-module progress, curated/unknown/missing/empty recovery, ordered six-module selection without route jumps/storage/writes, reorder/remove/clear focus, limits, exact source/auth/return context, real picker and next steps, missing/long snapshots, unresolved and accepted progress, source/account isolation, bilingual controls and unchanged contributor evidence');
}finally{if(root)await act(async()=>root.unmount());await server.close();dom.window.close();for(const [key,value] of previous){if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}}
