import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/'});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Ranking}=await server.ssrLoadModule('/src/InterviewQuestionRanking.tsx');
 const {ModuleNavigator}=await server.ssrLoadModule('/src/ModuleNavigator.tsx');
 const {default:Next}=await server.ssrLoadModule('/src/InterviewPracticeNextStep.tsx');
 const {emptyInterviewFilters,readInterviewRoute,interviewHref}=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const {interviewPracticeHref,interviewPracticeBatch}=await server.ssrLoadModule('/src/interviewPracticeBatch.ts');
 const {libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const {signInHref,accountReturnHref}=await server.ssrLoadModule('/src/accountNavigation.ts');
 const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 for(const study of ['all','todo','started','done']){
  const filters={...emptyInterviewFilters(),view:'questions',company:'google',role:'llm',stage:'technical',days:'90',study,queue:'22,26,32',q:'数学 & attention 😀'};
  const href=interviewHref(filters);assert.deepEqual(readInterviewRoute(href),filters);
  for(const destination of [href,'#/22?'+new URLSearchParams({from:href.slice(2)})])assert.equal(accountReturnHref(decodeProblemHash(signInHref(destination))),destination);
 }
 for(const query of ['study=bad','study=','study=todo&study=done','queue=','queue=0','queue=01','queue=1,1','queue=1,','queue=1,-2','queue=100000','queue=1&queue=2','queue='+Array.from({length:81},(_,index)=>index+1).join(',')]){
  assert.equal(readInterviewRoute('interviews?'+query),undefined,query);
  assert.equal(signInHref('#/interviews?'+query),'#/account',query);
 }
 const long={...emptyInterviewFilters(),view:'questions',study:'todo',q:'😀'.repeat(200),queue:Array.from({length:80},(_,index)=>index+1).join(',')};
 assert.deepEqual(readInterviewRoute(interviewHref(long)),long);
 const ordinary=readInterviewRoute(interviewHref({...long,view:'reports'}));assert.equal(ordinary.study,'all');assert.equal(ordinary.queue,'','Study controls cannot become public report filters');
 const problems=Array.from({length:14},(_,index)=>({id:'architecture/concept'+(index+1),number:index+1,title:'知识点 '+(index+1),translations:{en:{title:'Concept '+(index+1)}},paper_id:'paper'+index%3,starter:'initial',module:{optional:index%2===0,goal:'练习',goal_en:'Practice'},difficulty:'M',tests:{T0:1}}));
 const filters={...emptyInterviewFilters(),view:'questions',company:'google',role:'llm',study:'todo',days:'90'};
 const current=problems[8],href=interviewPracticeHref(current,filters,problems),returnTo=libraryReturnHref(href),captured=readInterviewRoute(returnTo);
 assert.equal(captured.queue,'7,8,9,10,11,12');assert.equal(captured.study,'todo');
 assert.equal(accountReturnHref(decodeProblemHash(signInHref(href))),href);
 const passed=problem=>({report:{problem:problem.id,summary:{T0:[1,1]},tests:[{id:'passed',layer:'T0',passed:true}]}});
 const saved={[problems[9].id]:passed(problems[9])};
 assert.deepEqual(interviewPracticeBatch(returnTo,problems,saved,current).modules.map(problem=>problem.number),[7,8,9,10,11,12]);
 assert.equal(interviewPracticeBatch(returnTo,problems,saved,current).next.number,11,'Continue in the selected batch, skipping real completed work');
 assert.equal(interviewPracticeBatch(undefined,problems,saved,current),undefined);
 assert.equal(readInterviewRoute(libraryReturnHref(interviewPracticeHref(problems[13],filters,problems))).queue,'13,14');
 assert.equal(readInterviewRoute(libraryReturnHref(interviewPracticeHref(current,filters,[]))).queue,'9','An explicitly opened module always belongs to its navigation context');
 const render=async(Component,props)=>act(async()=>{if(!root)root=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Component,props)));});
 const click=async element=>{assert(element,'Missing control');await act(async()=>element.click());};
 await render(ModuleNavigator,{current,modules:[current,problems[13]],allModules:problems,saved,ready:true,returnTo});
 await click(document.querySelector('.module-picker-trigger'));
 assert(document.querySelector('.module-picker-heading').textContent.includes('This preparation batch'));
 assert.equal(document.querySelectorAll('.module-picker-links a').length,6,'The picker includes this batch across papers, including its optional exercises');
 assert([...document.querySelectorAll('.module-picker-links a')].every(anchor=>readInterviewRoute(libraryReturnHref(anchor.getAttribute('href'))).queue===captured.queue));
 await click(document.querySelector('[aria-label="Next module"]'));assert(location.hash.startsWith('#/10?'));assert.equal(readInterviewRoute(libraryReturnHref(location.hash)).study,'todo');
 const actions={onNotes(){},onLab(){}};
 await render(Next,{current,problems,saved,next:problems[13],returnTo,...actions});
 assert(document.querySelector('.journey-next-link').href.includes('#/11?'));assert(document.querySelector('.journey-next-link').textContent.includes('Continue this preparation batch'));
 const completed=Object.fromEntries(problems.slice(6,12).map(problem=>[problem.id,passed(problem)]));
 await render(Next,{current,problems,saved:completed,next:problems[13],returnTo,...actions});assert.equal(document.querySelector('.journey-next-link').getAttribute('href'),returnTo);assert(document.body.textContent.includes('Practice batch complete'));
 const partial=problems.filter(problem=>problem.number!==12);
 await render(Next,{current,problems:partial,saved:completed,next:problems[13],returnTo,...actions});assert(document.body.textContent.includes('Some batch modules are unavailable'));assert.equal(document.querySelector('.journey-next-link').getAttribute('href'),returnTo);
 await render(ModuleNavigator,{current,modules:[current,problems[13]],allModules:partial,saved:completed,ready:true,returnTo});await click(document.querySelector('.module-picker-trigger'));assert(document.querySelector('.module-picker-heading').textContent.includes('5 / 6'));assert(document.querySelector('.module-batch-note'));
 const invalidScope=interviewHref({...filters,queue:'1,2'});
 await render(ModuleNavigator,{current,modules:[current,problems[13]],allModules:problems,saved,ready:true,returnTo:invalidScope});assert(document.querySelector('.module-batch-unavailable'));assert(!document.querySelector('.module-picker-trigger'),'Changed snapshots never silently switch to a different paper');
 await render(Next,{current,problems,saved,next:problems[13],returnTo:invalidScope,...actions});assert.equal(document.querySelector('.journey-next-link').getAttribute('href'),invalidScope);
 const questions=problems.map((problem,index)=>({problem_id:problem.id,reports:20-index,contributors:20-index,companies:[]}));
 const ranking={filters,questions,problems,saved:{},ready:true,onClearSearch(){}};
 await render(Ranking,ranking);
 assert(document.querySelector('.interview-ranking-filters [aria-pressed="true"]').textContent.includes('Not started'));
 await click([...document.querySelectorAll('.interview-ranking-filters button')].find(button=>button.textContent.includes('In progress')));
 assert.equal(readInterviewRoute(location.hash).study,'started');
 assert(document.body.textContent.includes('No concepts with this learning state'));
 await render(Ranking,{...ranking,filters:readInterviewRoute(location.hash)});assert(document.body.textContent.includes('No concepts with this learning state'),'A refreshed route restores the chosen state');
 const doneFilters={...filters,study:'done'};
 await render(Ranking,{...ranking,filters:doneFilters,saved:completed,ready:false});assert(document.querySelector('.interview-ranking-filters [aria-pressed="true"]').textContent.includes('Passed'));assert(document.body.textContent.includes('Your filter is kept'));assert(!document.querySelector('.interview-ranking-completion'));
 await click(document.querySelector('.interview-ranking-more'));assert.equal(document.querySelectorAll('.interview-ranking-row').length,12,'Unresolved progress keeps the chosen filter while allowing neutral practice browsing');
 await render(Ranking,{...ranking,filters:doneFilters,saved:{},ready:true});assert(document.body.textContent.includes('No concepts with this learning state'),'The next account evaluates the link against its own evidence');
 console.log('PASS interview study continuation: bounded Unicode/auth routes, durable learning filters, exact six-module groups across papers, real-grade next steps, missing/current mismatch recovery and account-safe unresolved progress');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
 for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
