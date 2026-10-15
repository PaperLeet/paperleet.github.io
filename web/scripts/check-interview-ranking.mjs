import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {readFileSync} from 'node:fs';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/'});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__rankingFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0')},plugins:[{
 name:'ranking-account-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0ranking-account-fixture';},
 load(id){if(id==='\0ranking-account-fixture')return `import React from 'react';
 export const useAccount=()=>globalThis.__rankingFixture.account;
 export const socialApi=(...args)=>globalThis.__rankingFixture.request(...args);
 export const errorMessage=e=>e.message;export class SocialError extends Error{}
 export const PersonLink=({user})=>React.createElement('span',null,user.display_name);
 export const DateLabel=({value})=>React.createElement('time',null,value);
 export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);`;}
}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Ranking}=await server.ssrLoadModule('/src/InterviewQuestionRanking.tsx');
 const {default:Hub}=await server.ssrLoadModule('/src/Interviews.tsx');
 const {emptyInterviewFilters,readInterviewRoute,interviewHref}=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const {libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const {interviewCatalog}=await server.ssrLoadModule('/src/interviewModels.ts');
 const problems=['architecture/attention','finetuning/lora','optimization/adamw','architecture/rope'].map((id,index)=>({id,number:index+1,title:['注意力','低秩适配','AdamW','旋转位置编码'][index],translations:{en:{title:['Attention','LoRA','AdamW','RoPE'][index]}},starter:'initial code',difficulty:index===1?'H':'M',estimated_minutes:25,module:{goal:'练习',goal_en:'Practice'},tests:{T0:1}}));
 const evidence=(problem,index)=>({problem_id:problem.id,reports:10-index,contributors:6-index,companies:[{company:'google',reports:10-index,contributors:6-index,problem_id:problem.id,latest:'2026-10-15'}]});
 const questions=problems.map(evidence),passed=problem=>({problem:problem.id,summary:{T0:[1,1]},tests:[{id:'passed',layer:'T0',passed:true}]});
 const saved={[problems[0].id]:{acceptedReport:passed(problems[0]),report:{...passed(problems[0]),error:'later failure'}},[problems[1].id]:{code:'older draft',updated:'2026-10-15T10:00:00Z'},[problems[2].id]:{code:'',updated:'2026-10-15T11:00:00Z'}};
 const filters={...emptyInterviewFilters(),view:'questions',company:'google',role:'llm',stage:'technical',days:'90',q:'',track:'llm',saved:true};
 let progressRetries=0,catalogRetries=0,searchClears=0;
 const props={filters,questions,problems,saved,ready:true,onRetryProgress:()=>progressRetries++,onRetryCatalog:()=>catalogRetries++,onClearSearch:()=>searchClears++};
 function LocaleControl(){const {setLocale}=useI18n();return React.createElement('button',{onClick:()=>setLocale('zh')},'Change locale');}
 const render=async(Component,extra={})=>act(async()=>{if(!root)root=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(LocaleControl),React.createElement(Component,{...props,...extra})));});
 const click=async element=>{assert(element,'Missing control');await act(async()=>element.click());};
 const filter=label=>[...document.querySelectorAll('.interview-ranking-filters button')].find(button=>button.querySelector('span').textContent===label);
 const rows=()=>[...document.querySelectorAll('.interview-ranking-row')];
 const stateLabels=()=>[...document.querySelectorAll('.interview-ranking-state')].map(node=>node.textContent);
 await render(Ranking,{ready:false});
 assert.deepEqual(stateLabels(),Array(4).fill('Progress not ready'));
 assert(!document.querySelector('progress,.interview-ranking-resume'));
 assert.equal(document.querySelectorAll('.interview-ranking-filters button:disabled').length,3);
 assert.equal(document.querySelectorAll('.preparation-prompt-list').length,0,'Guides remain closed until chosen');
 await render(Ranking);
 assert.deepEqual(stateLabels(),['Passed','In progress','In progress','Not started']);
 assert.equal(document.querySelector('progress').value,1);assert.equal(document.querySelector('progress').max,4);
 assert(document.querySelector('.interview-ranking-resume').href.includes('#/3?'),'Continue the latest unfinished draft, including deliberately empty code');
 assert.deepEqual(readInterviewRoute(libraryReturnHref(document.querySelector('.interview-ranking-resume').getAttribute('href'))),{...filters,queue:'1,2,3,4'},'Practice preserves the source view and account return context');
 const source=document.querySelector('.interview-ranking-evidence a');
 assert.deepEqual(readInterviewRoute(source.getAttribute('href')),{...filters,view:'reports',concept:problems[0].id,saved:false,page:1,q:''});
 const companySource=document.querySelector('.interview-company-tags a');
 assert.deepEqual(readInterviewRoute(companySource.getAttribute('href')),{...filters,view:'reports',company:'google',concept:problems[0].id,saved:false,page:1,q:''},'Company chips keep the exact concept, role, stage and publication period');
 await click(filter('In progress'));assert.equal(rows().length,2);assert.deepEqual(rows().map(row=>row.querySelector('.interview-rank').textContent),['02','03'],'Learning filters never renumber or reweight community ranks');
 assert(rows().every(row=>row.textContent.includes('Resume practice')));
 await click(filter('Passed'));assert.equal(rows().length,1);assert(rows()[0].textContent.includes('Review'));assert(!document.querySelector('.interview-ranking-resume'));
 await click(filter('Not started'));assert.equal(rows().length,1);assert(rows()[0].textContent.includes('RoPE'));assert(document.querySelector('.interview-ranking-resume').textContent.includes('Start practicing'));
 await render(Ranking,{saved:{}});assert.equal(rows().length,4,'Real progress updates the selected state without mutating public evidence');
 await click(filter('Passed'));assert.equal(rows().length,0);assert(document.body.textContent.includes('No concepts with this learning state'));
 await click(document.querySelector('.interview-empty button'));assert.equal(rows().length,4);
 await render(Ranking,{error:'offline'});assert(!document.querySelector('progress,.interview-ranking-resume'));assert(stateLabels().every(label=>label==='Progress not ready'));
 await click(document.querySelector('.interview-progress-status button'));assert.equal(progressRetries,1);assert.equal(rows().length,4);
 await render(Ranking,{problems:[],ready:false,catalogError:'offline'});assert.equal(rows().length,4,'A catalog failure keeps real source reports, rather than fabricating an empty ranking');
 assert.equal(document.querySelectorAll('.interview-ranking-actions > .button').length,0);assert.equal(document.querySelectorAll('.interview-ranking-evidence a').length,4);
 await click([...document.querySelectorAll('button')].find(button=>button.textContent.includes('Retry catalog')));assert.equal(catalogRetries,1);
 await render(Ranking,{problems:[],ready:false,catalogError:'offline',filters:{...filters,q:'注意力'}});assert(document.body.textContent.includes('Search is waiting for practice details'));assert(!document.body.textContent.includes('No matching concepts'));
 await click(document.querySelector('.interview-empty button'));assert.equal(searchClears,1);
 await render(Ranking,{filters:{...filters,q:'注意力'}});assert.equal(rows().length,1);assert(rows()[0].textContent.includes('Attention'));
 await render(Ranking,{filters:{...filters,q:'not-a-topic'}});assert.equal(rows().length,0);assert(document.body.textContent.includes('No matching concepts'));
 await render(Ranking,{filters:{...filters,q:'Attention'}});
 const guide=document.querySelector('.preparation-guide');await act(async()=>{guide.open=true;guide.dispatchEvent(new dom.window.Event('toggle'));await new Promise(resolve=>setTimeout(resolve,30));});
 assert.equal(document.querySelectorAll('.preparation-prompt-list > li').length,3);
 assert([...document.querySelectorAll('.preparation-hint')].every(hint=>!hint.open));
 assert.equal(document.querySelector('progress').value,1,'Opening prompts never awards a pass');
 await click([...document.querySelectorAll('button')].find(button=>button.textContent==='Change locale'));assert(document.body.textContent.includes('已通过'));assert(document.querySelector('.preparation-prompt-list').textContent.includes('注意力')||document.querySelector('.preparation-prompt-list').textContent.includes('缩放'));
 // The real hub keys local state by account and route; old selections cannot cross owners.
 await act(async()=>root.unmount());root=undefined;localStorage.setItem('paperleet.locale','en');
 const summary={reports:30,contributors:10,days:'90',basis:'published-community-reports',generated_at:'2026-10-15',companies:[{company:'google',reports:30,contributors:10}],questions};
 const requests=[];
 globalThis.__rankingFixture={account:{user:{id:'alice'}},request:async(path,body)=>{assert.equal(body,undefined,'The ranking is read-only');requests.push(path);return summary;}};
 const hubFilters={...filters,saved:false};
 await render(Hub,{route:interviewHref(hubFilters).slice(2),progressReady:true});
 await click(filter('Passed'));assert.equal(rows().length,1);
 globalThis.__rankingFixture.account={user:{id:'bob'}};
 await render(Hub,{route:interviewHref(hubFilters).slice(2),saved:{},progressReady:false});
 assert.equal(rows().length,4);assert.equal(filter('All concepts').getAttribute('aria-pressed'),'true');assert(!document.querySelector('.interview-ranking-completion'));
 await render(Hub,{route:interviewHref(hubFilters).slice(2),saved:{},progressReady:true});assert.equal(document.querySelector('.interview-ranking-completion progress').value,0);
 await click(filter('Not started'));
 await render(Hub,{route:interviewHref({...hubFilters,company:'amazon'}).slice(2),saved,progressReady:true});assert.equal(filter('All concepts').getAttribute('aria-pressed'),'true','Route changes reset local study filters');
 assert(requests.every(path=>path.startsWith('community/interviews/summary?')),'Study controls never mutate accounts, grades or public reports');
 // Empty live statistics still provide useful preparation, without invented evidence.
 const catalog=JSON.parse(readFileSync(new URL('../public/data/index.json',import.meta.url),'utf8')).problems;
 const allIds=[...new Set(interviewCatalog.paths.flatMap(path=>path.modules))];
 await render(Ranking,{questions:[],problems:catalog,filters:{...filters,role:'',saved:false},saved:{},ready:true});
 assert(document.querySelector('.interview-curated-context').textContent.includes('CURATED PRACTICE'));
 assert.equal(rows().length,6);assert(document.body.textContent.includes('Showing 6 of '+allIds.length));
 assert(!document.querySelector('.interview-ranking-evidence,.interview-company-tags a'),'Editorial preparation has no fabricated contributors, company tags or source counts');
 assert(rows().every(row=>row.querySelector('.interview-rank').getAttribute('aria-label')==='Curated exercise'),'Curriculum positions are not presented as community ranks');
 const more=document.querySelector('.interview-ranking-more');more.focus();await click(more);assert.equal(rows().length,12);
 assert.equal(document.activeElement,rows()[6].querySelector('.interview-ranking-title a'),'Keyboard reveal continues at the first newly available concept');
 const outside=[...document.querySelectorAll('button')].find(button=>button.textContent==='Change locale');
 await act(async()=>{const button=document.querySelector('.interview-ranking-more');button.focus();button.click();outside.focus();});
 assert.equal(document.activeElement,outside,'A later external focus is never stolen by list expansion');
 while(document.querySelector('.interview-ranking-more'))await click(document.querySelector('.interview-ranking-more'));
 assert.equal(rows().length,allIds.length);
 await click(filter('Not started'));assert.equal(rows().length,6,'Changing state returns to a compact first page');
 await render(Ranking,{questions:[],problems:catalog,filters:{...filters,role:'recommendation',saved:false},saved:{},ready:true});
 await click(filter('All concepts'));
 while(document.querySelector('.interview-ranking-more'))await click(document.querySelector('.interview-ranking-more'));
 const recommendedIds=new Set(interviewCatalog.paths.filter(path=>path.roles.includes('recommendation')).flatMap(path=>path.modules));
 assert.equal(rows().length,recommendedIds.size);
 assert.deepEqual(rows().map(row=>decodeURIComponent(row.querySelector('.interview-ranking-title a').id.slice('interview-ranked-'.length))).sort(),[...recommendedIds].sort(),'The preparation fallback follows the actual role curriculum');
 await render(Ranking,{questions:[],problems:catalog,filters:{...filters,concept:'architecture/attention'},saved,ready:true});
 assert.equal(rows().length,1);assert(rows()[0].textContent.includes('Attention'));
 assert.equal(document.querySelector('progress').value,0,'A one-case fixture cannot count as passing the real catalog suite');
 const attention=catalog.find(problem=>problem.id==='architecture/attention');
 const completeReport={
  problem:attention.id,
  summary:Object.fromEntries(Object.entries(attention.tests).map(([layer,count])=>[layer,[count,count]])),
  tests:Object.entries(attention.tests).flatMap(([layer,count])=>Array.from({length:count},(_,index)=>({id:layer+index,layer,passed:true}))),
 };
 const completeSaved={...saved,[attention.id]:{acceptedReport:completeReport}};
 await render(Ranking,{questions:[],problems:catalog,filters:{...filters,concept:attention.id},saved:completeSaved,ready:true});
 assert.equal(document.querySelector('progress').value,1);assert.equal(document.querySelector('progress').max,1);
 await render(Ranking,{questions:[],problems:[attention],filters:{...filters,role:''},saved:completeSaved,ready:true,catalogError:'offline'});
 assert.equal(document.querySelector('progress').value,1);assert.equal(document.querySelector('progress').max,allIds.length,'Unavailable editorial modules never count as completed');
 await render(Ranking,{questions:[],problems:[],filters:{...filters,role:''},saved,ready:true,catalogError:'offline'});
 assert.equal(document.querySelectorAll('.interview-ranking-filters button:disabled').length,3);assert(!document.querySelector('progress'));
 await render(Ranking,{questions,problems,filters,saved,ready:true});
 assert(!document.querySelector('.interview-curated-context'),'New real evidence replaces the editorial fallback');
 assert(document.querySelector('.interview-ranking-evidence a'));
 console.log('PASS interview ranking: real progress and retained passes, latest draft resume, stable contributor ranks, per-state recovery, catalog failure source retention, incomplete search, scoped company reports, lazy bilingual study prompts, account/route isolation, honest role-based empty-state curriculum, compact pagination, incomplete-suite rejection and read-only controls');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
 for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
