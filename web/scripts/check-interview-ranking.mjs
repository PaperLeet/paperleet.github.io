import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {readFileSync} from 'node:fs';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/'});
const reveals=[];dom.window.HTMLElement.prototype.scrollIntoView=function(options){reveals.push({node:this,options});};
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
 const {default:CompanySources}=await server.ssrLoadModule('/src/InterviewCompanySources.tsx');
 const {default:Hub}=await server.ssrLoadModule('/src/Interviews.tsx');
 const {emptyInterviewFilters,readInterviewRoute,interviewHref}=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const {libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const search=await server.ssrLoadModule('/src/interviewSearch.ts');
 const {interviewCatalog}=await server.ssrLoadModule('/src/interviewModels.ts');
 const problems=['architecture/attention','finetuning/lora','optimization/adamw','architecture/rope'].map((id,index)=>({id,number:index+1,title:['注意力','低秩适配','AdamW','旋转位置编码'][index],translations:{en:{title:['Attention','LoRA','AdamW','RoPE'][index]}},starter:'initial code',difficulty:index===1?'H':'M',estimated_minutes:25,module:{goal:'练习',goal_en:'Practice'},tests:{T0:1}}));
 const evidence=(problem,index)=>({problem_id:problem.id,reports:10-index,contributors:6-index,companies:[{company:'google',reports:10-index,contributors:6-index,problem_id:problem.id,latest:'2026-10-15'}]});
 const questions=problems.map(evidence),passed=problem=>({problem:problem.id,summary:{T0:[1,1]},tests:[{id:'passed',layer:'T0',passed:true}]});
 const saved={[problems[0].id]:{acceptedReport:passed(problems[0]),report:{...passed(problems[0]),error:'later failure'}},[problems[1].id]:{code:'older draft',updated:'2026-10-15T10:00:00Z'},[problems[2].id]:{code:'',updated:'2026-10-15T11:00:00Z'}};
 // Search spans independently maintained bilingual fields, while company evidence stays explicit.
 const searchable={...problems[0],paper:{title:'Attention Is All You Need'},module:{goal:'推导缩放与掩码',goal_en:'Derive scaling and causal masks'}};
 for(const q of ['attention','注意力','ＡＴＴＥＮＴＩＯＮ','architecture_attention','1','causal masks','缩放','All You Need'])assert(search.interviewConceptMatch(searchable.id,searchable,questions[0],q),q);
 assert.deepEqual(search.interviewConceptMatch(searchable.id,searchable,questions[0],'Google causal'),['goal','company']);
 assert.deepEqual(search.interviewConceptMatch(searchable.id,searchable,questions[0],'Need 注意力'),['title','paper']);
 assert.equal(search.interviewConceptMatch(searchable.id,searchable,undefined,'Google'),undefined,'A curated module never inherits the selected company as evidence');
 for(const q of ['Microsoft attention','Google LoRA','[attention]','.*','%','<img>'])assert.equal(search.interviewConceptMatch(searchable.id,searchable,questions[0],q),undefined,q);
 assert.deepEqual(search.interviewConceptMatch(searchable.id,undefined,questions[0],'google architecture attention'),['module','company'],'Real company sources remain searchable when the catalog is unavailable');
 assert.deepEqual(search.matchingInterviewCompanies('Ｇｏｏｇｌｅ DeepMind').map(company=>company.id),['google']);
 assert.deepEqual(search.matchingInterviewCompanies('微软 Microsoft').map(company=>company.id),['microsoft']);
 assert.deepEqual(search.matchingInterviewCompanies(''),[]);
 assert.equal(search.interviewConceptMatch(searchable.id,searchable,{...questions[0],companies:[{...questions[0].companies[0],reports:0,contributors:0}]},'Google'),undefined,'A zero-count company row is not evidence');
 const filters={...emptyInterviewFilters(),view:'questions',company:'google',role:'llm',stage:'technical',days:'90',q:'',track:'llm',saved:true};
 let progressRetries=0,catalogRetries=0,searchClears=0;
 const props={filters,questions,problems,saved,ready:true,onRetryProgress:()=>progressRetries++,onRetryCatalog:()=>catalogRetries++,onClearSearch:()=>searchClears++};
 let setFixtureLocale;function LocaleControl(){const {setLocale}=useI18n();setFixtureLocale=setLocale;return React.createElement('button',{onClick:()=>setLocale('zh')},'Change locale');}
 const render=async(Component,extra={})=>act(async()=>{if(Component===Hub&&extra.route!==undefined)window.history.replaceState(null,'','#/'+extra.route.replace(/^#\/?/,''));if(!root)root=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(LocaleControl),React.createElement(Component,{...props,...extra})));});
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
 const manyCompanies=['google','meta','microsoft','nvidia','amazon'].map(company=>({...questions[0].companies[0],company,reports:2,contributors:2}));
 await render(Ranking,{questions:[{...questions[0],companies:manyCompanies},...questions.slice(1)]});
 const companyArea=rows()[0].querySelector('.interview-company-tags'),companyDisclosure=companyArea.querySelector('details');assert(companyDisclosure,'Many company sources need a compact native disclosure');assert.equal(companyArea.querySelectorAll(':scope>a').length,3);assert.equal(companyDisclosure.querySelectorAll('a').length,2);
 const summaryControl=companyDisclosure.querySelector('summary');summaryControl.focus();await click(summaryControl);assert(companyDisclosure.open);assert.equal(document.activeElement,summaryControl);for(const companyLink of companyArea.querySelectorAll('a'))assert.deepEqual(readInterviewRoute(companyLink.getAttribute('href')),{...filters,view:'reports',company:manyCompanies.find(row=>companyLink.getAttribute('href').includes('company='+row.company)).company,concept:problems[0].id,saved:false,page:1,q:''});
 const lastCompany=companyDisclosure.querySelectorAll('a')[1];lastCompany.focus();await click(summaryControl);assert(!companyDisclosure.open);assert.equal(document.activeElement,summaryControl,'Collapsing must not strand focus inside hidden source links');
 const companyStyle=document.createElement('style');companyStyle.textContent=readFileSync('src/interview-company-sources.css','utf8')+'\n'+readFileSync('src/interviews.css','utf8');document.head.append(companyStyle);assert.equal(window.getComputedStyle(companyDisclosure.querySelector('.interview-company-extra-links')).display,'none','Closed source links stay out of the layout even with parent styles loaded later');
 function SourcePairs({companies=manyCompanies}){return React.createElement(React.Fragment,null,React.createElement('input',{id:'newer-source-focus'}),React.createElement(CompanySources,{companies,problemId:problems[0].id,name:'Attention',filters}),React.createElement(CompanySources,{companies,problemId:problems[0].id,name:'Attention',filters}));}
 await render(SourcePairs);const paired=[...document.querySelectorAll('.interview-company-more')],pairControl=paired[0].querySelector('summary');assert.notEqual(pairControl.getAttribute('aria-controls'),paired[1].querySelector('summary').getAttribute('aria-controls'));assert.equal(document.getElementById(pairControl.getAttribute('aria-controls')),paired[0].querySelector('.interview-company-extra-links'));
 const disclosureHref=location.href,storageSet=window.Storage.prototype.setItem;try{window.Storage.prototype.setItem=()=>{throw Error('Disclosing public sources must not write storage');};pairControl.focus();await click(pairControl);assert(paired[0].open&&!paired[1].open);assert.equal(document.activeElement,pairControl);assert.equal(location.href,disclosureHref);}finally{window.Storage.prototype.setItem=storageSet;}
 const inOrder=[...document.querySelectorAll('a,summary,input')];assert.equal(inOrder[inOrder.indexOf(pairControl)+1],paired[0].querySelector('a'),'Native Tab order continues into the newly revealed sources');
 await act(async()=>setFixtureLocale('zh'));assert(paired[0].open);assert(document.querySelector('.company-source-list').textContent.includes('微软'));assert.equal(window.getComputedStyle(pairControl.querySelector('.company-source-more-label')).display,'none');assert.equal(window.getComputedStyle(pairControl.querySelector('.company-source-less-label')).display,'inline');await act(async()=>setFixtureLocale('en'));assert(paired[0].open);
 await render(SourcePairs,{companies:manyCompanies.map(row=>({...row,reports:3}))});assert(paired[0].open,'Fresh count objects cannot close an active source disclosure');assert.equal(paired[0].querySelector('small').textContent,'3');
 const hiddenSource=paired[0].querySelector('a'),newerSourceFocus=document.getElementById('newer-source-focus');hiddenSource.focus();await act(async()=>{pairControl.click();newerSourceFocus.focus();});assert.equal(document.activeElement,newerSourceFocus,'Collapse yields to a newer focus choice');pairControl.focus();await click(pairControl);hiddenSource.focus();await act(async()=>{pairControl.click();location.hash='#/community';});assert.notEqual(document.activeElement,pairControl,'A route change cancels source focus repair');location.hash=disclosureHref.split('#')[1]||'';
 await render(CompanySources,{companies:manyCompanies.slice(0,3),problemId:problems[0].id});assert.equal(document.querySelectorAll('.company-source-list>a').length,3);assert(!document.querySelector('.interview-company-more'),'Three sources need no extra control');await render(CompanySources,{companies:[],problemId:problems[0].id});assert(!document.querySelector('.company-source-list'));await render(CompanySources,{companies:manyCompanies.map(row=>({...row,reports:0,contributors:0})),problemId:problems[0].id});assert(!document.querySelector('.company-source-list'),'Zero mentions cannot become company evidence');companyStyle.remove();
 console.log('PASS company source disclosure: three visible tags, native full-source reading and Tab order, independent unique regions, exact counts/routes, bilingual/fresh-count continuity, hidden-link focus repair, newer route/focus priority and no disclosure storage writes');
 await render(Ranking);
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
 await render(Ranking,{filters:{...filters,q:'Google 注意力'}});assert.equal(rows().length,1);assert(rows()[0].querySelector('.interview-search-match').textContent.includes('Source company'));assert.equal(rows()[0].querySelector('.interview-rank').textContent,'01');
 await render(Ranking,{questions:[questions[0],{...questions[1],companies:[{...questions[1].companies[0],company:'microsoft'}]}],filters:{...filters,company:'',q:'Microsoft LoRA'}});assert.equal(rows().length,1);assert.equal(rows()[0].querySelector('.interview-rank').textContent,'02','Multi-field search preserves the original evidence ranking');
 assert(rows()[0].querySelector('.interview-search-match').textContent.includes('Title · Source company'));
 await render(Ranking,{problems:[],ready:false,catalogError:'offline',filters:{...filters,q:'Google'}});assert.equal(rows().length,4);assert(!document.querySelector('.interview-ranking-actions > .button'));assert.equal(document.querySelectorAll('.interview-ranking-evidence a').length,4);
 await render(Ranking,{problems:[searchable,...problems.slice(1)],filters:{...filters,q:'Need scaling'}});assert.equal(rows().length,1);assert(rows()[0].querySelector('.interview-search-paper').textContent.includes('Attention Is All You Need'));
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
 await render(Hub,{route:interviewHref({...hubFilters,q:'Ｇｏｏｇｌｅ attention'}).slice(2),progressReady:true});
 const searchInput=document.querySelector('.interview-search input');assert.equal(searchInput.getAttribute('aria-label'),'Search concepts or source companies');const searchDescriptions=searchInput.getAttribute('aria-describedby').split(/\s+/).map(id=>document.getElementById(id));assert(searchDescriptions.every(Boolean));assert(searchDescriptions.some(node=>node.textContent.includes('Separate keywords')));assert(searchDescriptions.some(node=>node.textContent.includes('200 characters')));
 const clear=document.querySelector('.interview-search button[aria-label="Clear search"]');clear.focus();await click(clear);assert.equal(document.activeElement,searchInput,'Clearing search restores keyboard focus to the input');assert.equal(searchInput.value,'');assert.deepEqual(readInterviewRoute(location.hash),hubFilters,'Clear only the query, retaining company, role, time and study context');
 await render(Hub,{route:interviewHref({...hubFilters,view:'companies',company:'',q:'Ｇｏｏｇｌｅ DeepMind'}).slice(2),progressReady:true});assert.equal(document.querySelectorAll('.interview-company-card').length,1);assert(document.querySelector('.interview-company-card').textContent.includes('Google'));assert.equal(document.querySelector('.interview-search input').getAttribute('aria-label'),'Search companies');
 await render(Hub,{route:interviewHref(hubFilters).slice(2),progressReady:true});
 const resultsHeading=document.querySelector('.interview-section-intro'),beforeRecovery=reveals.length;await click(filter('Passed'));await render(Hub,{route:location.hash.replace(/^#\/?/,''),progressReady:true});assert.equal(rows().length,1);assert(document.activeElement===resultsHeading);assert(reveals.slice(beforeRecovery).some(row=>row.node===resultsHeading));
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
 await render(Ranking,{questions:[],problems:catalog,filters:{...filters,q:'Google',study:'started',queue:'1,2'},saved,ready:true});
 assert.equal(rows().length,0,'Company preparation links never fabricate concept evidence in a curated list');
 const companyPreparation=document.querySelector('.interview-search-companies a');assert(companyPreparation);assert.deepEqual(readInterviewRoute(companyPreparation.getAttribute('href')),{...filters,q:'',study:'started',queue:'',page:1});assert(!document.querySelector('.interview-company-tags a'));
 await render(Ranking,{questions:[],problems:catalog,filters:{...filters,q:'Microsoft'},saved,ready:true});assert.equal(readInterviewRoute(document.querySelector('.interview-search-companies a').getAttribute('href')).company,'microsoft');
 await render(Ranking,{questions:[],problems:catalog,filters:{...filters,q:'a'},saved,ready:true});const companyChoices=[...document.querySelectorAll('.interview-search-companies a')];assert.equal(companyChoices.length,5,'Broad searches show four compact company choices and an all-results link');const allCompanies=readInterviewRoute(companyChoices.at(-1).getAttribute('href'));assert.equal(allCompanies.view,'companies');assert.equal(allCompanies.q,'a');assert.equal(allCompanies.company,'');assert.equal(allCompanies.role,filters.role);assert.equal(allCompanies.saved,false);
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
 console.log('PASS interview ranking: literal bilingual multi-field/all-keyword search, evidence-only companies, full-width company names, explicit company preparation, search-clear focus, preserved contributor ranks and context, real progress and retained passes, latest draft resume, stable contributor ranks, per-state recovery, catalog failure source retention, incomplete search, scoped company reports, lazy bilingual study prompts, account/route isolation, honest role-based empty-state curriculum, compact pagination, incomplete-suite rejection and read-only controls');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
 for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
