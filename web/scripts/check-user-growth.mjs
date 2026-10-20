import assert from 'node:assert/strict';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom = new JSDOM('<div id="root"></div>', {url:'https://paperleet.test/'});
const names = ['window','document','location','localStorage','IS_REACT_ACT_ENVIRONMENT','__growthFetch'];
const originals = new Map(names.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
Object.assign(globalThis,{window:dom.window,document:dom.window.document,location:dom.window.location,localStorage:dom.window.localStorage,IS_REACT_ACT_ENVIRONMENT:true});
localStorage.setItem('paperleet.locale','en');
// Advance only this chart's owned daily callback; no wall-clock waits or network calls.
const intervals=new Map();let intervalId=0;
window.setInterval=(callback,milliseconds)=>{const id=++intervalId;intervals.set(id,{callback,milliseconds});return id;};
window.clearInterval=id=>intervals.delete(id);
const daily=()=>{assert.equal(intervals.size,1,'Exactly one owned refresh timer');const timer=[...intervals.values()][0];assert.equal(timer.milliseconds,24*60*60_000);return timer.callback;};

const React=await import('react'),{act}=React,{createRoot}=await import('react-dom/client');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
  name:'growth-http-fixture',enforce:'pre',resolveId(source,importer){if(source==='./apiClient'&&importer?.endsWith('/UserGrowthChart.tsx'))return '\0growth-http';},
  load(id){if(id==='\0growth-http')return 'export const apiFetch=(...args)=>globalThis.__growthFetch(...args);';}
}]});
let root;
try {
  const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
  const {default:Chart}=await server.ssrLoadModule('/src/UserGrowthChart.tsx');
  const {readRegistrationGrowth}=await server.ssrLoadModule('/src/registrationGrowth.ts');
  const sample={total_users:3,new_users_today:2,timezone:'UTC',start_date:'2026-10-18',end_date:'2026-10-20',updated_at:'2026-10-20T12:00:00Z',days:[
    {date:'2026-10-18',new_users:1,total_users:1},{date:'2026-10-19',new_users:0,total_users:1},{date:'2026-10-20',new_users:2,total_users:3},
  ]};
  const requests=[];let result=sample,fail=true,pending=false,resolvePending,setLocale;
  globalThis.__growthFetch=async(path,options)=>{requests.push({path,options});if(pending)return new Promise(resolve=>{resolvePending=resolve;});return{ok:!fail,json:async()=>result};};
  function Control(){setLocale=useI18n().setLocale;return null;}
  const container=document.querySelector('#root');root=createRoot(container);
  const render=key=>root.render(React.createElement(LocaleProvider,null,React.createElement(Control),React.createElement(Chart,{key})));
  await act(async()=>render('normal'));
  assert.match(container.textContent,/Growth chart is unavailable/);assert.equal(container.querySelector('.growth-total strong').textContent,'—');assert(!container.querySelector('.growth-line'));
  fail=false;await act(async()=>container.querySelector('button').click());
  assert.equal(container.querySelector('.growth-total strong').textContent,'3');
  assert.equal(requests.at(-1).path,'/api/community/registration-growth');assert.equal(requests.at(-1).options.credentials,'omit');
  const scheduled=daily(),beforeEvents=requests.length;
  await act(async()=>{window.dispatchEvent(new window.Event('focus'));window.dispatchEvent(new window.Event('online'));document.dispatchEvent(new window.Event('visibilitychange'));});
  assert.equal(requests.length,beforeEvents,'Focus/online/visibility do not refresh the daily aggregate');assert(daily()===scheduled);
  await act(async()=>scheduled());assert.equal(requests.length,beforeEvents+1,'The owned 24-hour callback refreshes once');
  fail=true;await act(async()=>scheduled());assert.equal(container.querySelector('.growth-total strong').textContent,'3');assert(container.querySelector('.growth-line'));assert.match(container.textContent,/Update unavailable/);
  fail=false;const beforeRetry=requests.length;await act(async()=>container.querySelector('.growth-recovery button').click());assert.equal(requests.length,beforeRetry+1,'Explicit retry still refreshes immediately');assert(daily()!==scheduled,'Retry replaces and cleans the prior timer');
  console.log('PASS daily growth refresh: actual 24-hour timer, no focus/online/visibility refresh, retained chart on scheduled failure, immediate explicit retry and timer replacement');
  const slider=container.querySelector('[role=slider]');
  assert.match(slider.getAttribute('aria-valuetext'),/2026-10-20: 3 registered, 2 joined/);
  await act(async()=>slider.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Home',bubbles:true})));
  assert.match(slider.getAttribute('aria-valuetext'),/2026-10-18: 1 registered, 1 joined/);
  await act(async()=>slider.dispatchEvent(new window.KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true})));
  assert.match(container.querySelector('.growth-day').textContent,/Oct 19\+0 joined/);
  // Touch selection uses the chart's coordinates, including the plot's margins.
  slider.getBoundingClientRect=()=>({left:0,width:260});
  const props=slider[Object.keys(slider).find(key=>key.startsWith('__reactProps$'))];
  await act(async()=>props.onPointerDown({currentTarget:slider,clientX:246,pointerType:'touch'}));
  assert.equal(slider.getAttribute('aria-valuenow'),'2');
  await act(async()=>setLocale('zh'));assert.match(container.textContent,/位注册用户/);assert.match(slider.getAttribute('aria-valuetext'),/当天新增 2 人/);
  for(const bad of [{...sample,total_users:4},{...sample,days:[sample.days[0],sample.days[2]]},{...sample,days:[{...sample.days[0],new_users:NaN},...sample.days.slice(1)]}])assert.throws(()=>readRegistrationGrowth(bad));
  result={...sample,total_users:0,new_users_today:0,start_date:'2026-10-20',days:[{date:'2026-10-20',new_users:0,total_users:0}]};
  await act(async()=>render('zero'));
  assert.equal(container.querySelector('.growth-total strong').textContent,'0');
  assert(!/NaN|Infinity/.test(container.querySelector('.growth-line').getAttribute('points')));
  pending=true;await act(async()=>render('pending'));const signal=requests.at(-1).options.signal;
  const retiredTimer=daily();
  await act(async()=>root.unmount());root=null;assert(signal.aborted);assert.equal(intervals.size,0);const beforeRetired=requests.length;await act(async()=>retiredTimer());assert.equal(requests.length,beforeRetired,'A retired daily callback cannot issue another request');
  await act(async()=>resolvePending({ok:true,json:async()=>sample}));
  assert.equal(container.textContent,'');
  console.log('PASS user growth: real chart, cumulative and daily counts, zero-day continuity, keyboard/touch, locale, honest unavailable/zero states, private-cookie-free request, invalid-series rejection and aborted unmount');
} finally {
  if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
  for(const[name,descriptor]of originals){if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];}
}
