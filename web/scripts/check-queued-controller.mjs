import assert from 'node:assert/strict';
import {createServer} from 'vite';
const vite=await createServer({server:{middlewareMode:true},appType:'custom'});
try {
 const {QueuedJudgeController,pendingKey}=await vite.ssrLoadModule('/src/queuedJudgeController.ts');
 const id='a'.repeat(32),client='b'.repeat(32),owner='user:alice',problem='architecture/attention',code='# captured original\n';
 const queued={id,problem,status:'queued',ahead:1};
 const report={problem,summary:{T0:[1,1]},tests:[{id:'one',layer:'T0',passed:true}]};
 const success={...queued,status:'succeeded',ahead:0,report};
 const full=()=>Response.json({error:'full',code:'queue_full'},{status:429,headers:{'Retry-After':'5'}});
 const defer=()=>{let resolve;const promise=new Promise(r=>{resolve=r;});return {promise,resolve};};
 const tick=async()=>{for(let i=0;i<16;i++)await Promise.resolve();};
 const make=(extra={})=>{
  const values=new Map(),views=[],reports=[];
  const storage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
  const control=new QueuedJudgeController({owner,client,expectedUser:'alice',storage,alive:()=>true,uuid:()=>id,onView:view=>views.push(view),onReport:(r,o)=>reports.push([r,o]),pollWait:async()=>{},...extra});
  return {control,values,views,reports};
 };
 const {submissionModuleLoadError}=await vite.ssrLoadModule('/src/judgeJobs.ts');
 const admission=await vite.ssrLoadModule('/src/queueAdmission.ts');
 let loadFails=true,loadAttempts=0,loadPosts=0,loadGets=0;
 const failedLoad=make({loadAdmission:async()=>{
  loadAttempts++;if(loadFails)throw new TypeError('Failed to fetch dynamically imported module');return admission;
 },fetcher:async(url,options)=>{
  if(options.method==='GET'){loadGets++;return Response.json(success);}
  loadPosts++;return Response.json(queued,{status:202});
 }});
 for(let attempt=1;attempt<=2;attempt++){
  await failedLoad.control.submit(problem,code);await tick();
  assert.equal(loadAttempts,attempt,'An explicit retry may retry loading, but cannot submit through a missing module');
  assert.equal(loadPosts,0);assert.equal(loadGets,0,'A failed lazy import must not look up a nonexistent server job');
  assert.equal(failedLoad.values.get(pendingKey(owner)),undefined,'No request receipt exists before admission loads');
  assert.equal(failedLoad.views.at(-1).error,submissionModuleLoadError);
  assert.equal(failedLoad.views.at(-1).reloadRequired,true);
  assert(!failedLoad.views.at(-1).pending&&!failedLoad.views.at(-1).reconnecting);
  assert.equal(failedLoad.reports.length,0);
 }
 loadFails=false;
 await failedLoad.control.submit(problem,code);await tick();
 assert.equal(loadPosts,1);assert.equal(loadGets,1);assert.equal(failedLoad.reports.length,1);
 assert(!failedLoad.views.at(-1).reloadRequired&&!failedLoad.views.at(-1).error,'Loading recovery clears the stale failure');
 assert.equal(failedLoad.values.get(pendingKey(owner)),undefined);failedLoad.control.dispose();

 const waiting=defer(),release=defer();let posts=0;
 const first=make({fetcher:async(url,options)=>{
  if(options.method==='GET')return Response.json(success);
  posts++;assert.equal(JSON.parse(options.body).code,code);if(posts===2)assert.equal(first.views.at(-1).waiting.attempting,true,'Keep the source review mounted while retrying');return posts===1?full():Response.json(queued,{status:202});
 },wait:async()=>{waiting.resolve();await release.promise;}});
 const pending=first.control.submit(problem,code);await waiting.promise;
 assert.equal(first.views.at(-1).waiting.source,code);
 assert.equal(first.values.get(pendingKey(owner)),undefined);
 await first.control.submit(problem,'new code');assert.equal(posts,1);
 release.resolve();await pending;await tick();assert.equal(posts,2);assert.equal(first.reports.length,1);
 assert.equal(first.reports[0][1],owner);assert(!first.views.at(-1).pending);first.control.dispose();

 const stopReady=defer();let stoppedPosts=0;
 const stopped=make({fetcher:async()=>{stoppedPosts++;return full();},wait:async(ms,signal)=>{stopReady.resolve();await new Promise((_,reject)=>{if(signal.aborted)reject(signal.reason);else signal.addEventListener('abort',()=>reject(signal.reason),{once:true});});}});
 const task=stopped.control.submit(problem,code);await stopReady.promise;
 assert(stopped.control.stopWaiting());await task;assert(!stopped.views.at(-1).pending);
 assert.equal(stoppedPosts,1);assert(!stopped.values.size&&!stopped.reports.length);stopped.control.dispose();

 const inFlight=defer(),started=defer();let inFlightPosts=0;
 const late=make({fetcher:async(url,options)=>{
  if(options.method==='GET')return Response.json(success);
  inFlightPosts++;started.resolve();await inFlight.promise;return Response.json(queued,{status:202});
 }});
 const flight=late.control.submit(problem,code);await started.promise;
 assert(!late.control.stopWaiting(),'A stale stop click cannot abandon an in-flight POST');
 assert(late.views.at(-1).pending&&late.values.has(pendingKey(owner)));
 inFlight.resolve();await flight;await tick();assert.equal(late.reports.length,1);assert.equal(inFlightPosts,1);late.control.dispose();

 for(const response of ['lost','malformed']){
  let postCount=0,getCount=0;const readStarted=defer(),readRelease=defer();
  const uncertain=make({fetcher:async(url,options)=>{
   if(options.method==='GET'){getCount++;readStarted.resolve();await readRelease.promise;return Response.json(success);}
   postCount++;if(response==='lost')throw new TypeError('reply lost');return new Response('broken',{status:202});
  }});
  await uncertain.control.submit(problem,code);await readStarted.promise;
  assert(uncertain.views.at(-1).pending&&uncertain.views.at(-1).reconnecting);
  assert(!uncertain.views.at(-1).reloadRequired,'An uncertain POST must not be called an unsent module-load failure');
  assert(uncertain.values.has(pendingKey(owner)),'Keep the receipt until an uncertain POST has been reconciled');
  await uncertain.control.submit(problem,'do not replay an uncertain submission');assert.equal(postCount,1);
  readRelease.resolve();await tick();assert.equal(postCount,1);assert.equal(getCount,1);assert.equal(uncertain.reports.length,1);uncertain.control.dispose();
 }
 let restores=0;
 const recovered=make({fetcher:async(url,options)=>{assert.equal(options.method,'GET');restores++;return Response.json(success);}});
 recovered.values.set(pendingKey(owner),JSON.stringify({id,problem,code:'never execute stored code',report:{passed:true}}));
 recovered.control.resume();await recovered.control.submit(problem,'do not create another execution');await tick();
 assert.equal(restores,1);assert.equal(recovered.reports.length,1);assert(!recovered.values.size);recovered.control.dispose();

 const ownerReady=defer();let alive=true,ownerPosts=0;
 const changed=make({alive:()=>alive,fetcher:async()=>{ownerPosts++;return full();},wait:async(ms,signal)=>{ownerReady.resolve();await new Promise((_,reject)=>signal.addEventListener('abort',()=>reject(signal.reason),{once:true}));}});
 const old=changed.control.submit(problem,code);await ownerReady.promise;alive=false;changed.control.dispose();
 const viewCount=changed.views.length;await old;await tick();assert.equal(ownerPosts,1);assert.equal(changed.views.length,viewCount);assert(!changed.reports.length);
 const pollReady=defer(),cancelled=defer();let cancelling=false,cancelPosts=0,cancelReads=0;
 const cancellation=make({fetcher:async(url,options)=>{
  if(options.method==='POST'){
   assert.equal(url,'/api/judge/jobs/'+id+'/cancel');assert.deepEqual(JSON.parse(options.body),{expected_user_id:'alice'});
   cancelling=true;cancelPosts++;cancelled.resolve();return Response.json({...queued,status:'cancelled',ahead:0});
  }
  cancelReads++;return Response.json(cancelling?{...queued,status:'cancelled',ahead:0}:queued);
 },pollWait:async()=>{pollReady.resolve();await cancelled.promise;}});
 cancellation.values.set(pendingKey(owner),JSON.stringify({id,problem}));cancellation.control.resume();await pollReady.promise;
 await cancellation.control.cancel();await tick();assert.equal(cancelPosts,1);assert.equal(cancelReads,2);
 assert.equal(cancellation.views.at(-1).job.status,'cancelled');assert(!cancellation.reports.length);cancellation.control.dispose();
 console.log('PASS deferred queued controller: failed lazy imports stay unsent with reload recovery; stable snapshots, duplicate-click gate, cancellation races, refresh GET-only recovery, lost/malformed acknowledgements and owner isolation');
}finally{await vite.close();}

// Exercise the actual hook when its first deferred controller import fails.
// A stored receipt changes recovery: the earlier request may already be running.
const {JSDOM}=await import('jsdom');
const React=await import('react');
const {act}=React;
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://queue-hook-test.invalid/'});
const bindings={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,
 HTMLElement:dom.window.HTMLElement,sessionStorage:dom.window.sessionStorage,IS_REACT_ACT_ENVIRONMENT:true};
const originals=new Map([...Object.keys(bindings),'fetch'].map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(bindings))Object.defineProperty(globalThis,key,{configurable:true,writable:true,value});
let hookRequests=0;
globalThis.fetch=async()=>{hookRequests++;throw Error('A missing controller must not contact the server');};
const hookServer=await createServer({configFile:false,logLevel:'silent',server:{middlewareMode:true},appType:'custom',
 optimizeDeps:{noDiscovery:true,entries:[]},plugins:[{
  name:'queued-hook-missing-controller-fixture',
  load(id){
   if(id.endsWith('/src/social.tsx'))return 'export function useAccount(){return {health:{mode:"hosted"},user:{id:"alice"}};}';
   if(id.endsWith('/src/queuedJudgeController.ts'))return 'throw new TypeError("Failed to fetch dynamically imported module");';
  },
 }]});
let hookRoot,hookView;
try{
 const {createRoot}=await import('react-dom/client');
 const {useQueuedJudge}=await hookServer.ssrLoadModule('/src/useQueuedJudge.ts');
 const {submissionModuleLoadError,pendingSubmissionModuleLoadError,jobErrorText}=await hookServer.ssrLoadModule('/src/judgeJobs.ts');
 function Probe({owner}){hookView=useQueuedJudge(owner,true,()=>assert.fail('An import failure cannot produce a report'));return null;}
 async function render(owner){await act(async()=>{
  hookRoot??=createRoot(document.querySelector('#root'));
  hookRoot.render(React.createElement(Probe,{owner}));
 });}
 async function settle(predicate){
  for(let n=0;n<100&&!predicate();n++)await act(async()=>new Promise(resolve=>setTimeout(resolve,5)));
  assert(predicate(),'The hook must surface its failed controller import');
 }
 await render('user:alice');
 await act(async()=>hookView.submit('rl/dpo','# user source'));
 assert.equal(hookView.reloadRequired,true,JSON.stringify(hookView));assert.equal(hookView.error,submissionModuleLoadError);
 assert.equal(sessionStorage.getItem('paperleet.judge.pending.v1:user:alice'),null);
 assert.equal(hookRequests,0);assert.equal(hookView.active,false);
 assert.match(jobErrorText(hookView.error,'zh'),/本次尚未发送/);

 const receipt=JSON.stringify({id:'c'.repeat(32),problem:'rl/dpo'});
 sessionStorage.setItem('paperleet.judge.pending.v1:user:bob',receipt);
 await render('user:bob');await settle(()=>hookView.reloadRequired);
 assert.equal(hookView.error,pendingSubmissionModuleLoadError);
 assert.equal(sessionStorage.getItem('paperleet.judge.pending.v1:user:bob'),receipt,'Existing admission receipt survives the failed recovery import');
 assert.equal(hookRequests,0);assert.match(jobErrorText(hookView.error,'zh'),/已有任务/);
 assert(!jobErrorText(hookView.error,'zh').includes('尚未发送'),'An existing submission cannot be described as unsent');
 await act(async()=>hookView.submit('rl/dpo','# do not replay while the controller cannot recover'));
 assert.equal(hookView.error,pendingSubmissionModuleLoadError);assert.equal(hookRequests,0);
 assert.equal(sessionStorage.getItem('paperleet.judge.pending.v1:user:bob'),receipt);
 console.log('PASS actual queued hook: missing initial component is unsent; existing receipts retain reconnect wording and are never replayed');
}finally{
 if(hookRoot)await act(async()=>hookRoot.unmount());
 await hookServer.close();dom.window.close();
 for(const [key,original] of originals){if(original)Object.defineProperty(globalThis,key,original);else delete globalThis[key];}
}
