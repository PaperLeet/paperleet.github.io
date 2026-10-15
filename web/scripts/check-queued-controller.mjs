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
  let postCount=0,getCount=0;
  const uncertain=make({fetcher:async(url,options)=>{
   if(options.method==='GET'){getCount++;return Response.json(success);}
   postCount++;if(response==='lost')throw new TypeError('reply lost');return new Response('broken',{status:202});
  }});
  await uncertain.control.submit(problem,code);await tick();assert.equal(postCount,1);assert.equal(getCount,1);assert.equal(uncertain.reports.length,1);uncertain.control.dispose();
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
 console.log('PASS deferred queued controller: stable snapshots, duplicate-click gate, cancellation races, refresh GET-only recovery, lost/malformed acknowledgements and owner isolation');
}finally{await vite.close();}
