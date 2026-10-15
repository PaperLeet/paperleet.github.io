import assert from 'node:assert/strict';
import fs from 'node:fs';
import React from 'react';
import {renderToString} from 'react-dom/server';
import {createServer} from 'vite';
const server=await createServer({server:{middlewareMode:true},appType:'custom'});
const storage=globalThis.localStorage;
try {
  const {admitSubmission,admissionDelay,ADMISSION_WINDOW,ADMISSION_TIMEOUT}=await server.ssrLoadModule('/src/queueAdmission.ts');
  const {JobRequestError,requestJob,watchJob}=await server.ssrLoadModule('/src/judgeJobs.ts');
  const id='a'.repeat(32),client='b'.repeat(32),problem='architecture/attention';
  const submission={id,problem,code:'# frozen original\n'};
  const queued={id,problem,status:'queued',ahead:2};
  const full=(header='5')=>Response.json({error:'Queue is full',code:'queue_full'},{status:429,headers:{'Retry-After':header}});
  let now=1_700_000_000_000,posts=[],delays=[],waits=[],attempts=0;
  const mutable={...submission};
  const result=await admitSubmission(mutable,client,new AbortController().signal,'alice',()=>{attempts++;},state=>{
    waits.push(state);mutable.code='# later editor edit';mutable.problem='rl/dpo';mutable.id='c'.repeat(32);
  },{now:()=>now,wall:()=>now,fetcher:async(url,options)=>{
    assert.equal(url,'/api/judge/jobs');assert.equal(options.method,'POST');
    assert.equal(options.headers['X-PaperLeet-Client-ID'],client);
    posts.push(JSON.parse(options.body));
    return posts.length===1?full():posts.length===2?full('30'):Response.json(queued,{status:202});
  },wait:async(ms,signal)=>{signal.throwIfAborted();delays.push(ms);now+=ms;}});
  assert.deepEqual(result,{...queued,report:undefined,error:undefined});assert.equal(attempts,3);
  assert.deepEqual(delays,[10_000,30_000]);
  assert(posts.every(body=>JSON.stringify(body)===JSON.stringify({...submission,expected_user_id:'alice'})));
  assert(waits.every(state=>state.source===submission.code));
  assert.equal(admissionDelay('Thu, 01 Jan 1970 00:01:00 GMT',0),60_000);
  for(const value of [undefined,'garbage','-1','0','5','1.5'])assert.equal(admissionDelay(value,now),10_000);

  let mono=0,wall=Date.now(),calls=0;
  await assert.rejects(admitSubmission(submission,client,new AbortController().signal,undefined,()=>{},()=>{}, {
    now:()=>mono,wall:()=>wall,fetcher:async()=>{calls++;return full();},wait:async(ms)=>{mono+=ms;wall-=10_000_000;}
  }),error=>error instanceof JobRequestError&&error.code==='admission_timeout'&&error.message===ADMISSION_TIMEOUT);
  assert.equal(mono,ADMISSION_WINDOW);assert.equal(calls,12,'Wall-clock changes cannot renew the hard retry deadline');
  mono=0;calls=0;
  await assert.rejects(admitSubmission(submission,client,new AbortController().signal,undefined,()=>{},()=>{}, {
    now:()=>mono,wall:()=>0,fetcher:async()=>{calls++;return full('300');},wait:async(ms)=>{mono+=ms;}
  }),/Automatic waiting stopped/);
  assert.equal(calls,1,'Never retry before a long Retry-After; end the bounded wait instead');

  for(const [status,code] of [[429,undefined],[429,'rate_limited'],[400,'queue_full'],[401,'queue_full'],[409,'queue_full'],[503,'queue_full']]) {
    let count=0;
    await assert.rejects(admitSubmission(submission,client,new AbortController().signal,undefined,()=>{},()=>{}, {
      fetcher:async()=>{count++;return Response.json({error:'refused',...(code?{code}:{})},{status});},
      wait:()=>{throw Error('Only explicit queue_full 429 can wait');}
    }),error=>error instanceof JobRequestError&&error.status===status);
    assert.equal(count,1);
  }
  const aborted=new AbortController();let executed=0;
  aborted.abort(new Error('owner changed'));
  await assert.rejects(admitSubmission(submission,client,aborted.signal,undefined,()=>{},()=>{}, {fetcher:async()=>{executed++;return full();}}),/owner changed/);
  assert.equal(executed,0);
  const stop=new AbortController();executed=0;
  await assert.rejects(admitSubmission(submission,client,stop.signal,undefined,()=>{},()=>stop.abort(new Error('stopped before admission')), {
    fetcher:async()=>{executed++;return full();},wait:async(ms,signal)=>signal.throwIfAborted()
  }),/stopped before admission/);
  assert.equal(executed,1);

  for(const reply of [()=>{throw new TypeError('reply lost');},()=>new Response('<html>proxy error</html>',{status:502}),()=>new Response('broken',{status:202})]) {
    let count=0;
    await assert.rejects(admitSubmission(submission,client,new AbortController().signal,undefined,()=>{},()=>{}, {
      fetcher:async()=>{count++;return reply();},wait:()=>{throw Error('Uncertain acknowledgements never repeat POST');}
    }));
    assert.equal(count,1);
    let reads=0;
    await watchJob(id,client,new AbortController().signal,()=>{},()=>{},async(url,options)=>{
      reads++;assert.equal(options.method,'GET');assert.equal(url,'/api/judge/jobs/'+id);
      return Response.json({...queued,status:'cancelled',ahead:0});
    });
    assert.equal(reads,1);
  }
  await assert.rejects(requestJob('',client,new AbortController().signal,submission,async()=>Response.json(null,{status:429})),error=>error instanceof JobRequestError&&error.status===429&&!error.code);

  const {QueueAdmissionFallback}=await server.ssrLoadModule('/src/QueueAdmissionFallback.tsx');
  const {default:QueueAdmissionNotice}=await server.ssrLoadModule('/src/QueueAdmissionNotice.tsx');
  const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
  for(const en of [false,true]) {
    globalThis.localStorage={getItem:()=>en?'en':'zh'};
    for(const Component of [QueueAdmissionFallback,QueueAdmissionNotice]) {
      const html=renderToString(React.createElement(LocaleProvider,null,React.createElement(Component,{
        waiting:{retryAt:Date.now()+10_000,deadline:Date.now()+120_000,source:'<script>private captured code</script>'},changed:true,href:'#/1',onStop(){},onDownload(){}
      })));
      assert(html.includes(en?'Stop waiting':'停止等待')&&html.includes(en?'Download captured code':'下载本次提交代码'));
      assert(html.includes(en?'editor has new changes':'编辑器已有新修改'));
      assert(!html.includes('private captured code'),'Closed source review must not dump the submitted snapshot into the notice');
      assert(html.includes(en?'has not joined':'尚未入队')&&html.includes(en?'Refresh stops':'刷新会停止'));
    }
  }
  const confirming=renderToString(React.createElement(QueueAdmissionFallback,{waiting:{retryAt:0,deadline:0,source:'private',attempting:true},changed:false,onStop(){},onDownload(){}}));
  assert(confirming.includes('disabled=""')&&confirming.includes('正在确认能否入队')&&!confirming.includes('尚未入队'));
  const hook=fs.readFileSync('src/useQueuedJudge.ts','utf8');
  assert(hook.includes("import('./queuedJudgeController')"),'Capacity policy stays deferred');
  console.log('PASS queue admission: captured source/ID/owner, explicit refusal only, Retry-After, monotonic two-minute cap, stop/owner abort, uncertain GET-only recovery and bilingual fallback controls');
} finally {
  if(storage===undefined)delete globalThis.localStorage;else globalThis.localStorage=storage;
  await server.close();
}
