import assert from 'node:assert/strict';
import {servePort,LIMITS} from './gpu-service.mjs';
import {MatrixRpc} from './rpc-client.mjs';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
const rows=[];
async function fixture(test,{limitChanges={},timeoutMs=1000}={}){
  const channel=new MessageChannel(),gates=[],handles=new Set(),calls=[];
  let lostResolve,closed=false,discarded=0,disposedResolve;
  const disposed=new Promise(resolve=>disposedResolve=resolve);
  const engine={info:{fake:true},lost:new Promise(resolve=>lostResolve=resolve),
    async execute(request){calls.push(request.op);if(request.op==='stats')return engine.stats();
      return new Promise((resolve,reject)=>gates.push({resolve:()=>{const handle=gates.length;handles.add(handle);resolve({handle});},reject}));},
    discard(value){handles.delete(value.handle);discarded++;},
    close(){closed=true;handles.clear();for(const gate of gates)gate.reject(new Error('device destroyed'));},
    stats(){return {closed,resident_handles:handles.size,total_bytes:handles.size*4,discarded};}};
  const service=servePort(channel.port1,engine,{limits:{...LIMITS,...limitChanges},onClosed:disposedResolve});
  const client=new MatrixRpc(channel.port2,{timeoutMs});await client.ready;
  try{await test({client,engine,gates,calls,service,disposed,lose:()=>lostResolve({reason:'unknown'})});}
  finally{await service.close('fixture completed');client.fail(new Error('fixture completed'));}
  assert.equal(engine.stats().total_bytes,0);assert.equal(client.pending.size,0);
}
const tick=()=>new Promise(resolve=>setTimeout(resolve,5));
async function waitFor(test){for(let i=0;i<100&&!test();i++)await tick();assert(test());}
async function check(name,test,options){await fixture(test,options);rows.push({name,passed:true});}
await check('queued cancellation never reaches engine',async({client,gates,calls})=>{
  const a=client.send('hold'),b=client.send('hold');const rejected=assert.rejects(b.promise,error=>error.code==='Cancelled');
  await waitFor(()=>gates.length===1);assert.equal((await client.cancel(b.id)).cancelled,true);
  gates[0].resolve();await a.promise;await rejected;assert.deepEqual(calls,['hold']);
});
await check('in-flight cancellation discards completed output',async({client,gates,engine})=>{
  const a=client.send('hold'),rejected=assert.rejects(a.promise,error=>error.code==='Cancelled');
  await waitFor(()=>gates.length===1);assert.equal((await client.cancel(a.id)).cancelled,true);
  gates[0].resolve();await rejected;assert.equal(engine.stats().resident_handles,0);assert.equal(engine.stats().discarded,1);
  assert.equal((await client.cancel(a.id)).cancelled,false);
});
await check('pending bound remains live after rejection',async({client,gates,engine})=>{
  const a=client.send('hold'),b=client.send('hold');const aError=assert.rejects(a.promise,error=>error.code==='Closed'),bError=assert.rejects(b.promise,error=>error.code==='Closed');
  await assert.rejects(client.send('hold').promise,error=>error.code==='Capacity');await client.close();await Promise.all([aError,bError]);assert(engine.stats().closed);
},{limitChanges:{pending:2}});
await check('host close destroys active and queued work',async({client,gates,service,disposed})=>{
  const a=client.send('hold'),b=client.send('hold');const failures=[assert.rejects(a.promise,error=>error.code==='Closed'),assert.rejects(b.promise,error=>error.code==='Closed')];
  await waitFor(()=>gates.length===1);await service.close('host hard stop');await Promise.all(failures);const value=await disposed;assert.equal(value.stats.total_bytes,0);
});
await check('device loss invalidates the session',async({client,gates,lose,disposed})=>{
  const a=client.send('hold'),failure=assert.rejects(a.promise,error=>error.code==='Closed');await waitFor(()=>gates.length===1);lose();await failure;await disposed;await waitFor(()=>client.closed);await assert.rejects(client.stats(),error=>error.code==='Closed');
});
await check('request ID reuse closes instead of ambiguous response',async({client,gates,disposed})=>{
  const a=client.send('hold'),failure=assert.rejects(a.promise,error=>error.code==='Closed');await waitFor(()=>gates.length===1);client.port.postMessage({id:a.id,op:'hold'});await failure;assert((await disposed).reason.includes('increase'));
});
await check('request budget terminates session',async({client,disposed})=>{
  await client.stats();await client.stats();await assert.rejects(client.stats(),error=>error.code==='Capacity');await disposed;
},{limitChanges:{requests:2}});
await check('abandoned port lifetime bounded by host lease',async({client,disposed})=>{
  const value=await disposed;assert.equal(value.reason,'lease expired');assert.equal(value.stats.total_bytes,0);await waitFor(()=>client.closed);await assert.rejects(client.stats(),error=>error.code==='Closed');
},{limitChanges:{leaseMs:25}});
await check('client timeout abandons and closes owned session',async({client,gates,disposed})=>{
  const a=client.send('hold');await assert.rejects(a.promise,error=>error.code==='Timeout');const result=await disposed;assert.equal(result.stats.total_bytes,0);assert.equal(client.pending.size,0);
},{timeoutMs:25});
const source_sha256=Object.fromEntries(['gpu-service.mjs','rpc-client.mjs','check-protocol.mjs'].map(name=>[name,createHash('sha256').update(fs.readFileSync(new URL('./'+name,import.meta.url))).digest('hex')]));
const report={status:'passed',cases:rows.length,rows,source_sha256,scope:'Deterministic service/client lifecycle and protocol tests with an injected fake engine; not real GPU execution.'};
if(process.env.GPU_PROTOCOL_RESULT)fs.writeFileSync(process.env.GPU_PROTOCOL_RESULT,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
