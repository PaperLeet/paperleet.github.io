import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {MatrixRpc} from './rpc-client.mjs';
import {MatrixRpc as LegacyClient} from './fixtures/protocol-v0/rpc-client.mjs';
import {servePort, createGpuEngine, LIMITS} from './gpu-service.mjs';
import {servePort as legacyServe, createGpuEngine as legacyEngine} from './fixtures/protocol-v0/gpu-service.mjs';
import {fakeGpu} from './gpu-fixture.mjs';
import {PROTOCOL} from './rpc-protocol.mjs';

const rows=[];
const tick=()=>new Promise(resolve=>setTimeout(resolve,2));
async function until(test){for(let i=0;i<200&&!test();i++)await tick();assert(test());}
function outgoing(port, transform){
  return {postMessage(value, transfer){port.postMessage(transform(value), transfer);},
    set onmessage(value){port.onmessage=value;}, set onmessageerror(value){port.onmessageerror=value;},
    start(){port.start();}, close(){port.close();}};
}
async function fixture(test,{oldHost=false,oldClient=false,ready=message=>message,timeoutMs=1000}={}){
  const fake=fakeGpu(),names=['navigator','GPUBufferUsage','GPUMapMode','isSecureContext'];
  const previous=Object.fromEntries(names.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const globals={navigator:{gpu:fake.gpu},GPUBufferUsage:{STORAGE:1,COPY_SRC:2,COPY_DST:4,UNIFORM:8,MAP_READ:16},GPUMapMode:{READ:1},isSecureContext:true};
  for(const name of names)Object.defineProperty(globalThis,name,{value:globals[name],configurable:true});
  let engine,service,client;let disposed;const calls=[];const channel=new MessageChannel();
  try{
    engine=await (oldHost?legacyEngine:createGpuEngine)();const execute=engine.execute;
    engine.execute=async request=>{calls.push(request.op);return execute(request);};
    const serverPort=outgoing(channel.port1,value=>value.event==='ready'?ready(value):value);
    service=(oldHost?legacyServe:servePort)(serverPort,engine,{onClosed:value=>disposed=value});
    client=new (oldClient?LegacyClient:MatrixRpc)(channel.port2,{timeoutMs});
    await test({client,engine,calls,channel,disposed:()=>disposed});
  }finally{
    await service?.close('negotiation fixture completed');client?.fail(new Error('fixture completed'));
    channel.port1.close();channel.port2.close();assert.equal(fake.controls.live.size,0);
    for(const name of names)previous[name]?Object.defineProperty(globalThis,name,previous[name]):delete globalThis[name];
  }
}
async function check(name,test,options){try{await fixture(test,options);}catch(error){console.error('FAILED',name);throw error;}rows.push({name,passed:true});console.log('PASS',name);}
const inputs=()=>[{rows:2,cols:2,data:new Float32Array([1,2,3,4])},{rows:2,cols:2,data:new Float32Array([1,0,0,1])}];
const nodes=[{a:0,b:1,transposeA:false,transposeB:false}];
const outputs=[{ref:2,read:true,retain:false}];

await check('new client and host advertise immutable capabilities without an RPC',async({client,calls})=>{
  await client.ready;assert.equal(client.nextId,1);assert.deepEqual(calls,[]);
  assert.equal(client.protocol.major,1);assert.equal(client.protocol.legacy,false);
  assert.equal(client.supports('program-retain-inputs'),true);assert.equal(client.supports('unknown'),false);
  assert.throws(()=>client.protocol.features.push('unknown'),TypeError);
  assert.throws(()=>client.protocol={major:2},TypeError);
  await client.program(inputs(),nodes,outputs);assert.deepEqual(calls,['program']);
});
await check('new client and legacy host retain computed-output programs',async({client,calls,engine})=>{
  await client.ready;assert.equal(client.protocol.major,0);assert.equal(client.protocol.legacy,true);
  client.requireFeatures('program-matmul-f32');assert.equal(client.supports('program-retain-inputs'),false);
  const result=await client.program(inputs(),nodes,[{ref:2,read:true,retain:true}]);
  assert(result.outputs[0].handle);assert.equal(result.outputs[0].data.length,4);
  await client.releaseMany([result.outputs[0].handle]);assert.equal(engine.stats().total_bytes,0);
  assert.deepEqual(calls,['program','release_many']);
},{oldHost:true});
await check('legacy client and new host retain their original contract',async({client,engine})=>{
  await client.ready;
  const result=await client.program(inputs(),nodes,[{ref:2,read:false,retain:true}]);
  const array=await client.read(result.outputs[0].handle);assert.equal(array.length,4);
  await client.release(result.outputs[0].handle);assert.equal(engine.stats().total_bytes,0);
},{oldClient:true});
await check('legacy refusal precedes conversion upload and request ID consumption',async({client,calls,engine})=>{
  await client.ready;let converted=0;const data={toJs(){converted++;throw Error('Must not copy');}};
  await assert.rejects(client.program([{rows:2,cols:2,data},...inputs().slice(1)],nodes,[...outputs,{ref:0,read:false,retain:true}]),error=>error.code==='UnsupportedFeature'&&error.message.includes('program-retain-inputs'));
  assert.equal(converted,0);assert.equal(client.nextId,1);assert.deepEqual(calls,[]);assert.equal(engine.stats().allocations,0);
  await client.program(inputs(),nodes,outputs);assert.deepEqual(calls,['program']);
},{oldHost:true});
await check('new retained input ownership and batch release still work',async({client,engine})=>{
  await client.ready;const data=inputs();const original=data[0].data.slice();
  const result=await client.program(data,nodes,[...outputs,{ref:0,read:false,retain:true}]);
  assert(result.outputs[1].handle);assert.deepEqual(data[0].data,original);
  assert.equal(engine.stats().resident_handles,1);await client.releaseMany([result.outputs[1].handle]);assert.equal(engine.stats().total_bytes,0);
});
await check('program waits for ready before copying or submitting',async({client,calls})=>{
  let converted=0;const data=inputs();data[0].data={toJs(){converted++;return new Float32Array(4);}};
  const pending=client.program(data,nodes,outputs);assert.equal(converted,0);assert.equal(client.nextId,1);
  assert.equal(client.supports('program-matmul-f32'),false);assert.throws(()=>client.requireFeatures('program-matmul-f32'),error=>error.code==='NotReady');
  await pending;assert.equal(converted,1);assert.deepEqual(calls,['program']);
});
await check('later ready messages cannot upgrade a legacy connection',async({client,channel})=>{
  await client.ready;channel.port1.postMessage({event:'ready',protocol:PROTOCOL});await tick();
  assert.equal(client.supports('program-retain-inputs'),false);assert.equal(client.protocol.legacy,true);
},{oldHost:true});
await check('future minor and unknown feature preserve the known intersection',async({client})=>{
  await client.ready;assert.equal(client.protocol.minor,5);assert.equal(client.supports('program-retain-inputs'),true);assert.equal(client.supports('future-op'),false);
},{ready:value=>({...value,protocol:{...PROTOCOL,minor:5,features:[...PROTOCOL.features,'future-op']}})});
await check('explicit feature subset cannot silently enable input retention',async({client,calls})=>{
  await client.ready;assert.equal(client.protocol.legacy,false);assert.equal(client.supports('program-retain-inputs'),false);
  await assert.rejects(client.program(inputs(),nodes,[...outputs,{ref:0,read:false,retain:true}]),error=>error.code==='UnsupportedFeature');assert.deepEqual(calls,[]);
  await client.program(inputs(),nodes,outputs);
},{ready:value=>({...value,protocol:{...PROTOCOL,features:['program-matmul-f32']}})});
await check('empty feature set rejects programs while leaving diagnostics usable',async({client,calls})=>{
  await client.ready;await assert.rejects(client.program(inputs(),nodes,outputs),error=>error.code==='UnsupportedFeature');await client.stats();assert.deepEqual(calls,['stats']);
},{ready:value=>({...value,protocol:{...PROTOCOL,features:[]}})});
await check('closed client does not advertise usable features',async({client})=>{
  await client.ready;await client.close();assert.equal(client.supports('program-matmul-f32'),false);
  assert.throws(()=>client.requireFeatures('program-matmul-f32'),error=>error.code==='Closed');
});
const invalid=[
  ['null',null],['array',[]],['wrong name',{...PROTOCOL,name:'other'}],
  ['future major',{...PROTOCOL,major:2}],['negative major',{...PROTOCOL,major:-1}],['string major',{...PROTOCOL,major:'1'}],
  ['missing minor',{name:PROTOCOL.name,major:1,features:[]}],['fractional minor',{...PROTOCOL,minor:.5}],['negative minor',{...PROTOCOL,minor:-1}],
  ['string minor',{...PROTOCOL,minor:'0'}],['unsafe minor',{...PROTOCOL,minor:Number.MAX_SAFE_INTEGER+1}],
  ['nonarray features',{...PROTOCOL,features:{}}],['too many features',{...PROTOCOL,features:Array.from({length:33},(_,i)=>'feature-'+i)}],
  ['duplicate feature',{...PROTOCOL,features:['program-matmul-f32','program-matmul-f32']}],
  ['object feature',{...PROTOCOL,features:[{}]}],['empty feature name',{...PROTOCOL,features:['']}],
  ['overlong feature',{...PROTOCOL,features:['x'.repeat(65)]}],['bad feature spelling',{...PROTOCOL,features:['operation\n']}],
  ['retention without program',{...PROTOCOL,features:['program-retain-inputs']}],
];
for(const [label,protocol] of invalid)await check('reject '+label+' and dispose host without math',async({client,calls,engine,disposed})=>{
  await assert.rejects(client.ready,error=>error.code==='IncompatibleProtocol');
  await until(()=>disposed());assert.equal(engine.stats().closed,true);assert.equal(engine.stats().total_bytes,0);assert.deepEqual(calls,[]);assert.equal(client.pending.size,0);
},{ready:value=>({...value,protocol})});
await check('primitive upload also waits for compatible startup before copying',async({client,calls,disposed})=>{
  let converted=0;const data={toJs(){converted++;return new Float32Array(4);}};
  const startup=assert.rejects(client.ready,error=>error.code==='IncompatibleProtocol');
  const operation=assert.rejects(client.upload(data,2,2),error=>error.code==='IncompatibleProtocol');
  assert.equal(converted,0);assert.equal(client.nextId,1);await Promise.all([startup,operation]);
  await until(()=>disposed());assert.deepEqual(calls,[]);
},{ready:value=>({...value,protocol:{...PROTOCOL,major:2}})});
for(const [label,call] of [['matmul',c=>c.matmul(1,2)],['read',c=>c.read(1)],['release',c=>c.release(1)],['releaseMany',c=>c.releaseMany([1])],['stats',c=>c.stats()],['cancel',c=>c.cancel(1)]])await check(label+' waits for compatible startup',async({client,calls,disposed})=>{
  const startup=assert.rejects(client.ready,error=>error.code==='IncompatibleProtocol');
  const operation=assert.rejects(call(client),error=>error.code==='IncompatibleProtocol');
  assert.equal(client.nextId,1);await Promise.all([startup,operation]);await until(()=>disposed());assert.deepEqual(calls,[]);
},{ready:value=>({...value,protocol:{...PROTOCOL,major:2}})});
await check('startup timeout actively disposes host without waiting for its lease',async({client,calls,disposed})=>{
  await assert.rejects(client.ready,error=>error.code==='Timeout');await until(()=>disposed());assert.deepEqual(calls,[]);assert.equal(client.protocol,null);
},{ready:()=>({event:'ignored-ready'}),timeoutMs:25});
await check('closing during startup preempts negotiation without GPU work',async({client,calls,disposed})=>{
  const startup=assert.rejects(client.ready,error=>error.code==='Closed');await client.close();await startup;await until(()=>disposed());assert.deepEqual(calls,[]);
});
const source_sha256=Object.fromEntries(['gpu-service.mjs','rpc-client.mjs','rpc-protocol.mjs','check-negotiation.mjs','fixtures/protocol-v0/gpu-service.mjs','fixtures/protocol-v0/rpc-client.mjs','fixtures/protocol-v0/plan-program.mjs'].map(name=>[name,createHash('sha256').update(fs.readFileSync(new URL('./'+name,import.meta.url))).digest('hex')]));
console.log(JSON.stringify({status:'passed',cases:rows.length,rows,source_sha256,scope:'Real MessageChannel, historical pre-retention client/service/planner and injected GPU; no real GPU numerical or performance claim'}));
