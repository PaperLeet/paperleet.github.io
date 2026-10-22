import {fixture} from './gpu-fixture.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {servePort,LIMITS} from './gpu-service.mjs';
import {MatrixRpc} from './rpc-client.mjs';
const rows=[];
function request({retain=true,read=true}={}){return {id:1,op:'program',inputs:[{rows:2,cols:2,data:new Float32Array([1,2,3,4]).buffer},{rows:2,cols:2,data:new Float32Array([1,0,0,1]).buffer}],nodes:[{a:0,b:1,transposeA:false,transposeB:false}],outputs:[{ref:2,read,retain}]};}
async function check(name,test,options){await fixture(test,options);rows.push({name,passed:true});}
for(const point of ['allocate:1','allocate:2','allocate:3','allocate:4','allocate:5','write','encode','bind','pass','dispatch','copy','finish','submit','map','range','unmap','validation','pop']){
  await check('rollback on '+point,async({engine,controls})=>{
    controls.point=point;const before=engine.stats();await assert.rejects(engine.execute(request()));
    assert(controls.triggered);assert.equal(engine.stats().resident_handles,1);assert.equal(engine.stats().total_bytes,before.total_bytes);assert.equal(controls.live.size,1);
    if(controls.trace.includes('write')&&!['validation','pop','range','unmap'].includes(point)){
      const fence=controls.trace.indexOf('fence'),destroy=controls.trace.indexOf('destroy');assert(fence>=0&&destroy>fence,'Queued writes must drain before destruction: '+point+' '+controls.trace.join(','));
    }else if(['validation','pop','range','unmap'].includes(point)){
      // A resolved mapAsync already fences the submitted writes. Later host
      // conversion or scope errors do not need a second queue fence.
      assert(controls.trace.indexOf('map')<controls.trace.indexOf('destroy'));
    }
  });
}
await check('failed drain invalidates device and all handles',async({engine,controls})=>{
  controls.point='encode';controls.failDrain=true;await assert.rejects(engine.execute(request()));
  assert.equal(engine.stats().closed,true);assert.equal(engine.stats().total_bytes,0);assert.equal(engine.stats().resident_handles,0);
});
await check('exact aggregate peak includes uploads uniforms and readback',async({engine,controls})=>{
  const value=await engine.execute(request());assert.equal(engine.stats().peak_bytes,112);assert.equal(engine.stats().total_bytes,32);assert.equal(value.outputs[0].data.byteLength,16);
  engine.discard(value);assert.equal(engine.stats().total_bytes,16);assert.equal(controls.live.size,1);
},{changes:{bytes:112}});
await check('aggregate byte rejection allocates nothing',async({engine,controls})=>{
  await assert.rejects(engine.execute(request()),error=>error.code==='Capacity');assert.equal(controls.allocation,0);assert.equal(controls.trace.length,0);assert.equal(engine.stats().total_bytes,16);
},{changes:{bytes:111}});
await check('aggregate MAC rejection includes every node',async({engine,controls})=>{
  const value=request();value.nodes.push({a:2,b:1,transposeA:false,transposeB:false},{a:3,b:1,transposeA:false,transposeB:false});value.outputs[0].ref=4;
  await assert.rejects(engine.execute(value),error=>error.code==='Capacity');assert.equal(controls.allocation,0);
},{changes:{macs:16}});
await check('ephemeral program works at full published handle capacity',async({engine})=>{
  await assert.rejects(engine.execute(request()),error=>error.code==='Capacity');
  const result=await engine.execute(request({retain:false}));assert.equal(result.outputs[0].data.byteLength,16);assert.equal(engine.stats().total_bytes,16);assert.equal(engine.stats().resident_handles,1);
},{changes:{handles:1}});
const invalid=[
 ['future reference',r=>r.nodes[0].a=2,'Reference'],['input as output',r=>r.outputs[0].ref=0,'Reference'],
 ['duplicate output',r=>r.outputs.push({...r.outputs[0]}),'Protocol'],['no observable output',r=>r.outputs[0]={ref:2,read:false,retain:false},'Protocol'],
 ['nonboolean transpose',r=>r.nodes[0].transposeA=1,'Protocol'],['inner shape mismatch',r=>r.inputs[1]={rows:3,cols:1,data:new ArrayBuffer(12)},'Shape'],
 ['byte mismatch',r=>r.inputs[0].data=new ArrayBuffer(12),'Shape'],['untyped upload',r=>r.inputs[0].data=new Float32Array(4),'Dtype'],
 ['unknown shader field',r=>r.shader='untrusted','Protocol'],['too many inputs',r=>r.inputs=Array(9).fill(r.inputs[0]),'Protocol'],
 ['too many nodes',r=>r.nodes=Array(9).fill(r.nodes[0]),'Protocol'],['empty nodes',r=>r.nodes=[],'Protocol'],
 ['too many outputs',r=>r.outputs=Array(9).fill(r.outputs[0]),'Protocol'],['unknown resident handle',r=>r.inputs[0]={handle:999},'Handle'],
 ['merged field spelling',r=>r.nodes[0]={'a,b':0,transposeA:false,transposeB:false},'Protocol'],
];
for(const [name,mutate,code] of invalid)await check('preflight '+name,async({engine,controls})=>{
  const value=request();mutate(value);await assert.rejects(engine.execute(value),error=>error.code===code);assert.equal(controls.allocation,0);assert.equal(controls.trace.length,0);
});
await check('release_many validates whole list before mutation',async({engine,existing})=>{
  const value=await engine.execute(request({read:false})),handle=value.outputs[0].handle;
  await assert.rejects(engine.execute({id:2,op:'release_many',handles:[existing.handle,999]}),error=>error.code==='Handle');assert.equal(engine.stats().resident_handles,2);
  await assert.rejects(engine.execute({id:3,op:'release_many',handles:[handle,handle]}),error=>error.code==='Protocol');assert.equal(engine.stats().resident_handles,2);
  assert.equal((await engine.execute({id:4,op:'release_many',handles:[existing.handle,handle]})).released,2);assert.equal(engine.stats().total_bytes,0);
});
await check('cancelled real-engine result discards every retained output',async({engine,controls})=>{
  const channel=new MessageChannel();const service=servePort(channel.port1,engine);const client=new MatrixRpc(channel.port2);await client.ready;
  const held={};held.promise=new Promise((resolve,reject)=>{held.resolve=resolve;held.reject=reject;});controls.held=held;
  const req=request({read:false});req.nodes.push({a:2,b:1,transposeA:false,transposeB:false});req.outputs.push({ref:3,read:false,retain:true});
  const {id:_,op:__,...fields}=req,ticket=client.send('program',fields);const rejected=assert.rejects(ticket.promise,error=>error.code==='Cancelled');
  for(let i=0;i<100&&!controls.trace.includes('fence');i++)await new Promise(resolve=>setTimeout(resolve,2));assert(controls.trace.includes('fence'));
  assert.equal((await client.cancel(ticket.id)).cancelled,true);controls.held=null;held.resolve();await rejected;
  assert.equal(engine.stats().resident_handles,1);assert.equal(engine.stats().total_bytes,16);await service.close('done');client.fail(new Error('done'));
});

await check('fresh uploaded input retention shares the same atomic program',async({engine})=>{
 const value=request({retain:false});value.outputs.push({ref:0,read:false,retain:true});
 const result=await engine.execute(value);assert.equal(result.outputs[0].data.byteLength,16);assert(result.outputs[1].handle);assert.equal(result.outputs[1].data,undefined);
 assert.equal(engine.stats().resident_handles,2);assert.equal(engine.stats().total_bytes,32);assert.equal(engine.stats().peak_bytes,112);
 engine.discard(result);assert.equal(engine.stats().resident_handles,1);assert.equal(engine.stats().total_bytes,16);
});
await check('existing input handle cannot be exported as a second owner',async({engine,existing,controls})=>{
 const value=request({retain:false});value.inputs[0]={handle:existing.handle};value.outputs.push({ref:0,read:false,retain:true});
 await assert.rejects(engine.execute(value),error=>error.code==='Reference');assert.equal(controls.allocation,0);
 assert.equal(engine.stats().resident_handles,1);
});
await check('fresh input retention consumes handle budget before allocation',async({engine,controls})=>{
 const value=request();value.outputs.push({ref:0,read:false,retain:true});
 await assert.rejects(engine.execute(value),error=>error.code==='Capacity');assert.equal(controls.allocation,0);
},{changes:{handles:2}});
await check('duplicate retained upload is rejected before allocation',async({engine,controls})=>{
 const value=request();value.outputs.push({ref:0,read:false,retain:true},{ref:0,read:false,retain:true});
 await assert.rejects(engine.execute(value),error=>error.code==='Protocol');assert.equal(controls.allocation,0);
});
for(const point of ['allocate:1','allocate:2','allocate:3','allocate:4','allocate:5','write','encode','bind','pass','dispatch','copy','finish','submit','map','range','unmap','validation','pop']){
 await check('retained upload transaction rollback on '+point,async({engine,controls})=>{
  const value=request();value.outputs.push({ref:0,read:false,retain:true},{ref:1,read:false,retain:true});
  controls.point=point;const before=engine.stats();await assert.rejects(engine.execute(value));assert(controls.triggered);
  assert.equal(engine.stats().resident_handles,1);assert.equal(engine.stats().total_bytes,before.total_bytes);assert.equal(controls.live.size,1);
 });
}
await check('cancellation discards uploaded inputs as well as computed results',async({engine,controls})=>{
 const channel=new MessageChannel();const service=servePort(channel.port1,engine);const client=new MatrixRpc(channel.port2);await client.ready;
 const held={};held.promise=new Promise((resolve,reject)=>{held.resolve=resolve;held.reject=reject;});controls.held=held;
 const value=request({read:false});value.outputs.push({ref:0,read:false,retain:true},{ref:1,read:false,retain:true});
 const {id:_,op:__,...fields}=value;const ticket=client.send('program',fields);const rejected=assert.rejects(ticket.promise,error=>error.code==='Cancelled');
 for(let i=0;i<100&&!controls.trace.includes('fence');i++)await new Promise(resolve=>setTimeout(resolve,2));assert(controls.trace.includes('fence'));
 assert.equal((await client.cancel(ticket.id)).cancelled,true);controls.held=null;held.resolve();await rejected;
 assert.equal(engine.stats().resident_handles,1);assert.equal(engine.stats().total_bytes,16);await service.close('done');client.fail(new Error('done'));
});
const source_sha256=Object.fromEntries(['gpu-service.mjs','plan-program.mjs','rpc-client.mjs','check-program.mjs','gpu-fixture.mjs'].map(name=>[name,createHash('sha256').update(fs.readFileSync(new URL('./'+name,import.meta.url))).digest('hex')]));
const report={status:'passed',cases:rows.length,rows,source_sha256,scope:'Actual planner/engine and RPC code with an injected WebGPU surface, to validate preflight/allocation/fence/error/retained-handle lifecycle. No shader execution or numerical GPU claim.'};
if(process.env.GPU_PROGRAM_RESULT)fs.writeFileSync(process.env.GPU_PROGRAM_RESULT,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({status:report.status,cases:report.cases}));
