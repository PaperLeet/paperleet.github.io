import {fixture} from './gpu-fixture.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {servePort,LIMITS} from './gpu-service.mjs';
import {MatrixRpc} from './rpc-client.mjs';
const rows=[];
function request({retain=true,read=true}={}){return {id:1,op:'program',inputs:[{rows:2,cols:2,data:new Float32Array([1,2,3,4]).buffer},{rows:2,cols:2,data:new Float32Array([1,0,0,1]).buffer}],nodes:[{a:0,b:1,transposeA:false,transposeB:false}],outputs:[{ref:2,read,retain}]};}
async function check(name,test,options){await fixture(test,options);rows.push({name,passed:true});}


for(const [operation,point] of [['upload','write'],['matmul','bind'],['read','map'],['program','bind']]){
 await fixture(async({engine,existing,controls})=>{
  const channel=new MessageChannel(),service=servePort(channel.port1,engine),client=new MatrixRpc(channel.port2);await client.ready;
  const held={};held.promise=new Promise((resolve,reject)=>{held.resolve=resolve;held.reject=reject;});controls.held=held;controls.point=point;
  const requestValue=operation==='program'?request():operation==='upload'?{op:operation,rows:2,cols:2,data:new ArrayBuffer(16)}:operation==='matmul'?{op:operation,a:existing.handle,b:existing.handle,transposeA:false,transposeB:false}:{op:operation,handle:existing.handle};
  const {id:_,op,...fields}=requestValue,ticket=client.send(op,fields);const outcome=ticket.promise.then(()=>({status:'resolved'}),error=>({status:'rejected',code:error.code}));
  try{
   for(let i=0;i<100&&!controls.trace.includes('fence');i++)await new Promise(resolve=>setTimeout(resolve,1));assert(controls.trace.includes('fence'));
   const cancellation=await client.cancel(ticket.id);assert.equal(cancellation.cancelled,true);assert(!controls.trace.includes('destroy'));
   const queuedStats=client.stats();controls.held=null;held.resolve();const observed=await outcome,stats=await queuedStats;assert.equal(stats.total_bytes,16);assert.equal(stats.resident_handles,1);
   assert.equal(observed.code,'Cancelled');rows.push({operation,point,cancellation,outcome:observed,stats,trace:controls.trace.slice(),passed:observed.code==='Cancelled'});
  }finally{controls.held=null;held.resolve();await service.close('done');client.fail(Object.assign(new Error('done'),{code:'Closed'}));}
 });
}
const source_sha256=Object.fromEntries(['gpu-service.mjs','rpc-client.mjs','gpu-fixture.mjs','check-cancel-drain.mjs'].map(name=>[name,createHash('sha256').update(fs.readFileSync(new URL('./'+name,import.meta.url))).digest('hex')]));
const report={status:'passed',cases:rows.length,rows,source_sha256,scope:'Injected failure while queue drain is held; acknowledged cancellation and subsequent request ownership.'};
if(process.env.GPU_CANCEL_DRAIN_RESULT)fs.writeFileSync(process.env.GPU_CANCEL_DRAIN_RESULT,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({status:report.status,cases:report.cases}));
