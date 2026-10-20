import {fixture} from './gpu-fixture.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {servePort,LIMITS} from './gpu-service.mjs';
import {MatrixRpc} from './rpc-client.mjs';
const rows=[];
function request({retain=true,read=true}={}){return {id:1,op:'program',inputs:[{rows:2,cols:2,data:new Float32Array([1,2,3,4]).buffer},{rows:2,cols:2,data:new Float32Array([1,0,0,1]).buffer}],nodes:[{a:0,b:1,transposeA:false,transposeB:false}],outputs:[{ref:2,read,retain}]};}
async function check(name,test,options){await fixture(test,options);rows.push({name,passed:true});}

function primitive(operation,handle){return operation==='upload'?{id:1,op:'upload',rows:2,cols:2,data:new Float32Array([1,2,3,4]).buffer}:operation==='matmul'?{id:1,op:'matmul',a:handle,b:handle,transposeA:false,transposeB:false}:{id:1,op:'read',handle};}
for(const [operation,points] of [
 ['upload',['allocate:1','push','write','fence','validation','pop']],
 ['matmul',['allocate:1','allocate:2','push','write','bind','encode','pass','dispatch','finish','submit','fence','validation','pop']],
 ['read',['allocate:1','push','encode','copy','finish','submit','map','range','unmap','validation','pop']]]){
 for(const point of points)await check(`${operation} rollback ${point}`,async({engine,existing,controls})=>{
  const before=engine.stats();controls.point=point;await assert.rejects(engine.execute(primitive(operation,existing.handle)));assert(controls.triggered);
  assert.equal(engine.stats().total_bytes,before.total_bytes);assert.equal(engine.stats().resident_handles,1);assert.equal(controls.live.size,1);assert.equal(engine.stats().closed,false);
  const fence=controls.trace.indexOf('fence'),destroy=controls.trace.indexOf('destroy');
  if(operation!=='read'&&controls.trace.includes('write')||operation==='read'&&['submit','map'].includes(point))assert(fence>=0&&destroy>fence,controls.trace.join(','));
  if(operation==='read'&&['range','unmap','validation','pop'].includes(point))assert(controls.trace.indexOf('map')<destroy);
 });
}
for(const [operation,point] of [['upload','write'],['matmul','bind'],['read','map']]){
 await check(`${operation} failed drain invalidates every resident`,async({engine,existing,controls})=>{
  controls.point=point;controls.failDrain=true;await assert.rejects(engine.execute(primitive(operation,existing.handle)));assert(engine.stats().closed);assert.equal(engine.stats().resident_handles,0);assert.equal(engine.stats().total_bytes,0);assert.equal(controls.live.size,0);
 });
 await check(`${operation} waits for outstanding work before reclaiming capacity`,async({engine,existing,controls})=>{
  const held={};held.promise=new Promise((resolve,reject)=>{held.resolve=resolve;held.reject=reject;});controls.held=held;controls.point=point;
  const operationPromise=engine.execute(primitive(operation,existing.handle));const rejected=assert.rejects(operationPromise);
  for(let i=0;i<100&&!controls.trace.includes('fence');i++)await new Promise(resolve=>setTimeout(resolve,1));assert(controls.trace.includes('fence'));
  assert(!controls.trace.includes('destroy'));assert(engine.stats().total_bytes>16);controls.held=null;held.resolve();await rejected;assert.equal(engine.stats().total_bytes,16);assert.equal(engine.stats().resident_handles,1);
 });
 await check(`${operation} successful primitive preserves ownership`,async({engine,existing})=>{
  const value=await engine.execute(primitive(operation,existing.handle));if(operation==='read')assert.equal(value.data.byteLength,16);else{assert.equal(engine.stats().resident_handles,2);engine.discard(value);}assert.equal(engine.stats().resident_handles,1);assert.equal(engine.stats().total_bytes,16);
 });
}
const source_sha256=Object.fromEntries(['gpu-service.mjs','plan-program.mjs','gpu-fixture.mjs','check-primitive.mjs'].map(name=>[name,createHash('sha256').update(fs.readFileSync(new URL('./'+name,import.meta.url))).digest('hex')]));
const report={status:'passed',cases:rows.length,rows,source_sha256,scope:'Injected host lifecycle checks only. No real GPU execution, numerical, performance, or physical allocation claim.'};
if(process.env.GPU_PRIMITIVE_RESULT)fs.writeFileSync(process.env.GPU_PRIMITIVE_RESULT,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({status:report.status,cases:report.cases}));
