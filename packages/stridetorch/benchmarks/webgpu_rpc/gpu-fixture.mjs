import assert from 'node:assert/strict';
import {createGpuEngine,LIMITS} from './gpu-service.mjs';
export function fakeGpu(){
  const controls={point:null,allocation:0,trace:[],live:new Set(),held:null,failDrain:false,triggered:false};
  const hit=name=>{controls.trace.push(name);if(controls.point===name){controls.triggered=true;controls.point=null;throw new Error('Injected '+name);}};
  let lostResolve;const lost=new Promise(resolve=>lostResolve=resolve);
  const device={lost,
    pushErrorScope(){hit('push');},async popErrorScope(){hit('pop');if(controls.point==='validation'){controls.point=null;controls.triggered=true;return {message:'Injected validation'};}return null;},
    createShaderModule(){return {};},async createComputePipelineAsync(){return {getBindGroupLayout:()=>({})};},createBindGroup(){hit('bind');return {};},
    createBuffer({size}){hit('allocate:'+ ++controls.allocation);const buffer={memory:new ArrayBuffer(size),destroyed:false,
      destroy(){if(!buffer.destroyed){buffer.destroyed=true;controls.live.delete(buffer);controls.trace.push('destroy');}},
      async mapAsync(){hit('map');if(controls.held)await controls.held.promise;},
      getMappedRange(){hit('range');return buffer.memory;},unmap(){hit('unmap');}};controls.live.add(buffer);return buffer;},
    createCommandEncoder(){hit('encode');return {beginComputePass(){hit('pass');return {setPipeline(){},setBindGroup(){},dispatchWorkgroups(){hit('dispatch');},end(){}};},copyBufferToBuffer(){hit('copy');},finish(){hit('finish');return {};}};},
    queue:{writeBuffer(buffer,offset,data){hit('write');new Uint8Array(buffer.memory).set(new Uint8Array(data.buffer||data,data.byteOffset||0,data.byteLength),offset);},
      submit(){hit('submit');},async onSubmittedWorkDone(){hit('fence');if(controls.failDrain)throw new Error('Injected fence failure');if(controls.held)await controls.held.promise;}},
    destroy(){for(const buffer of [...controls.live])buffer.destroy();controls.held?.reject(new Error('Destroyed'));lostResolve({reason:'destroyed'});}};
  return {device,controls,gpu:{requestAdapter:async()=>({info:{vendor:'fake'},requestDevice:async()=>device})}};
}
export async function fixture(test,{changes={}}={}){
  const fake=fakeGpu(),names=['navigator','GPUBufferUsage','GPUMapMode','isSecureContext'];
  const old=Object.fromEntries(names.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const globals={navigator:{gpu:fake.gpu},GPUBufferUsage:{STORAGE:1,COPY_SRC:2,COPY_DST:4,UNIFORM:8,MAP_READ:16},GPUMapMode:{READ:1},isSecureContext:true};
  for(const name of names)Object.defineProperty(globalThis,name,{value:globals[name],configurable:true});
  let engine;
  try{
    engine=await createGpuEngine({...LIMITS,...changes});
    const existing=await engine.execute({id:0,op:'upload',rows:2,cols:2,data:new Float32Array([7,8,9,10]).buffer});
    fake.controls.trace=[];fake.controls.allocation=0;
    await test({engine,existing,...fake});
  }finally{engine?.close();assert.equal(fake.controls.live.size,0);for(const name of names)old[name]?Object.defineProperty(globalThis,name,old[name]):delete globalThis[name];}
}
