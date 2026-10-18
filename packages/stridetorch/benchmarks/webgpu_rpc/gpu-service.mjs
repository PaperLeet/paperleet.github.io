// Experimental fixed-operation service. This is not a supported torch device.
import {planProgram} from './plan-program.mjs';
export const LIMITS = Object.freeze({bytes:32*1024*1024,matrixBytes:4*1024*1024,
  dimension:1024,macs:256*1024*1024,handles:24,pending:8,requests:4096,leaseMs:120000,
  programInputs:8,programNodes:8,programOutputs:8});
const shader = `
struct D {m:u32,n:u32,k:u32,ac:u32,bc:u32,ta:u32,tb:u32,pad:u32}
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read_write> c:array<f32>;
@group(0) @binding(3) var<uniform> d:D;
var<workgroup> at:array<f32,64>;
var<workgroup> bt:array<f32,64>;
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>) {
 let row=id.y; let col=id.x; let slot=local.y*8u+local.x;
 var sum:f32=0.0;
 for(var base:u32=0u;base<d.k;base=base+8u){
  let ak=base+local.x;let bk=base+local.y;
  var ai=row*d.ac+ak;if(d.ta==1u){ai=ak*d.ac+row;}
  var bi=bk*d.bc+col;if(d.tb==1u){bi=col*d.bc+bk;}
  at[slot]=0.0;bt[slot]=0.0;
  if(row<d.m && ak<d.k){at[slot]=a[ai];}
  if(col<d.n && bk<d.k){bt[slot]=b[bi];}
  workgroupBarrier();
  for(var x:u32=0u;x<8u;x=x+1u){sum=sum+at[local.y*8u+x]*bt[x*8u+local.x];}
  workgroupBarrier();
 }
 if(row<d.m && col<d.n){c[row*d.n+col]=sum;}
}`;
function fault(code,message){return Object.assign(new Error(message),{code});}
function requireValue(ok,code,message){if(!ok)throw fault(code,message);}
function record(value){return value!==null&&typeof value==='object'&&!Array.isArray(value)&&Object.getPrototypeOf(value)===Object.prototype;}
function keys(value,expected){requireValue(record(value)&&Object.keys(value).length===expected.length&&expected.every(key=>Object.hasOwn(value,key)),'Protocol','Unexpected fields');}
function integer(value,min,max){return Number.isSafeInteger(value)&&value>=min&&value<=max;}

export async function createGpuEngine(limits=LIMITS){
  requireValue(!!navigator.gpu,'Unavailable','WebGPU unavailable in service context');
  const adapter=await navigator.gpu.requestAdapter();
  requireValue(!!adapter,'Unavailable','No WebGPU adapter');
  const device=await adapter.requestDevice();
  const allocations=new Set(),handles=new Map();
  let nextHandle=1,total=0,peak=0,closed=false,submitted=0,completed=0;
  const info=adapter.info;
  let pipeline;
  try {
    device.pushErrorScope('validation');
    pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:device.createShaderModule({code:shader}),entryPoint:'main'}});
    const error=await device.popErrorScope();if(error)throw fault('GPUValidation',error.message);
  } catch(error){device.destroy();throw error;}
  function alloc(bytes,usage){
    requireValue(!closed,'Closed','Service closed');
    requireValue(integer(bytes,4,limits.bytes)&&total+bytes<=limits.bytes,'Capacity','GPU allocation budget exceeded');
    const gpu=device.createBuffer({size:bytes,usage});
    const item={gpu,bytes,alive:true};allocations.add(item);total+=bytes;peak=Math.max(peak,total);return item;
  }
  function drop(item){if(item?.alive){item.alive=false;item.gpu.destroy();allocations.delete(item);total-=item.bytes;}}
  function matrix(rows,cols){
    requireValue(integer(rows,1,limits.dimension)&&integer(cols,1,limits.dimension),'Shape','Positive bounded 2D shapes required');
    requireValue(rows*cols*4<=limits.matrixBytes,'Capacity','Matrix byte limit exceeded');
    requireValue(handles.size<limits.handles,'Capacity','Resident handle limit exceeded');
    return {rows,cols,item:alloc(rows*cols*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST)};
  }
  function find(id){requireValue(integer(id,1,0x7fffffff)&&handles.has(id),'Handle','Unknown or released handle');return handles.get(id);}
  function publish(value){requireValue(!closed,'Closed','Service closed');const handle=nextHandle++;handles.set(handle,value);return {handle,rows:value.rows,cols:value.cols};}
  async function complete(){submitted++;await device.queue.onSubmittedWorkDone();completed++;requireValue(!closed,'Closed','Service closed');}
  async function scoped(work){
    device.pushErrorScope('validation');
    let value,failed;
    try{value=await work();}catch(error){failed=error;}
    const error=await device.popErrorScope();
    if(failed)throw failed;if(error)throw fault('GPUValidation',error.message);return value;
  }
  const engine={
    info:{vendor:info?.vendor,architecture:info?.architecture,fallback:info?.isFallbackAdapter,secure_context:isSecureContext},
    async execute(request){
      requireValue(!closed,'Closed','Service closed');
      if(request.op==='program'){
        const plan=planProgram(request,{limits,find,residentCount:handles.size,totalBytes:total});
        const created=[],kept=new Set(),published=[];const values=[];
        let workIssued=false,enqueued=false,finished=false;
        const allocate=(bytes,usage)=>{const item=alloc(bytes,usage);created.push(item);return item;};
        try{
          const result=await scoped(async()=>{
            for(const input of plan.inputs){
              if(input.existing){values.push(input.existing);continue;}
              const item=allocate(input.bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
              workIssued=true;device.queue.writeBuffer(item.gpu,0,input.data);values.push({rows:input.rows,cols:input.cols,item});
            }
            const encoder=device.createCommandEncoder();
            for(const node of plan.nodes){
              const a=values[node.a],b=values[node.b];
              const item=allocate(node.bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
              const uniform=allocate(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
              workIssued=true;device.queue.writeBuffer(uniform.gpu,0,new Uint32Array([node.rows,node.cols,node.k,a.cols,b.cols,+node.transposeA,+node.transposeB,0]));
              const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[a.item,b.item,item,uniform].map((resource,binding)=>({binding,resource:{buffer:resource.gpu}}))});
              const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);
              pass.dispatchWorkgroups(Math.ceil(node.cols/8),Math.ceil(node.rows/8));pass.end();values.push({rows:node.rows,cols:node.cols,item});
            }
            const stage=plan.readBytes?allocate(plan.readBytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST):null;
            let offset=0;
            for(const output of plan.outputs)if(output.read){encoder.copyBufferToBuffer(values[output.ref].item.gpu,0,stage.gpu,offset,output.bytes);offset+=output.bytes;}
            device.queue.submit([encoder.finish()]);enqueued=true;submitted++;
            if(!stage){await device.queue.onSubmittedWorkDone();completed++;finished=true;requireValue(!closed,'Closed','Service closed');return {outputs:plan.outputs.map(output=>({rows:output.rows,cols:output.cols}))};}
            await stage.gpu.mapAsync(GPUMapMode.READ);completed++;finished=true;requireValue(!closed,'Closed','Service closed');
            const mapped=stage.gpu.getMappedRange();offset=0;
            const outputs=plan.outputs.map(output=>{
              const value={rows:output.rows,cols:output.cols};
              if(output.read){value.data=mapped.slice(offset,offset+output.bytes);offset+=output.bytes;}
              return value;
            });stage.gpu.unmap();return {outputs};
          });
          for(let i=0;i<plan.outputs.length;i++)if(plan.outputs[i].retain){
            const value=values[plan.outputs[i].ref],handle=publish(value).handle;published.push(handle);kept.add(value.item);result.outputs[i].handle=handle;
          }
          return result;
        }catch(error){
          if(workIssued&&!finished&&!closed){
            try{await device.queue.onSubmittedWorkDone();if(enqueued)completed++;}
            catch{engine.close();}
          }
          for(const handle of published)handles.delete(handle);kept.clear();throw error;
        }
        finally{for(const item of created)if(!kept.has(item))drop(item);}
      }
      if(request.op==='upload'){
        keys(request,['id','op','rows','cols','data']);
        requireValue(request.data instanceof ArrayBuffer,'Dtype','A transferred float32 ArrayBuffer is required');
        requireValue(integer(request.rows,1,limits.dimension)&&integer(request.cols,1,limits.dimension),'Shape','Positive bounded 2D shapes required');
        requireValue(request.data.byteLength===request.rows*request.cols*4,'Shape','Buffer byte length does not match shape');
        const value=matrix(request.rows,request.cols);
        try{await scoped(async()=>{device.queue.writeBuffer(value.item.gpu,0,request.data);await complete();});return publish(value);}
        catch(error){drop(value.item);throw error;}
      }
      if(request.op==='matmul'){
        keys(request,['id','op','a','b','transposeA','transposeB']);
        requireValue(typeof request.transposeA==='boolean'&&typeof request.transposeB==='boolean','Protocol','Transpose flags must be booleans');
        const a=find(request.a),b=find(request.b),ta=request.transposeA,tb=request.transposeB;
        const m=ta?a.cols:a.rows,k=ta?a.rows:a.cols,n=tb?b.rows:b.cols;
        requireValue(k===(tb?b.cols:b.rows),'Shape','Matrix inner dimensions differ');
        requireValue(m*n*k<=limits.macs,'Capacity','Matrix multiplication work limit exceeded');
        const value=matrix(m,n);let uniform;
        try{
          uniform=alloc(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
          await scoped(async()=>{
            device.queue.writeBuffer(uniform.gpu,0,new Uint32Array([m,n,k,a.cols,b.cols,+ta,+tb,0]));
            const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[a.item,b.item,value.item,uniform].map((item,binding)=>({binding,resource:{buffer:item.gpu}}))});
            const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
            pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(n/8),Math.ceil(m/8));pass.end();
            device.queue.submit([encoder.finish()]);await complete();
          });
          return publish(value);
        }catch(error){drop(value.item);throw error;}finally{drop(uniform);}
      }
      if(request.op==='read'){
        keys(request,['id','op','handle']);const value=find(request.handle);
        const stage=alloc(value.item.bytes,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
        try{return await scoped(async()=>{
          const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(value.item.gpu,0,stage.gpu,0,value.item.bytes);
          device.queue.submit([encoder.finish()]);submitted++;
          await stage.gpu.mapAsync(GPUMapMode.READ);completed++;
          requireValue(!closed,'Closed','Service closed');
          const data=stage.gpu.getMappedRange().slice(0);stage.gpu.unmap();
          return {rows:value.rows,cols:value.cols,data};
        });}finally{drop(stage);}
      }
      if(request.op==='release'){
        keys(request,['id','op','handle']);const value=find(request.handle);handles.delete(request.handle);drop(value.item);return {released:true};
      }
      if(request.op==='release_many'){
        keys(request,['id','op','handles']);
        requireValue(Array.isArray(request.handles)&&integer(request.handles.length,1,limits.handles)&&new Set(request.handles).size===request.handles.length,'Protocol','A bounded unique handle list is required');
        const values=request.handles.map(find);
        for(let i=0;i<values.length;i++){handles.delete(request.handles[i]);drop(values[i].item);}
        return {released:values.length};
      }
      if(request.op==='stats'){keys(request,['id','op']);return engine.stats();}
      throw fault('Operation','Unsupported operation');
    },
    discard(result){for(const output of result?.outputs||[result])if(output?.handle&&handles.has(output.handle)){const value=handles.get(output.handle);handles.delete(output.handle);drop(value.item);}},
    stats(){return {resident_handles:handles.size,resident_bytes:[...handles.values()].reduce((n,value)=>n+value.item.bytes,0),total_bytes:total,allocations:allocations.size,peak_bytes:peak,submitted,completed,closed};},
    close(){if(!closed){closed=true;handles.clear();for(const item of [...allocations])drop(item);device.destroy();}},
    lost:device.lost,
  };
  return engine;
}

// Exported engine injection is for deterministic protocol tests; clients cannot
// select an engine, shader, limit or lease through their MessagePort.
export function servePort(port,engine,{limits=LIMITS,onClosed=()=>{}}={}){
  let lastId=0,requests=0,closing=false,closePromise,pumpPromise;
  const tickets=new Map(),queue=[];
  const send=(value,transfer=[])=>{try{port.postMessage(value,transfer);}catch{void close('transport');}};
  function answer(id,ok,result){send(ok?{id,ok:true,result}:{id,ok:false,error:{code:result.code||'GPUError',message:String(result.message||result).slice(0,512)}});}
  async function pump(){
    while(queue.length&&!closing){
      const ticket=queue.shift();
      try{
        if(ticket.cancelled)throw fault('Cancelled','Request cancelled before execution');
        const result=await engine.execute(ticket.request);
        if(ticket.cancelled){engine.discard(result);throw fault('Cancelled','Completed result discarded after cancellation');}
        const transfer=[result?.data,...(Array.isArray(result?.outputs)?result.outputs.map(output=>output.data):[])].filter(value=>value instanceof ArrayBuffer);
        if(!closing)send({id:ticket.request.id,ok:true,result},[...new Set(transfer)]);
      }catch(error){if(!closing)answer(ticket.request.id,false,error);}
      finally{tickets.delete(ticket.request.id);ticket.request=null;}
    }
  }
  async function close(reason){
    if(closing)return closePromise;
    closing=true;clearTimeout(lease);engine.close();
    for(const [id] of tickets)answer(id,false,fault('Closed',reason));
    queue.length=0;tickets.clear();
    closePromise=(async()=>{
      await pumpPromise;const stats=engine.stats();send({event:'closed',reason,stats});port.close();onClosed({reason,stats});return stats;
    })();return closePromise;
  }
  const lease=setTimeout(()=>void close('lease expired'),limits.leaseMs);
  port.onmessage=event=>{
    if(closing)return;
    const request=event.data;
    try{
      requireValue(record(request)&&integer(request.id,1,0x7fffffff),'Protocol','A positive bounded request ID is required');
      if(request.id<=lastId){void close('Request IDs must increase');return;}lastId=request.id;
      if(++requests>limits.requests){answer(request.id,false,fault('Capacity','Session request limit exceeded'));void close('request limit');return;}
      if(request.op==='close'){keys(request,['id','op']);answer(request.id,true,{closing:true});void close('client closed');return;}
      if(request.op==='cancel'){
        keys(request,['id','op','target']);requireValue(integer(request.target,1,0x7fffffff),'Protocol','Invalid cancellation target');
        const ticket=tickets.get(request.target);if(ticket)ticket.cancelled=true;answer(request.id,true,{cancelled:!!ticket});return;
      }
      requireValue(tickets.size<limits.pending,'Capacity','Pending request limit exceeded');
      const ticket={request,cancelled:false};tickets.set(request.id,ticket);queue.push(ticket);
      if(!pumpPromise)pumpPromise=pump().finally(()=>{pumpPromise=null;});
    }catch(error){answer(integer(request?.id,1,0x7fffffff)?request.id:0,false,error);}
  };
  port.onmessageerror=()=>void close('message error');port.start();
  engine.lost.then(value=>{if(!closing)void close('GPU device lost: '+value.reason);});
  send({event:'ready',info:engine.info,limits});
  return {close,stats:()=>({...engine.stats(),pending:tickets.size})};
}

// One trusted host owns one Worker/device/session and its hard-stop lifecycle.
if(typeof self!=='undefined'&&typeof self.postMessage==='function'&&typeof document==='undefined'){
  let attached=false,service,disposing=false;
  self.onmessage=async event=>{
    if(event.data?.type==='dispose'){disposing=true;if(service)await service.close('host disposed');return;}
    if(event.data?.type!=='connect'||attached||event.ports.length!==1)return;
    attached=true;
    try{
      const engine=await createGpuEngine();
      if(disposing){engine.close();event.ports[0].postMessage({event:'closed',reason:'host disposed during startup'});event.ports[0].close();self.postMessage({type:'disposed',reason:'host disposed during startup',stats:engine.stats()});return;}
      service=servePort(event.ports[0],engine,{onClosed:value=>self.postMessage({type:'disposed',...value})});
      self.postMessage({type:'ready',info:engine.info});
    }catch(error){event.ports[0].postMessage({event:'unavailable',error:String(error)});event.ports[0].close();self.postMessage({type:'unavailable',error:String(error)});}
  };
}
