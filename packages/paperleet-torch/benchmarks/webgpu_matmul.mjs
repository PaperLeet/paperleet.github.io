// Experimental backend feasibility probe. This is not a torch device/backend.
// Import in a secure-context browser and call benchmarkMatmul(). Upload/readback
// timings are a lower bound: Python/JS conversion and autograd are not included.
const shader = `
struct Shape { m:u32, k:u32, n:u32, padding:u32 }
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read_write> c:array<f32>;
@group(0) @binding(3) var<uniform> shape:Shape;
var<workgroup> tileA:array<f32,256>;
var<workgroup> tileB:array<f32,256>;
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_id) local:vec3<u32>) {
  let row=group.y*16u+local.y;
  let col=group.x*16u+local.x;
  let slot=local.y*16u+local.x;
  var total=0.0;
  for(var block=0u; block<(shape.k+15u)/16u; block++) {
    let ka=block*16u+local.x;
    let kb=block*16u+local.y;
    tileA[slot]=0.0; tileB[slot]=0.0;
    if(row<shape.m && ka<shape.k) { tileA[slot]=a[row*shape.k+ka]; }
    if(kb<shape.k && col<shape.n) { tileB[slot]=b[kb*shape.n+col]; }
    workgroupBarrier();
    for(var k=0u; k<16u; k++) { total+=tileA[local.y*16u+k]*tileB[k*16u+local.x]; }
    workgroupBarrier();
  }
  if(row<shape.m && col<shape.n) { c[row*shape.n+col]=total; }
}`;

function median(values) {
  const sorted=[...values].sort((a,b)=>a-b);
  const middle=Math.floor(sorted.length/2);
  return sorted.length%2 ? sorted[middle] : (sorted[middle-1]+sorted[middle])/2;
}

export async function benchmarkMatmul({shapes=[[32,32,32],[64,128,128],[128,128,128],
  [256,256,256],[512,512,512],[37,53,29]],repeat=11,residentIterations=10}={}) {
  if (!Number.isInteger(repeat) || repeat<1 || !Number.isInteger(residentIterations) || residentIterations<1 ||
      !shapes.length || shapes.some(shape=>shape.length!==3 || shape.some(n=>!Number.isInteger(n) || n<1))) {
    throw new Error('Positive shapes/repeat/residentIterations required');
  }
  if (!navigator.gpu) return {available:false,reason:'navigator.gpu unavailable'};
  const adapter=await navigator.gpu.requestAdapter();
  if (!adapter) return {available:false,reason:'No GPU adapter'};
  const device=await adapter.requestDevice();
  const info=adapter.info;
  const report={available:true,adapter:{vendor:info.vendor,architecture:info.architecture,
    device:info.device,description:info.description},repeat,residentIterations,results:[]};
  try {
    device.pushErrorScope('validation');
    const start=performance.now();
    const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{
      module:device.createShaderModule({code:shader}),entryPoint:'main'}});
    report.compile_ms=performance.now()-start;
    for (const [m,k,n] of shapes) {
      const resources=[];
      const buffer=(size,usage)=>{const result=device.createBuffer({size,usage});resources.push(result);return result;};
      try {
        const a=Float32Array.from({length:m*k},(_,i)=>(i%17-8)/9);
        const b=Float32Array.from({length:k*n},(_,i)=>(i%13-6)/7);
        const left=buffer(a.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
        const right=buffer(b.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
        const output=buffer(m*n*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
        const staging=buffer(m*n*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
        const uniform=buffer(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
        device.queue.writeBuffer(uniform,0,new Uint32Array([m,k,n,0]));
        const bindings=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:
          [left,right,output,uniform].map((resource,binding)=>({binding,resource:{buffer:resource}}))});
        const submit=(count,readback)=>{
          const encoder=device.createCommandEncoder();
          for(let i=0;i<count;i++) {
            const pass=encoder.beginComputePass();
            pass.setPipeline(pipeline);pass.setBindGroup(0,bindings);
            pass.dispatchWorkgroups(Math.ceil(n/16),Math.ceil(m/16));pass.end();
          }
          if(readback)encoder.copyBufferToBuffer(output,0,staging,0,m*n*4);
          device.queue.submit([encoder.finish()]);
        };
        const roundtrip=async()=>{
          device.queue.writeBuffer(left,0,a);device.queue.writeBuffer(right,0,b);
          submit(1,true);await staging.mapAsync(GPUMapMode.READ);
          try { return new Float32Array(staging.getMappedRange()).slice(); }
          finally { staging.unmap(); }
        };
        const actual=await roundtrip();
        let maxError=0;
        for(let row=0;row<m;row++)for(let col=0;col<n;col++) {
          let expected=0;
          for(let j=0;j<k;j++)expected+=a[row*k+j]*b[j*n+col];
          const error=Math.abs(actual[row*n+col]-expected);
          if(!Number.isFinite(error) || error>1e-4+Math.abs(expected)*2e-4)throw new Error(`Incorrect matmul at ${row},${col}: ${error}`);
          maxError=Math.max(maxError,error);
        }
        for(let i=0;i<3;i++)await roundtrip();
        const transfers=[],resident=[];
        for(let i=0;i<repeat;i++) {
          const begin=performance.now();await roundtrip();transfers.push(performance.now()-begin);
          const second=performance.now();submit(residentIterations,false);await device.queue.onSubmittedWorkDone();
          resident.push((performance.now()-second)/residentIterations);
        }
        report.results.push({shape:[m,k,n],max_abs_error:maxError,
          roundtrip_ms:median(transfers),roundtrip_samples_ms:transfers,
          resident_ms:median(resident),resident_samples_ms:resident});
      } finally { for(const resource of resources)resource.destroy(); }
    }
    const validation=await device.popErrorScope();
    if(validation)throw new Error(validation.message);
    return report;
  } finally { device.destroy(); }
}
