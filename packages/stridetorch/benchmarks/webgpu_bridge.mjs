// Developer capability probe, separate from the supported torch device policy.
// Call inside the execution context being assessed with an initialized Pyodide.
// A successful scalar roundtrip does not qualify a tensor/autograd backend.
let probeSequence = 0;
export async function probeWebGpuBridge(py) {
  if (typeof py?.runPython !== 'function' || typeof py?.runPythonAsync !== 'function')
    throw new TypeError('An initialized Pyodide instance is required');
  const result = {
    secure_context:globalThis.isSecureContext ?? false,
    origin:globalThis.origin ?? null,
    cross_origin_isolated:globalThis.crossOriginIsolated ?? false,
    pyodide:py.version,
    jspi:{suspending:typeof WebAssembly.Suspending, promising:typeof WebAssembly.promising},
    gpu_exposed:!!globalThis.navigator?.gpu,
    entry_tests:[],
  };
  let pending, gpuValue;
  do {
    const suffix = Date.now() + '_' + ++probeSequence;
    pending = '_stride_bridge_pending_' + suffix;
    gpuValue = '_stride_bridge_gpu_' + suffix;
  } while (py.globals.has(pending) || py.globals.has(gpuValue));
  const awaitValue = name => "__import__('pyodide.ffi',fromlist=['run_sync']).run_sync(" + name + "())";
  py.globals.set(pending, () => new Promise(resolve => setTimeout(() => resolve(37),10)));
  try {
    for (const entry of ['runPython','runPythonAsync']) {
      try {
        const value = await py[entry](awaitValue(pending));
        result.entry_tests.push({entry,passed:value===37,value});
      } catch (error) {
        result.entry_tests.push({entry,passed:false,error:String(error).slice(-3000)});
      }
    }
    if (!result.gpu_exposed) return result;
    const adapter = await navigator.gpu.requestAdapter();
    result.adapter_available = !!adapter;
    if (!adapter) return result;
    const info = adapter.info;
    result.adapter_info = {vendor:info?.vendor,architecture:info?.architecture,
      device:info?.device,description:info?.description,fallback:info?.isFallbackAdapter};
    const device = await adapter.requestDevice();
    const resources = [];
    try {
      device.pushErrorScope('validation');
      const buffer = device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      resources.push(buffer);
      const read = device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      resources.push(read);
      const module = device.createShaderModule({code:
        '@group(0) @binding(0) var<storage,read_write> a:array<f32>; @compute @workgroup_size(1) fn main(){a[0]=6.0*7.0;}'});
      const pipeline = device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main'}});
      const group = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),
        entries:[{binding:0,resource:{buffer}}]});
      py.globals.set(gpuValue, async () => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0,group);
        pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(buffer,0,read,0,4);
        device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        try { return new Float32Array(read.getMappedRange())[0]; }
        finally { read.unmap(); }
      });
      const value = await py.runPythonAsync(awaitValue(gpuValue));
      const validation = await device.popErrorScope();
      result.gpu_python_roundtrip = {passed:value===42 && !validation,value,
        ...(validation ? {error:validation.message} : {})};
    } finally {
      for (const resource of resources) resource.destroy();
      device.destroy();
    }
  } catch (error) {
    result.error = String(error.stack || error);
  } finally {
    if (py.globals.has(pending)) py.globals.delete(pending);
    if (py.globals.has(gpuValue)) py.globals.delete(gpuValue);
  }
  return result;
}
