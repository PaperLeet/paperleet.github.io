// The sender retains its NumPy allocation; upload transfers a separate JS copy.
import {negotiateProtocol} from './rpc-protocol.mjs';
export class MatrixRpc {
  constructor(port,{timeoutMs=15000}={}){
    this.port=port;this.timeoutMs=timeoutMs;this.nextId=1;this.pending=new Map();this.closed=false;
    let protocol=null;
    Object.defineProperty(this,'protocol',{enumerable:true,get:()=>protocol});
    this.ready=new Promise((resolve,reject)=>{this.readyResolve=resolve;this.readyReject=reject;});
    this.readyTimer=setTimeout(()=>this.abandon(Object.assign(new Error('Service startup timeout'),{code:'Timeout'})),timeoutMs);
    port.onmessage=event=>{
      const value=event.data;
      if(value?.event==='ready'){
        if(this.closed||protocol!==null)return;
        try{protocol=negotiateProtocol(value);}
        catch(error){
          this.abandon(error);return;
        }
        clearTimeout(this.readyTimer);this.info=value.info;this.limits=value.limits;this.readyResolve(value);return;
      }
      if(value?.event==='closed'||value?.event==='unavailable'){this.fail(Object.assign(new Error(value.reason||value.error||'Closed'),{code:'Closed'}));return;}
      const ticket=this.pending.get(value?.id);if(!ticket)return;
      this.pending.delete(value.id);clearTimeout(ticket.timer);
      value.ok?ticket.resolve(value.result):ticket.reject(Object.assign(new Error(value.error?.message||'RPC failure'),{code:value.error?.code||'Protocol'}));
    };
    port.onmessageerror=()=>this.fail(Object.assign(new Error('Message transfer failed'),{code:'Transport'}));port.start();
  }
  fail(error){if(this.closed)return;this.closed=true;clearTimeout(this.readyTimer);this.readyReject(error);for(const value of this.pending.values()){clearTimeout(value.timer);value.reject(error);}this.pending.clear();this.port.close();}
  abandon(error){if(this.closed)return;try{this.port.postMessage({id:this.nextId++,op:'close'});}catch{}this.fail(error);}
  supports(feature){return !this.closed&&!!this.protocol?.features.includes(feature);}
  requireFeatures(...features){
    if(this.closed)throw Object.assign(new Error('Client closed'),{code:'Closed'});
    if(!this.protocol)throw Object.assign(new Error('Await MatrixRpc.ready before checking features'),{code:'NotReady'});
    for(const feature of features)if(!this.supports(feature))throw Object.assign(new Error('GPU service does not support '+feature+'; reload the client and service together'),{code:'UnsupportedFeature'});
  }
  send(op,fields={},transfer=[]){
    if(this.closed)return {id:0,promise:Promise.reject(Object.assign(new Error('Client closed'),{code:'Closed'}))};
    const id=this.nextId++;
    const promise=new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{
        this.abandon(Object.assign(new Error('RPC timeout; session abandoned'),{code:'Timeout'}));
      },this.timeoutMs);
      this.pending.set(id,{resolve,reject,timer});
      try{this.port.postMessage({id,op,...fields},transfer);}catch(error){clearTimeout(timer);this.pending.delete(id);reject(error);}
    });
    return {id,promise};
  }
  async upload(data,rows,cols){
    await this.ready;
    // PyProxy.toJs copies numeric ndarray buffers. Destroy any temporary proxy
    // in the caller when conversion does not return a JS TypedArray.
    if(typeof data?.toJs==='function')data=data.toJs();
    if(!(data instanceof Float32Array))throw new TypeError('Upload requires contiguous float32 data');
    const copy=new Float32Array(data);
    return this.send('upload',{rows,cols,data:copy.buffer},[copy.buffer]).promise;
  }
  async matmul(a,b,transposeA=false,transposeB=false){await this.ready;return this.send('matmul',{a,b,transposeA,transposeB}).promise;}
  async read(handle){await this.ready;const value=await this.send('read',{handle}).promise;return new Float32Array(value.data);}
  async release(handle){await this.ready;return this.send('release',{handle}).promise;}
  async releaseMany(handles){await this.ready;return this.send('release_many',{handles}).promise;}
  async program(inputs,nodes,outputs){
    await this.ready;
    this.requireFeatures('program-matmul-f32');
    if(outputs.some(output=>Number.isSafeInteger(output.ref)&&output.ref>=0&&output.ref<inputs.length&&output.retain===true))this.requireFeatures('program-retain-inputs');
    const transfer=[];
    const owned=inputs.map(input=>{
      if(Object.hasOwn(input,'handle'))return {handle:input.handle};
      let data=input.data;if(typeof data?.toJs==='function')data=data.toJs();
      if(!(data instanceof Float32Array))throw new TypeError('Program upload requires contiguous float32 data');
      const copy=new Float32Array(data);transfer.push(copy.buffer);return {rows:input.rows,cols:input.cols,data:copy.buffer};
    });
    const result=await this.send('program',{inputs:owned,nodes,outputs},transfer).promise;
    for(const output of result.outputs)if(output.data instanceof ArrayBuffer)output.data=new Float32Array(output.data);
    return result;
  }
  async stats(){await this.ready;return this.send('stats').promise;}
  async cancel(target){await this.ready;return this.send('cancel',{target}).promise;}
  async close(){if(this.closed)return;if(!this.protocol){this.abandon(Object.assign(new Error('Client closed during startup'),{code:'Closed'}));return;}try{await this.send('close').promise;}finally{this.fail(Object.assign(new Error('Client closed'),{code:'Closed'}));}}
}
