// The sender retains its NumPy allocation; upload transfers a separate JS copy.
export class MatrixRpc {
  constructor(port,{timeoutMs=15000}={}){
    this.port=port;this.timeoutMs=timeoutMs;this.nextId=1;this.pending=new Map();this.closed=false;
    this.ready=new Promise((resolve,reject)=>{this.readyResolve=resolve;this.readyReject=reject;});
    this.readyTimer=setTimeout(()=>this.fail(Object.assign(new Error('Service startup timeout'),{code:'Timeout'})),timeoutMs);
    port.onmessage=event=>{
      const value=event.data;
      if(value?.event==='ready'){clearTimeout(this.readyTimer);this.info=value.info;this.limits=value.limits;this.readyResolve(value);return;}
      if(value?.event==='closed'||value?.event==='unavailable'){this.fail(Object.assign(new Error(value.reason||value.error||'Closed'),{code:'Closed'}));return;}
      const ticket=this.pending.get(value?.id);if(!ticket)return;
      this.pending.delete(value.id);clearTimeout(ticket.timer);
      value.ok?ticket.resolve(value.result):ticket.reject(Object.assign(new Error(value.error?.message||'RPC failure'),{code:value.error?.code||'Protocol'}));
    };
    port.onmessageerror=()=>this.fail(Object.assign(new Error('Message transfer failed'),{code:'Transport'}));port.start();
  }
  fail(error){if(this.closed)return;this.closed=true;clearTimeout(this.readyTimer);this.readyReject(error);for(const value of this.pending.values()){clearTimeout(value.timer);value.reject(error);}this.pending.clear();this.port.close();}
  send(op,fields={},transfer=[]){
    if(this.closed)return {id:0,promise:Promise.reject(Object.assign(new Error('Client closed'),{code:'Closed'}))};
    const id=this.nextId++;
    const promise=new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{
        try{this.port.postMessage({id:this.nextId++,op:'close'});}catch{}
        this.fail(Object.assign(new Error('RPC timeout; session abandoned'),{code:'Timeout'}));
      },this.timeoutMs);
      this.pending.set(id,{resolve,reject,timer});
      try{this.port.postMessage({id,op,...fields},transfer);}catch(error){clearTimeout(timer);this.pending.delete(id);reject(error);}
    });
    return {id,promise};
  }
  async upload(data,rows,cols){
    // PyProxy.toJs copies numeric ndarray buffers. Destroy any temporary proxy
    // in the caller when conversion does not return a JS TypedArray.
    if(typeof data?.toJs==='function')data=data.toJs();
    if(!(data instanceof Float32Array))throw new TypeError('Upload requires contiguous float32 data');
    const copy=new Float32Array(data);
    return this.send('upload',{rows,cols,data:copy.buffer},[copy.buffer]).promise;
  }
  matmul(a,b,transposeA=false,transposeB=false){return this.send('matmul',{a,b,transposeA,transposeB}).promise;}
  async read(handle){const value=await this.send('read',{handle}).promise;return new Float32Array(value.data);}
  release(handle){return this.send('release',{handle}).promise;}
  releaseMany(handles){return this.send('release_many',{handles}).promise;}
  async program(inputs,nodes,outputs){
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
  stats(){return this.send('stats').promise;}
  cancel(target){return this.send('cancel',{target}).promise;}
  async close(){if(this.closed)return;try{await this.send('close').promise;}finally{this.fail(Object.assign(new Error('Client closed'),{code:'Closed'}));}}
}
