// Validate the complete fixed-matmul DAG before creating GPU resources.
const fail=(code,message)=>{throw Object.assign(new Error(message),{code});};
const requireValue=(ok,code,message)=>{if(!ok)fail(code,message);};
const integer=(value,min,max)=>Number.isSafeInteger(value)&&value>=min&&value<=max;
function keys(value,names){requireValue(value!==null&&typeof value==='object'&&!Array.isArray(value)&&Object.getPrototypeOf(value)===Object.prototype&&Object.keys(value).length===names.length&&names.every(name=>Object.hasOwn(value,name)),'Protocol','Unexpected program fields');}
export function planProgram(request,{limits,find,residentCount,totalBytes}){
  keys(request,['id','op','inputs','nodes','outputs']);
  for(const [name,max] of [['inputs',limits.programInputs],['nodes',limits.programNodes],['outputs',limits.programOutputs]])
    requireValue(Array.isArray(request[name])&&integer(request[name].length,1,max),'Protocol','Invalid bounded program '+name);
  let bytes=0,macs=0,retained=0,readBytes=0;
  const values=[],inputs=[],nodes=[],outputs=[];
  function shape(rows,cols){
    requireValue(integer(rows,1,limits.dimension)&&integer(cols,1,limits.dimension),'Shape','Positive bounded 2D shapes required');
    requireValue(rows*cols*4<=limits.matrixBytes,'Capacity','Program matrix byte limit exceeded');
    return {rows,cols,bytes:rows*cols*4};
  }
  for(const entry of request.inputs){
    if(entry!==null&&typeof entry==='object'&&Object.hasOwn(entry,'handle')){
      keys(entry,['handle']);const value=find(entry.handle);inputs.push({existing:value});values.push(shape(value.rows,value.cols));
    }else{
      keys(entry,['rows','cols','data']);const value=shape(entry.rows,entry.cols);
      requireValue(entry.data instanceof ArrayBuffer,'Dtype','Program upload requires a float32 ArrayBuffer');
      requireValue(entry.data.byteLength===value.bytes,'Shape','Program upload length differs from shape');
      bytes+=value.bytes;inputs.push({...value,data:entry.data});values.push(value);
    }
  }
  for(const entry of request.nodes){
    keys(entry,['a','b','transposeA','transposeB']);
    requireValue(integer(entry.a,0,values.length-1)&&integer(entry.b,0,values.length-1),'Reference','Program references must precede their node');
    requireValue(typeof entry.transposeA==='boolean'&&typeof entry.transposeB==='boolean','Protocol','Transpose flags must be booleans');
    const a=values[entry.a],b=values[entry.b],m=entry.transposeA?a.cols:a.rows,k=entry.transposeA?a.rows:a.cols,n=entry.transposeB?b.rows:b.cols;
    requireValue(k===(entry.transposeB?b.cols:b.rows),'Shape','Program inner dimensions differ');
    const value=shape(m,n);macs+=m*n*k;bytes+=value.bytes+32;
    requireValue(macs<=limits.macs,'Capacity','Aggregate program work limit exceeded');
    nodes.push({...entry,...value,k});values.push(value);
  }
  const seen=new Set();
  for(const entry of request.outputs){
    keys(entry,['ref','read','retain']);
    requireValue(integer(entry.ref,0,values.length-1),'Reference','Invalid output reference');
    if(entry.ref<inputs.length)requireValue(!inputs[entry.ref].existing && entry.read===false && entry.retain===true,'Reference','Only new uploads may be exported as unread retained inputs');
    requireValue(!seen.has(entry.ref),'Protocol','Program outputs must be unique');seen.add(entry.ref);
    requireValue(typeof entry.read==='boolean'&&typeof entry.retain==='boolean'&&(entry.read||entry.retain),'Protocol','An output must be read or retained');
    const value=values[entry.ref];if(entry.read)readBytes+=value.bytes;if(entry.retain)retained++;
    outputs.push({...entry,...value});
  }
  requireValue(residentCount+retained<=limits.handles,'Capacity','Program retained handle limit exceeded');
  requireValue(totalBytes+bytes+readBytes<=limits.bytes,'Capacity','Aggregate program allocation budget exceeded');
  return {inputs,nodes,outputs,bytes:bytes+readBytes,readBytes,macs};
}
