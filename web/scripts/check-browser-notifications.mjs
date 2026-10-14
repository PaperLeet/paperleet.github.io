import assert from 'node:assert/strict';
import 'fake-indexeddb/auto';

export async function checkBrowserNotifications(server) {
  const {browserNotifications}=await server.ssrLoadModule('/src/browserNotifications.ts');
  const {createLearningController}=await server.ssrLoadModule('/src/browserLearningController.ts');
  const {loadBrowserLearning,saveBrowserLearning}=await server.ssrLoadModule('/src/browserLearningStore.ts');
  const {learningKey}=await server.ssrLoadModule('/src/browserLearning.ts');
  const original=new Map(['window','document','BroadcastChannel'].map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const channels=[];
  class Channel {
    constructor(name){this.name=name;this.posts=[];this.closed=0;channels.push(this);}
    postMessage(message){this.posts.push(message);if(this.sendFails)throw new DOMException('Unavailable','InvalidStateError');}
    close(){this.closed++;if(this.closeFails)throw new Error('Close failed');}
  }
  const until=async predicate=>{
    const deadline=Date.now()+3000;
    while(!predicate()){assert(Date.now()<deadline,'Timed out waiting for the actual learning transaction');await new Promise(resolve=>setTimeout(resolve,5));}
  };
  let controller;
  try {
    globalThis.window=new EventTarget();globalThis.document=new EventTarget();document.visibilityState='visible';
    for(const implementation of [undefined,class{constructor(){throw new DOMException('Blocked','SecurityError');}}]){
      globalThis.BroadcastChannel=implementation;
      let refreshes=0;
      const notifications=browserNotifications('practice','local',()=>refreshes++);
      notifications.notify();
      window.dispatchEvent(new Event('focus'));
      document.visibilityState='hidden';document.dispatchEvent(new Event('visibilitychange'));
      assert.equal(refreshes,1);
      document.visibilityState='visible';document.dispatchEvent(new Event('visibilitychange'));
      assert.equal(refreshes,2,'Visibility refresh remains available when channel creation is unavailable');
      notifications.dispose();window.dispatchEvent(new Event('focus'));
      assert.equal(refreshes,2,'Disposed listeners cannot refresh another owner');
    }
    globalThis.BroadcastChannel=Channel;
    let refreshes=0;
    const notifications=browserNotifications('practice','local',()=>refreshes++),channel=channels.at(-1);
    channel.onmessage({data:{owner:'anonymous'}});channel.onmessage({data:null});
    assert.equal(refreshes,0);
    channel.onmessage({data:{owner:'local'}});assert.equal(refreshes,1);
    notifications.notify();assert.deepEqual(channel.posts,[{owner:'local'}],'Cross-tab hints contain only the owner, never source or reports');
    channel.sendFails=true;channel.closeFails=true;
    notifications.notify();notifications.notify();
    assert.equal(channel.posts.length,2,'A broken channel is disabled rather than retried on each save');
    assert.equal(channel.closed,1);
    window.dispatchEvent(new Event('focus'));assert.equal(refreshes,2);
    const lateMessage=channel.onmessage;notifications.dispose();lateMessage({data:{owner:'local'}});
    document.dispatchEvent(new Event('visibilitychange'));assert.equal(refreshes,2);

    const problem={id:'learning/notification-test',tests:{T0:1},browser_judge:{bundle_url:'browser-judge/notification-suite.json'}};
    const passed={problem:problem.id,runtime:'browser-numpy-wasm',tests:[{id:'case',layer:'T0',passed:true}],summary:{T0:[1,1]}};
    const failed={...passed,tests:[{id:'case',layer:'T0',passed:false}],summary:{T0:[0,1]}};
    const key=learningKey(problem.id,problem.browser_judge.bundle_url);
    for(const [mode,implementation] of [['blocked',class{constructor(){throw new DOMException('Blocked','SecurityError');}}],['send',Channel]]){
      globalThis.BroadcastChannel=implementation;
      const owner=`user:notifications-${mode}`,views=[];
      controller=createLearningController(owner,{},view=>views.push(view));
      if(mode==='send'){channels.at(-1).sendFails=true;channels.at(-1).closeFails=true;}
      await until(()=>views.at(-1)?.ready);
      controller.enqueue({type:'report',problem,report:passed,version:'pass'});
      controller.enqueue({type:'report',problem,report:failed,version:'fail'});
      await until(()=>!views.at(-1).saving&&views.at(-1).records[key]?.version==='fail');
      assert(views.every(view=>!view.failed),'A notification failure is never reported as a learning-cache write failure');
      const durable=(await loadBrowserLearning(owner))[key];
      assert(!durable.report.tests[0].passed&&durable.acceptedReport.tests[0].passed,'Both serialized writes commit despite notification failure');
      await saveBrowserLearning(owner,problem,passed,'peer-pass');
      window.dispatchEvent(new Event('focus'));
      await until(()=>views.at(-1).records[key]?.version==='peer-pass');
      await saveBrowserLearning(owner,problem,failed,'peer-failure');
      document.dispatchEvent(new Event('visibilitychange'));
      await until(()=>views.at(-1).records[key]?.version==='peer-failure');
      const count=views.length;controller.dispose();controller=undefined;
      window.dispatchEvent(new Event('focus'));document.dispatchEvent(new Event('visibilitychange'));
      await new Promise(resolve=>setTimeout(resolve,10));assert.equal(views.length,count);
    }
  } finally {
    controller?.dispose();
    for(const [name,descriptor] of original){if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];}
  }
  console.log('PASS optional tab notifications: blocked/missing API, send/close failure, owner-only hints, disposal, durable controller pass/failure ordering and focus/visibility peer recovery');
}
