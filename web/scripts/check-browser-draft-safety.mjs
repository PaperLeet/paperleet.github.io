import assert from 'node:assert/strict';

export async function checkBrowserDraftSafety(server) {
  const {keepDraftSafety,clearDraftSafety,readDraftSafety,hasUnsavedDraftSafety} = await server.ssrLoadModule('/src/browserDraftSafety.ts');
  const originalLocal = Object.getOwnPropertyDescriptor(globalThis,'localStorage');
  const originalSession = Object.getOwnPropertyDescriptor(globalThis,'sessionStorage');
  const storage = () => {
    const entries = new Map();
    return {get length(){return entries.size;},key:i=>[...entries.keys()][i] ?? null,
      getItem:key=>entries.get(key) ?? null,setItem:(key,value)=>entries.set(key,value),removeItem:key=>entries.delete(key)};
  };
  const local = storage(), session = storage();
  const use = (name,value) => Object.defineProperty(globalThis,name,{configurable:true,value});
  const id = 'architecture/guest-cache', key = `paperleet.practice.safety.v2:anonymous:${id}`;
  const write = {code:'def 变量():\r\n\treturn "🧪"\r\n',updated:'2026-10-14T12:00:00Z',token:'first'};
  try {
    use('localStorage',local); use('sessionStorage',session);
    keepDraftSafety('anonymous',id,write,'original-version');
    const durable = local.getItem(key);
    assert.equal(JSON.parse(durable).code,write.code,'Typing must synchronously cache the exact guest code before any async save');
    assert(hasUnsavedDraftSafety());
    clearDraftSafety('anonymous',id,write.token);
    assert(!hasUnsavedDraftSafety());
    // Simulate closing the document: no memory or session journal survives.
    local.setItem(key,durable); use('sessionStorage',storage());
    assert.equal(readDraftSafety('anonymous')[0].code,write.code,'A fresh tab can recover an interrupted save from localStorage');
    assert.deepEqual(readDraftSafety('local'),[],'Guest and local workspace journals stay separate');
    local.setItem('paperleet.practice.safety.v2:anonymous:broken','{bad json');
    assert.equal(readDraftSafety('anonymous').length,1,'A broken entry cannot suppress other cached modules');
    const empty = {...write,code:'',token:'empty'};
    keepDraftSafety('anonymous',id,empty,'original-version');
    clearDraftSafety('anonymous',id,write.token);
    assert.equal(JSON.parse(local.getItem(key)).code,'','A late save cannot erase a newer empty-file edit');
    clearDraftSafety('anonymous',id,empty.token);
    assert.equal(local.getItem(key),null,'A committed save removes its temporary cache');
    const blocked = {get length(){throw Error('blocked');},setItem(){throw Error('blocked');},getItem(){throw Error('blocked');}};
    use('sessionStorage',blocked);
    local.setItem(key,durable);
    assert.equal(readDraftSafety('anonymous')[0].code,write.code,'Closing/reopening still recovers code when sessionStorage is unavailable');
    use('localStorage',blocked); use('sessionStorage',session);
    keepDraftSafety('anonymous',id,write,'original-version');
    assert.equal(JSON.parse(session.getItem(key)).code,write.code,'Session recovery remains available if localStorage is full');
    clearDraftSafety('anonymous',id,write.token);
    use('sessionStorage',blocked);
    keepDraftSafety('anonymous',id,write,'original-version');
    assert.equal(readDraftSafety('anonymous')[0].code,write.code,'Blocked browser storage retains the visible in-memory draft and unload guard');
    clearDraftSafety('anonymous',id,write.token);
  } finally {
    for (const [name,descriptor] of [['localStorage',originalLocal],['sessionStorage',originalSession]]) {
      if (descriptor) Object.defineProperty(globalThis,name,descriptor); else delete globalThis[name];
    }
  }
  console.log('PASS guest draft journal: synchronous local cache, fresh-tab recovery, empty/Unicode source, owner isolation, corrupt entries, stale-save cleanup and storage failures');
}
