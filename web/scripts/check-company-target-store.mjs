import assert from 'node:assert/strict';
import {createServer} from 'vite';
const server=await createServer({server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
try{
 const {createCompanyTargetStore,companyTargetsSnapshot,MAX_COMPANY_TARGETS}=await server.ssrLoadModule('/src/companyTargetStore.ts');
 const at='2026-10-18T00:00:00Z',owner='alice',snap=(revision,ids=[],who=owner)=>({owner_id:who,revision,items:ids.map(company_id=>({company_id,created_at:at}))});
 assert.equal(MAX_COMPANY_TARGETS,12);assert.deepEqual(companyTargetsSnapshot(snap(0),owner),snap(0));
 for(const value of [null,snap(0,[],'bob'),snap(-1),snap(1.5),{...snap(1),items:{}},snap(1,['google','google']),snap(1,['../escape']),snap(1,Array.from({length:13},(_,n)=>'company-'+n)),{...snap(1,['google']),items:[{company_id:'google',created_at:'invalid'}]}])assert.throws(()=>companyTargetsSnapshot(value,owner));
 const pending=[],changes=[];const store=createCompanyTargetStore(owner,(body,signal)=>new Promise((resolve,reject)=>pending.push({body,signal,resolve,reject})),()=>changes.push(true));
 const get=()=>store.getSnapshot(),take=()=>{assert(pending.length);return pending.shift();};
 const first=store.refresh(),coalesced=store.refresh();assert.equal(pending.length,1);take().resolve(snap(0));await Promise.all([first,coalesced]);
 let saving=store.set('google',true),request=take();assert.equal(get().snapshot.items.length,0,'Unconfirmed changes never look saved');assert.equal(await store.set('google',true),false);assert.deepEqual(request.body,{expected_user_id:owner,company_id:'google',active:true});request.reject(Error('Lost response after commit'));assert.equal(await saving,false);assert(get().pending.google.error);
 saving=store.set('google',false);request=take();assert.equal(request.body.active,true,'Retry retains the original intent');request.resolve({...snap(1,['google']),company_id:'google',active:true});assert.equal(await saving,true);assert.equal(changes.length,1);assert.equal(get().pending.google,undefined);
 const reading=store.refresh();request=take();const adding=store.set('amazon',true),write=take();write.resolve({...snap(2,['amazon','google']),company_id:'amazon',active:true});assert(await adding);request.resolve(snap(1,['google']));await reading;assert.equal(get().snapshot.revision,2,'An earlier GET cannot roll back a confirmed save');
 const a=store.set('meta',true),ar=take(),b=store.set('openai',true),br=take();br.resolve({...snap(4,['openai','meta','amazon','google']),company_id:'openai',active:true});assert(await b);ar.resolve({...snap(3,['meta','amazon','google']),company_id:'meta',active:true});assert(await a);assert.equal(get().snapshot.revision,4);
 const conflict=store.refresh();take().resolve(snap(4,['google']));await conflict;assert(get().error);assert.equal(get().snapshot.items.length,4);assert.equal(await store.set('google',false),false);const recovery=store.refresh();take().resolve(snap(4,['openai','meta','amazon','google']));await recovery;
 saving=store.set('google',false);take().resolve({...snap(5,['openai','meta','amazon']),company_id:'google',active:true});assert.equal(await saving,false,'Mismatched acknowledgement is not confirmed');assert(get().pending.google.error);
 saving=store.set('google',true);take().resolve({...snap(5,['openai','meta','amazon']),company_id:'google',active:false});assert(await saving);assert(!get().snapshot.items.some(row=>row.company_id==='google'));
 const oldRead=store.refresh(),old=take();store.stop();assert(old.signal.aborted);store.start();const freshRead=store.refresh(),fresh=take();old.resolve(snap(99,['google']));fresh.resolve(snap(6,['meta']));await Promise.all([oldRead,freshRead]);assert.equal(get().snapshot.revision,6,'StrictMode restart ignores the retired generation');
 saving=store.set('google',true);request=take();store.stop();assert(request.signal.aborted);request.resolve({...snap(7,['google','meta']),company_id:'google',active:true});assert.equal(await saving,false);const before=changes.length;assert.equal(await store.set('amazon',true),false);assert.equal(changes.length,before);store.start();const identity=store.refresh();take().resolve(snap(0,[],'bob'));await identity;assert(get().accountError);assert.equal(get().snapshot,undefined);assert.deepEqual(get().pending,{});store.stop();
 // Either a mismatched GET or a revoked write retires every concurrent request.
 for(const revokedWrite of [false,true]){
  const queue=[],events=[];const guarded=createCompanyTargetStore(owner,(body,signal)=>new Promise((resolve,reject)=>queue.push({body,signal,resolve,reject})),()=>events.push(true));
  const initial=guarded.refresh();queue.shift().resolve(snap(0));await initial;
  const writing=guarded.set('google',true),write=queue.shift();
  const confirming=revokedWrite?guarded.set('meta',true):guarded.refresh(),confirmation=queue.shift();
  if(revokedWrite)confirmation.reject(Object.assign(Error('Session revoked'),{status:401}));else confirmation.resolve(snap(0,[],'bob'));
  await confirming;assert(guarded.getSnapshot().accountError);assert(write.signal.aborted,'Identity failure must abort every in-flight operation');
  write.resolve({...snap(99,['google']),company_id:'google',active:true});assert.equal(await writing,false);assert.equal(guarded.getSnapshot().snapshot,undefined,'Account mismatch must retire every concurrent write before late acknowledgements can restore private targets');assert.deepEqual(events,[]);guarded.stop();
 }
 console.log('PASS target-company store: strict private snapshots and bounds; coalesced reads; confirmed-only membership; fixed-direction lost-response retry; independent concurrent companies and monotonic revisions; conflicting and malformed acknowledgements; read recovery; StrictMode generation retirement; aborted owner responses and no private state on account mismatch');
}finally{await server.close();}
