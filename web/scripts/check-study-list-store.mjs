import assert from 'node:assert/strict';
import {createServer} from 'vite';
import {readFileSync} from 'node:fs';
const server=await createServer({server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
try{
 const {createStudyListStore,studyListsSnapshot}=await server.ssrLoadModule('/src/studyListStore.ts');
 const {favoriteHref,readFavoriteRoute,withLibraryContext,libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const {signInHref,accountReturnHref}=await server.ssrLoadModule('/src/accountNavigation.ts');
 const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 const id='a'.repeat(32),pid='architecture/attention',at='2026-10-17T00:00:00Z',owner='store-owner';
 const list=(title='Review',ids=[])=>({id,title,created_at:at,updated_at:at,items:ids.map(problem_id=>({problem_id,created_at:at}))});
 const snap=(revision,lists=[],who=owner)=>({owner_id:who,revision,lists});
 assert.deepEqual(studyListsSnapshot(snap(0),owner),snap(0));
 const catalog=JSON.parse(readFileSync('public/data/catalog.json','utf8'));assert.equal(studyListsSnapshot(snap(1,[list('All maintained modules',catalog.problems.map(p=>p.id))]),owner).lists[0].items.length,catalog.problems.length);
 for(const bad of [snap(-1),snap(1,[],'other'),snap(1,[list(),list()]),snap(1,[list('')]),snap(1,[list('x'.repeat(61))]),snap(1,[list('bad\nname')]),snap(1,[list('Review',[pid,pid])]),snap(1,[list('Review',['../../secret'])]),{...snap(0),lists:null}])assert.throws(()=>studyListsSnapshot(bad,owner));
 const href=favoriteHref({query:'注意力 & LoRA',state:'todo',list:id});assert.deepEqual(readFavoriteRoute(href),{query:'注意力 & LoRA',state:'todo',list:id});
 for(const path of [22,pid]){const target=withLibraryContext(path,href);assert.equal(libraryReturnHref(target),href);assert.equal(accountReturnHref(decodeProblemHash(signInHref(target))),target);}
 for(const bad of ['favorites?list=','favorites?list='+'A'.repeat(32),'favorites?list='+id+'&list='+id,'favorites?list='+id+'&owner=other','favorites?list='+id+'&from=community'])assert.equal(readFavoriteRoute(bad),undefined);
 let reads=[],writes=[],notices=0;const store=createStudyListStore(owner,(body,signal)=>new Promise((resolve,reject)=>(body?writes:reads).push({body,signal,resolve,reject})),()=>notices++);
 const first=store.refresh(),duplicate=store.refresh();assert.equal(reads.length,1);reads.shift().resolve(snap(0));await Promise.all([first,duplicate]);
 const op={action:'create',list_id:id,title:'Review',problem_ids:[pid]},original=structuredClone(op),write=store.mutate(op);op.title='Caller changed';op.problem_ids.push('finetuning/lora');assert.deepEqual(writes[0].body,{...original,expected_user_id:owner,expected_revision:0,request_id:writes[0].body.request_id});
 const command=writes[0].body;const response=(snapshot,body=command,revision=snapshot.revision)=>{const {expected_user_id,expected_revision,request_id,...operation}=body;return {...snapshot,ack:{request_id,operation,revision}};};
 const late=store.refresh();assert.equal(reads.length,1);writes.shift().resolve(response(snap(1,[list('Review',[pid])])));assert(await write);reads.shift().resolve(snap(0));await late;assert.equal(store.getSnapshot().snapshot.revision,1);assert.equal(notices,1);
 const rename=store.mutate({action:'rename',list_id:id,title:'Updated'}),lost=writes.shift();lost.reject(Error('Lost response'));assert.equal(await rename,false);assert(store.getSnapshot().pending.error);const blocked=await store.mutate({action:'delete',list_id:id});assert.equal(blocked,false);assert.equal(writes.length,0);
 const retry=store.retry(),replayed=writes.shift();assert.deepEqual(replayed.body,lost.body);replayed.resolve(response(snap(3,[list('Changed later',[pid])]),replayed.body,2));assert(await retry);assert.equal(store.getSnapshot().snapshot.lists[0].title,'Changed later','A confirmed replay preserves newer server state');assert(!store.getSnapshot().pending);
 const stale=store.mutate({action:'rename',list_id:id,title:'Draft stays'}),conflict=writes.shift();conflict.reject(Object.assign(Error('Another device'),{status:409,code:'study_list_changed'}));assert.equal(await stale,false);assert(store.getSnapshot().pending.conflict);assert.equal(await store.retry(),false);const latest=store.loadLatest();reads.shift().resolve(snap(4,[list('Other device',[pid])]));assert(await latest);assert(!store.getSnapshot().pending);
 const invalid=store.mutate({action:'delete',list_id:id}),unconfirmed=writes.shift();unconfirmed.resolve(response(snap(5,[list('Still here',[pid])]),unconfirmed.body));assert.equal(await invalid,false);assert(store.getSnapshot().pending.error);assert.equal(store.getSnapshot().snapshot.revision,4,'An echoed deletion with the item still present is not accepted');
 const confirm=store.retry(),final=writes.shift();assert.deepEqual(final.body,unconfirmed.body);final.resolve(response(snap(5),final.body));assert(await confirm);
 const rejected=store.mutate({action:'create',list_id:id,title:'New'}),denied=writes.shift();denied.reject(Object.assign(Error('Invalid title'),{status:400}));assert.equal(await rejected,false);assert(store.getSnapshot().pending.rejected);store.dismissRejected();assert(!store.getSnapshot().pending);
 const foreign=store.refresh();reads.shift().resolve(snap(6,[list('Private other')],'other'));await foreign;assert(store.getSnapshot().error);assert.equal(store.getSnapshot().snapshot.revision,5);assert.equal(store.getSnapshot().snapshot.lists.length,0);
 console.log('PASS private study-list store: strict owner/list/order/count/title snapshots, canonical filtered practice/login returns, coalesced reads and monotonic revisions, captured immutable intent, exact ambiguous retries and newer-state receipts, effect-confirmed acknowledgements, explicit concurrent-edit recovery, known rejection and foreign-response isolation');
}finally{await server.close();}
