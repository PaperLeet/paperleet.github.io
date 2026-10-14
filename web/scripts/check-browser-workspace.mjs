import assert from 'node:assert/strict';
import React from 'react';
import {renderToString} from 'react-dom/server';
import 'fake-indexeddb/auto';

export async function checkBrowserWorkspace(server) {
  const {loadBrowserWorkspace,writeBrowserDraft,writeBrowserReport,deleteBrowserBackup,legacyBrowserWorkspace} = await server.ssrLoadModule('/src/browserWorkspaceStore.ts');
  const {BrowserDraftBackups} = await server.ssrLoadModule('/src/BrowserDraftBackups.tsx');
  const {LocaleProvider} = await server.ssrLoadModule('/src/i18n.tsx');
  const owner='local', stamp='2026-10-14T12:00:00Z';
  const problem={id:'architecture/store-test',test_count:1};
  const accepted={problem:problem.id,tests:[{id:'case',layer:'T0',passed:true}],summary:{T0:[1,1]},created_at:stamp};
  const failed={...accepted,tests:[{...accepted.tests[0],passed:false}],summary:{T0:[0,1]}};
  const source='def 变量():\r\n\treturn "🧪"\r\n';
  await Promise.all(Array.from({length:12},(_,i)=>writeBrowserDraft(owner,`module/${i}`,`tab-${i}`,'',source+i,stamp)));
  let snapshot=await loadBrowserWorkspace(owner);
  for(let i=0;i<12;i++)assert.equal(snapshot.records[`module/${i}`].saved.code,source+i,'Independent modules cannot erase another tab’s source');
  const initial=(await writeBrowserDraft(owner,'module/shared','seed','','original',stamp)).record;
  const races=await Promise.all([
    writeBrowserDraft(owner,'module/shared','tab-a',initial.version,'A',stamp),
    writeBrowserDraft(owner,'module/shared','tab-b',initial.version,'B',stamp),
  ]);
  assert.equal(races.filter(result=>result.conflict).length,1,'Exactly one writer may advance a shared base');
  snapshot=await loadBrowserWorkspace(owner);
  assert.deepEqual(new Set(snapshot.backups.filter(row=>row.id==='module/shared').map(row=>row.code)),new Set(['original','A','B']),'Even the previously saved source remains recoverable before another tab sees a notification');
  const shared=snapshot.records['module/shared'];
  const equal=await writeBrowserDraft(owner,'module/shared','third-tab','stale-version',shared.saved.code,stamp);
  assert(!equal.conflict&&equal.record.version===shared.version,'Stale equal source is a no-op rather than an overwrite');
  const switchId='module/version-switch', switchWriter='switching-tab';
  const original=(await writeBrowserDraft(owner,switchId,switchWriter,'','source before choosing another version',stamp)).record;
  const displaced={key:'first-choice',code:original.saved.code,updated:stamp,base:original.version};
  const selected=(await writeBrowserDraft(owner,switchId,switchWriter,original.version,'chosen alternative',stamp,[displaced])).record;
  snapshot=await loadBrowserWorkspace(owner);
  assert(snapshot.backups.some(row=>row.id===switchId&&row.code===displaced.code),'Choosing another version must not destroy the only retained copy of the editor source');
  await writeBrowserDraft(owner,switchId,switchWriter,selected.version,'typed after the choice',stamp,[displaced]);
  snapshot=await loadBrowserWorkspace(owner);
  assert.equal(snapshot.backups.filter(row=>row.id===switchId&&row.writer==='preserved:first-choice').length,1,'A coalesced edit retries the same preservation intent without duplicating it');
  const heldVersion=snapshot.backups.find(row=>row.id===switchId&&row.writer==='preserved:first-choice').version;
  const newer=snapshot.records[switchId];
  await writeBrowserDraft(owner,switchId,switchWriter,newer.version,'ordinary next edit',stamp,[displaced]);
  snapshot=await loadBrowserWorkspace(owner);
  assert.equal(snapshot.backups.find(row=>row.id===switchId&&row.writer==='preserved:first-choice').version,heldVersion,'A retry cannot change the version of an already preserved source');
  const b=(await writeBrowserDraft(owner,'module/shared','editor',shared.version,'temporary',stamp)).record;
  const a=(await writeBrowserDraft(owner,'module/shared','editor',b.version,shared.saved.code,stamp)).record;
  assert.notEqual(a.version,shared.version);
  assert((await writeBrowserDraft(owner,'module/shared','late-tab',shared.version,'late edit',stamp)).conflict,'A→B→A still rejects an obsolete base');
  const before=(await writeBrowserDraft(owner,problem.id,'runner','','submitted',stamp)).record;
  await writeBrowserReport(owner,problem,accepted);
  await writeBrowserDraft(owner,problem.id,'editor',before.version,'newer draft while judging',stamp);
  await writeBrowserReport(owner,problem,failed);
  snapshot=await loadBrowserWorkspace(owner);
  assert.equal(snapshot.records[problem.id].saved.code,'newer draft while judging');
  assert(snapshot.records[problem.id].saved.acceptedReport.tests[0].passed,'A later failed report must preserve earlier acceptance');
  assert(!snapshot.records[problem.id].saved.report.tests[0].passed);
  const review=snapshot.backups.find(row=>row.id==='module/shared'&&row.writer==='late-tab');
  await writeBrowserDraft(owner,'module/shared','late-tab','stale-version','a newer backup',stamp);
  await deleteBrowserBackup(review);
  snapshot=await loadBrowserWorkspace(owner);
  assert.equal(snapshot.backups.find(row=>row.id===review.id&&row.writer===review.writer).code,'a newer backup','A delayed discard cannot delete newer content in the same slot');
  const latestBackup=snapshot.backups.find(row=>row.id===review.id&&row.writer===review.writer);
  await deleteBrowserBackup(latestBackup);
  assert(!(await loadBrowserWorkspace(owner)).backups.some(row=>row.writer===review.writer));
  await Promise.all([
    loadBrowserWorkspace(owner,{'migration/one':{code:'old cache'},'migration/two':{code:'other old cache'}}),
    loadBrowserWorkspace(owner,{'migration/one':{code:'competing migration'}}),
  ]);
  const migrated=(await loadBrowserWorkspace(owner)).records['migration/one'];
  await writeBrowserDraft(owner,'migration/one','migrated-tab',migrated.version,'after migration',stamp);
  assert.equal((await loadBrowserWorkspace(owner,{'migration/one':{code:'obsolete legacy cache'}})).records['migration/one'].saved.code,'after migration');
  await writeBrowserDraft('anonymous','module/shared','guest','','guest-only',stamp);
  assert.equal((await loadBrowserWorkspace('anonymous')).records['module/shared'].saved.code,'guest-only');
  assert.notEqual((await loadBrowserWorkspace(owner)).records['module/shared'].saved.code,'guest-only');
  const originalPut=IDBObjectStore.prototype.put;
  try {
    IDBObjectStore.prototype.put=function(value,...rest){if(this.name==='backups')throw new DOMException('Simulated storage exhaustion','QuotaExceededError');return originalPut.call(this,value,...rest);};
    await assert.rejects(writeBrowserDraft(owner,'module/shared','quota',a.version,'must roll back',stamp),{name:'QuotaExceededError'});
    await assert.rejects(writeBrowserDraft(owner,switchId,switchWriter,snapshot.records[switchId].version,'failed switch',stamp,
      [{key:'quota-switch',code:'ordinary next edit',updated:stamp,base:snapshot.records[switchId].version}]),{name:'QuotaExceededError'});
  } finally {IDBObjectStore.prototype.put=originalPut;}
  snapshot=await loadBrowserWorkspace(owner);
  assert.equal(snapshot.records['module/shared'].saved.code,a.saved.code,'A backup write failure must roll back the shared source too');
  assert(!snapshot.backups.some(row=>row.writer==='quota'));
  assert.equal(snapshot.records[switchId].saved.code,'ordinary next edit');
  assert(!snapshot.backups.some(row=>row.writer==='preserved:quota-switch'),'A failed preservation transaction cannot partially archive or replace a draft');
  const originalStorage=globalThis.localStorage, originalLocation=globalThis.location;
  try {
    globalThis.location={origin:'https://paperleet.example'};
    globalThis.localStorage={getItem:key=>key==='paperleet.workspace.v1'?JSON.stringify({'migration/source':{code:source}}):JSON.stringify({'migration/source':{code:'anonymous draft',updated:stamp}})};
    assert.equal(legacyBrowserWorkspace('local')['migration/source'].code,source);
    assert.equal(legacyBrowserWorkspace('anonymous')['migration/source'].code,'anonymous draft');
    globalThis.localStorage={getItem(){throw new Error('blocked');}};
    assert.deepEqual(legacyBrowserWorkspace('local'),{});
    for(const en of [false,true]){
      globalThis.localStorage={getItem:()=>en?'en':'zh'};
      const backup={owner:'local',id:'architecture/demo',writer:'a',version:'a1',base:'old',code:'<script>not executable</script>\n',updated:stamp};
      const html=renderToString(React.createElement(LocaleProvider,null,React.createElement(BrowserDraftBackups,{
        backups:[backup,{...backup,writer:'b',version:'b1'}],source:'current source',conflict:{code:'other source'},busy:false,onRestore(){},async onDiscard(){},
      })));
      assert(html.includes(en?'Other recoverable versions (1)':'可恢复的其他版本（1）'),'Identical heads appear as one recoverable source');
      assert(!html.includes('<script>')&&html.includes('&lt;script&gt;'));
      assert(html.includes('tabindex="0"')&&html.includes(en?'Download backup':'下载备份'));
    }
  } finally {
    if(originalStorage===undefined)delete globalThis.localStorage;else globalThis.localStorage=originalStorage;
    if(originalLocation===undefined)delete globalThis.location;else globalThis.location=originalLocation;
  }
  console.log('PASS transactional browser drafts: cross-tab module isolation, concurrent/ABA conflicts, exact source, recoverable version switches, idempotent preservation, durable tab heads, late reports, acceptance preservation, migration, owner isolation, stale discard and atomic quota rollback');
}
