import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';
const source=await readFile(new URL('../src/readingPreferences.ts',import.meta.url),'utf8');
const js=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const model=await import('data:text/javascript;base64,'+Buffer.from(js).toString('base64'));
const key=model.READING_PREFERENCES_KEY,defaults=model.defaultReadingPreferences;
for(const raw of [null,'','{','[]','null','true','{}',JSON.stringify({version:2,size:'18',spacing:'default'}),JSON.stringify({version:1,size:18,spacing:'default'}),JSON.stringify({version:1,size:'20px; background: url(https://tracker.test)',spacing:'default'}),JSON.stringify({version:1,size:'20',spacing:'default',code:'PRIVATE'}),'x'.repeat(1000)])assert.deepEqual(model.parseReadingPreferences(raw),defaults);
assert.deepEqual(model.parseReadingPreferences(JSON.stringify({version:1,size:'18',spacing:'relaxed'})),{version:1,size:'18',spacing:'relaxed'});
function fixture(raw){
 const values=new Map([['paperleet.editor.fontSize','16'],['private-draft','PRIVATE_DRAFT']]);if(raw!==undefined)values.set(key,raw);
 const attributes=new Map([['data-theme','dark']]),events=new Map();let writes=0,reads=0,blocked=false;
 const storage={getItem(name){reads++;if(blocked)throw Error('Storage blocked');return values.get(name)??null;},setItem(name,value){assert.equal(name,key);if(blocked)throw Error('Storage blocked');writes++;values.set(name,value);},removeItem(name){assert.equal(name,key);if(blocked)throw Error('Storage blocked');writes++;values.delete(name);}};
 const host={document:{documentElement:{hasAttribute:name=>attributes.has(name),getAttribute:name=>attributes.get(name)??null,setAttribute:(name,value)=>attributes.set(name,value),removeAttribute:name=>attributes.delete(name)}},get localStorage(){if(blocked)throw Error('Storage access blocked');return storage;},addEventListener(name,listener){if(!events.has(name))events.set(name,new Set());events.get(name).add(listener);},removeEventListener(name,listener){events.get(name)?.delete(listener);}};
 return {host,storage,values,attributes,events,get writes(){return writes;},get reads(){return reads;},set blocked(value){blocked=value;},emit(extra={}){for(const listener of events.get('storage')||[])listener({key,storageArea:storage,...extra});}};
}
const first=fixture(JSON.stringify({version:1,size:'18',spacing:'relaxed'})),store=model.createReadingPreferencesStore(first.host);let notifications=0;const unsubscribe=store.subscribe(()=>notifications++);
assert.equal(first.writes,0);assert.equal(first.attributes.get('data-reading-size'),'18');assert.equal(first.attributes.get('data-reading-spacing'),'relaxed');assert.equal(first.events.get('storage').size,1);const initial=store.getSnapshot();assert.equal(store.getSnapshot(),initial);assert(Object.isFrozen(initial));
store.set({size:'20'});assert.equal(first.attributes.get('data-reading-size'),'20');assert.equal(notifications,1);assert.deepEqual(JSON.parse(first.values.get(key)),{version:1,size:'20',spacing:'relaxed'});assert.equal(first.values.get('private-draft'),'PRIVATE_DRAFT');assert.equal(first.values.get('paperleet.editor.fontSize'),'16');assert.equal(first.attributes.get('data-theme'),'dark');
const writes=first.writes,stable=store.getSnapshot();store.set({size:'20'});assert.equal(first.writes,writes);assert.equal(store.getSnapshot(),stable);
for(const patch of [null,[],{size:'99'},{size:undefined},{spacing:'tiny'},{version:2},{fontSize:24},{size:'18',code:'PRIVATE'}])store.set(patch);assert.equal(first.writes,writes);assert.equal(store.getSnapshot(),stable);
console.log('PASS reading preference state: bounded strict versioned values, immutable stable snapshots, safe CSS attributes, exact known preference writes, duplicate/invalid update rejection and independent editor/private/theme state');
const beforeReads=first.reads;first.emit({key:'private-draft',newValue:'PRIVATE_EVENT'});first.emit({storageArea:{}});assert.equal(first.reads,beforeReads);
first.emit({newValue:JSON.stringify({version:1,size:'18',spacing:'default'})});assert.equal(store.getSnapshot().size,'20','Stale event payload cannot roll back a newer stored value');assert.equal(first.writes,writes);
first.values.set(key,JSON.stringify({version:1,size:'18',spacing:'default'}));first.emit();assert.equal(store.getSnapshot().size,'18');assert(!first.attributes.has('data-reading-spacing'));assert.equal(first.writes,writes,'External updates never echo writes');
first.values.delete(key);first.emit({key:null});assert.deepEqual({...store.getSnapshot()},{...defaults,saved:true});assert(!first.attributes.has('data-reading-size'));assert.equal(first.values.get('private-draft'),'PRIVATE_DRAFT');
first.blocked=true;store.set({size:'20',spacing:'relaxed'});assert.equal(store.getSnapshot().size,'20');assert.equal(store.getSnapshot().saved,false);assert.equal(first.attributes.get('data-reading-size'),'20');first.blocked=false;store.retry();assert.equal(store.getSnapshot().saved,true);assert.equal(JSON.parse(first.values.get(key)).size,'20');store.reset();assert(!first.values.has(key));assert(!first.attributes.has('data-reading-size'));assert(!first.attributes.has('data-reading-spacing'));assert.equal(first.values.get('private-draft'),'PRIVATE_DRAFT');assert.equal(first.values.get('paperleet.editor.fontSize'),'16');
unsubscribe();store.destroy();assert.equal(first.events.get('storage').size,0);const ended=store.getSnapshot();first.values.set(key,JSON.stringify({version:1,size:'18',spacing:'default'}));first.emit();assert.equal(store.getSnapshot(),ended);
console.log('PASS reading preference persistence: fresh cross-tab reads, foreign-key/area rejection, external clear/reset, blocked-storage session application and explicit retry, no write echo, source-key-only reset and subscription/listener cleanup');
const malformed=fixture('PRIVATE_INVALID_JSON'),invalid=model.createReadingPreferencesStore(malformed.host);assert.equal(malformed.writes,0);assert.equal(malformed.values.get(key),'PRIVATE_INVALID_JSON');assert.equal(invalid.getSnapshot().size,'default');invalid.destroy();
const inaccessible=fixture();inaccessible.blocked=true;const volatile=model.createReadingPreferencesStore(inaccessible.host);assert.equal(volatile.getSnapshot().saved,false);volatile.set({size:'18'});assert.equal(inaccessible.attributes.get('data-reading-size'),'18');volatile.destroy();
const server=model.createReadingPreferencesStore();assert.deepEqual(server.getServerSnapshot(),{...defaults,saved:true});assert.equal(server.getServerSnapshot(),server.getServerSnapshot());server.destroy();
console.log('PASS reading preference recovery: malformed stored data is ignored without destructive repair, inaccessible storage retains working session appearance, and deterministic server snapshots need no browser or account');

const statementCode=await readFile(new URL('../src/webStatementReadme.ts',import.meta.url),'utf8');
const statementJs=ts.transpileModule(statementCode,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {webStatementReadme}=await import('data:text/javascript;base64,'+Buffer.from(statementJs).toString('base64'));
for(const heading of ['怎么运行','编写与运行','在网页中运行','Write and run','Run your implementation','Running']){
 const input='# Module\n\n## 5. Contract\nKeep the math.\n\n## 6. '+heading+'\nObsolete instructions.\n\n## 7. Checks\nKeep these checks.\n';
 const browser=webStatementReadme(input,'browser',(_zh,en)=>en),server=webStatementReadme(input,'server',(_zh,en)=>en),chinese=webStatementReadme(input,'browser',zh=>zh);
 assert(browser.includes('Run tests to unlock Notes'));assert(!browser.includes('Obsolete instructions'));assert(browser.includes('Guest drafts save in this browser'));assert(browser.includes('## 7. Checks\nKeep these checks.'));assert(server.includes('The server judges your solution'));assert(chinese.includes('点击「运行测试」'));assert(!browser.includes('\\n'));
}
for(const id of ['architecture/attention','architecture/flash_attention','optimization/adamw','foundations/next_token_loss']){
 const readme=await readFile(new URL('../../problems/'+id+'/README.md',import.meta.url),'utf8');assert.equal(webStatementReadme(readme,'browser',(_zh,en)=>en),readme,'Modern lesson contracts and formulas stay exact');
}
console.log('PASS deferred statement instructions: browser/server and bilingual legacy run sections,retained checks and real newlines,with four current lesson contracts left byte-for-byte unchanged; no startup or judging work');
