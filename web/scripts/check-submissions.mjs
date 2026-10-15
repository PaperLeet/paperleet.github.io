import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

const dom = new JSDOM('<!doctype html><div id="root"></div>', {url:'http://127.0.0.1/'});
Object.defineProperty(dom.window.navigator,'languages',{value:['zh-CN'],configurable:true});
const previous = Object.fromEntries(['window','document','navigator','HTMLElement','IS_REACT_ACT_ENVIRONMENT','fetch','__proposalAccountFixture'].map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for (const key of ['window','document','navigator','HTMLElement']) Object.defineProperty(globalThis,key,{value:dom.window[key],configurable:true,writable:true});
globalThis.IS_REACT_ACT_ENVIRONMENT=true;
const React=await import('react');
const {createRoot}=await import('react-dom/client');
const {act}=React;
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
  name:'submission-account-fixture',enforce:'pre',
  resolveId(source,importer){if(source==='./social'&&importer?.endsWith('/src/Submissions.tsx'))return '\0submission-account-fixture';},
  load(id){if(id==='\0submission-account-fixture')return 'export const useAccount=()=>globalThis.__proposalAccountFixture;';},
}]});
let root;
try {
  const {LegacySubmissions:Submissions}=await server.ssrLoadModule('/src/Submissions.tsx');
  const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
  function LanguageSwitch(){const {setLocale}=useI18n();return React.createElement('button',{onClick:()=>setLocale('en')},'English fixture');}
  const requestQueue=[];
  globalThis.fetch=(url,options={})=>new Promise((resolve,reject)=>requestQueue.push({url,method:options.method||'GET',body:options.body,resolve,reject}));
  const item=(id,title,status='submitted')=>({id,title,status,paper_url:'https://arxiv.org/abs/2002.02126',created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',reviews:[]});
  const A=item('a'.repeat(32),'Proposal A'), B=item('b'.repeat(32),'Proposal B');
  const next=(url,method='GET')=>{const i=requestQueue.findIndex(r=>r.url===url&&r.method===method);assert(i>=0,`Missing ${method} ${url}: ${requestQueue.map(r=>r.method+' '+r.url)}`);return requestQueue.splice(i,1)[0];};
  const reply=async(request,data,status=200)=>{await act(async()=>request.resolve(new Response(JSON.stringify(data),{status})));};
  const fail=async(request,message)=>{await act(async()=>request.reject(Error(message)));};
  const button=text=>[...document.querySelectorAll('button')].find(el=>el.textContent.trim()===text);
  const click=async(el)=>{assert(el,'Missing UI control');await act(async()=>el.click());};
  const choose=title=>click([...document.querySelectorAll('.submission-list-item')].find(el=>el.querySelector('strong')?.textContent===title));
  const heading=()=>document.querySelector('.submission-detail-heading h2')?.textContent;
  const field=label=>[...document.querySelectorAll('label')].find(el=>[...el.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent).join('').trim()===label)?.querySelector('input,textarea');
  const edit=async(label,value)=>{const el=field(label);assert(el,`Missing field ${label}`);await act(async()=>{const type=el.tagName==='INPUT'?'HTMLInputElement':'HTMLTextAreaElement';Object.getOwnPropertyDescriptor(dom.window[type].prototype,'value').set.call(el,value);el.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});};
  const localAccount={user:undefined,health:{mode:'local'},capabilities:{}};
  const tree=()=>React.createElement(LocaleProvider,null,React.createElement(LanguageSwitch),React.createElement(Submissions,{onPublished(){}}));
  const hydrate=async(ready,rows)=>{
    await reply(next('/api/submissions/capabilities'),{mode:globalThis.__proposalAccountFixture.health.mode,can_submit:true,can_generate:true,can_review:true,generation:{ready,provider:'fixture',model:'fixture'},sandbox:{ready:false},human_review_required:true});
    await reply(next('/api/submissions'),{submissions:rows});
  };
  const start=async({ready=false,rows=[A,B],keepStorage=false,account=localAccount}={})=>{
    if(!keepStorage)dom.window.sessionStorage.clear();
    globalThis.__proposalAccountFixture=account;
    requestQueue.length=0;document.querySelector('#root').innerHTML='';root=createRoot(document.querySelector('#root'));
    await act(async()=>root.render(tree()));
    await hydrate(ready,rows);
  };
  const stop=async()=>{await act(async()=>root.unmount());root=undefined;};
  await start();
  await choose('Proposal A');const oldA=next('/api/submissions/'+A.id);
  await choose('Proposal B');await reply(next('/api/submissions/'+B.id),B);
  await reply(oldA,{...A,title:'Late A'});
  assert.equal(heading(),'Proposal B','A slower previous detail must not replace the current proposal');
  await choose('Proposal B');const oldB=next('/api/submissions/'+B.id);
  await choose('Proposal A');await reply(next('/api/submissions/'+A.id),A);
  await fail(oldB,'Old B connection failed');
  assert.equal(heading(),'Proposal A');assert(!document.querySelector('[role="alert"]'),'A stale error must not attach to another proposal');
  await choose('Proposal B');const hidden=next('/api/submissions/'+B.id);
  await click(button('新提交'));await reply(hidden,B);
  assert(field('论文标题'),'A late detail must not leave the new-proposal form');
  await stop();

  await start();
  await edit('论文标题','Original draft');await edit('论文链接','https://arxiv.org/abs/2002.02126');await edit('练习重点（可选）','Original focus');
  await act(async()=>document.querySelector('form.submission-form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
  const create=next('/api/submissions','POST');assert.equal(JSON.parse(create.body).title,'Original draft');
  await edit('论文标题','New work while saving');await edit('练习重点（可选）','Keep my new focus');
  await reply(create,item('c'.repeat(32),'Original draft'));
  assert.equal(field('论文标题')?.value,'New work while saving','A successful save must preserve a newer draft and keep the authoring view');
  assert.equal(field('练习重点（可选）')?.value,'Keep my new focus');
  assert([...document.querySelectorAll('.submission-list-item strong')].some(el=>el.textContent==='Original draft'),'Saved proposal remains discoverable without taking over new work');
  await click(button('English fixture'));
  assert([...document.querySelectorAll('[role="status"]')].some(el=>el.textContent.includes('Your current view and newer edits were kept.')),'The saved-draft notice follows the current UI language');
  await stop();

  await start();
  await edit('论文标题','Unchanged draft');await edit('论文链接','https://arxiv.org/abs/2002.02126');
  await act(async()=>document.querySelector('form.submission-form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
  await reply(next('/api/submissions','POST'),item('d'.repeat(32),'Unchanged draft'));
  assert.equal(heading(),'Unchanged draft','An unchanged submitted draft should open the saved proposal');
  await click(button('新提交'));assert.equal(field('论文标题').value,'','Clear only the exact draft that was saved');
  await stop();
  await start({ready:true});
  await choose('Proposal A');const beforeMutation=next('/api/submissions/'+A.id);
  const generate=button('生成代码与测试');
  await act(async()=>{generate.click();generate.click();});
  const mutation=next('/api/submissions/'+A.id+'/generate','POST');
  assert(!requestQueue.some(r=>r.method==='POST'),'The same immediate action cannot issue duplicate mutation requests');
  await choose('Proposal B');await reply(next('/api/submissions/'+B.id),B);
  await reply(mutation,{...A,status:'generating'});
  await reply(beforeMutation,A);
  assert.equal(heading(),'Proposal B','Background action completion cannot steal the current selection');
  assert([...document.querySelectorAll('.submission-list-item')].find(el=>el.querySelector('strong')?.textContent==='Proposal A').textContent.includes('生成中'),'A background action still updates its own saved row');
  await stop();

  await start({ready:true});
  await choose('Proposal A');await reply(next('/api/submissions/'+A.id),A);
  await click(document.querySelector('button[aria-label="刷新提交和服务状态"]'));
  await reply(next('/api/submissions/capabilities'),{mode:'local',can_submit:true,can_generate:true,generation:{ready:true},sandbox:{ready:false}});
  const staleList=next('/api/submissions');
  await click(button('生成代码与测试'));
  await reply(next('/api/submissions/'+A.id+'/generate','POST'),{...A,status:'generating'});
  await reply(staleList,{submissions:[A,B]});
  assert(document.querySelector('.submission-list-item.selected').textContent.includes('生成中'),'An old list cannot undo a newer mutation result');
  await stop();

  await start({ready:true});
  await choose('Proposal A');await reply(next('/api/submissions/'+A.id),A);
  await click(button('生成代码与测试'));const pendingGeneration=next('/api/submissions/'+A.id+'/generate','POST');
  await choose('Proposal B');await reply(next('/api/submissions/'+B.id),B);
  await choose('Proposal A');const duringGeneration=next('/api/submissions/'+A.id);
  await reply(pendingGeneration,{...A,status:'generating',updated_at:'2026-10-15T00:00:00.000002Z'});
  await reply(duringGeneration,{...A,updated_at:'2026-10-15T00:00:00.000001Z'});
  assert.equal(document.querySelector('.submission-detail-heading .submission-status').textContent,'生成中','Returning to the same proposal receives its mutation result and rejects earlier detail snapshots');
  await stop();

  await start({ready:true});
  await choose('Proposal A');await reply(next('/api/submissions/'+A.id),A);
  await click(button('生成代码与测试'));const earlyAck=next('/api/submissions/'+A.id+'/generate','POST');
  await choose('Proposal B');await reply(next('/api/submissions/'+B.id),B);
  await choose('Proposal A');await reply(next('/api/submissions/'+A.id),{...A,status:'needs_review',updated_at:'2026-10-15T00:00:00.000003Z'});
  await reply(earlyAck,{...A,status:'generating',updated_at:'2026-10-15T00:00:00.000002Z'});
  assert.equal(document.querySelector('.submission-detail-heading .submission-status').textContent,'待人工评审','An operation acknowledgement cannot regress a newer server snapshot, even within the same millisecond');
  await stop();

  const originalSetTimeout=globalThis.setTimeout,originalClearTimeout=globalThis.clearTimeout;
  const polls=new Map();let timer=0;
  globalThis.setTimeout=(fn,ms,...args)=>{if(ms!==2000)return originalSetTimeout(fn,ms,...args);const id={poll:++timer};polls.set(id,fn);return id;};
  globalThis.clearTimeout=id=>{if(!polls.delete(id))originalClearTimeout(id);};
  try {
    await start({rows:[{...A,status:'generating'},B]});
    await choose('Proposal A');await reply(next('/api/submissions/'+A.id),{...A,status:'generating'});
    assert.equal(polls.size,1);
    const [id,tick]=[...polls][0];polls.delete(id);await act(async()=>tick());
    const oldPoll=next('/api/submissions/'+A.id);
    assert.equal(polls.size,0,'A slow poll must finish before scheduling another poll');
    await choose('Proposal B');await reply(next('/api/submissions/'+B.id),B);
    await choose('Proposal A');await reply(next('/api/submissions/'+A.id),{...A,status:'needs_review'});
    await reply(oldPoll,{...A,status:'generating'});
    assert.equal(document.querySelector('.submission-detail-heading .submission-status').textContent,'待人工评审','Leaving and returning to the same ID must still reject its previous poll');
    assert.equal(polls.size,0);
    await stop();

    await start({ready:true});
    await choose('Proposal A');await reply(next('/api/submissions/'+A.id),A);
    await click(button('生成代码与测试'));const delayedAck=next('/api/submissions/'+A.id+'/generate','POST');
    await choose('Proposal B');await reply(next('/api/submissions/'+B.id),B);
    await choose('Proposal A');await reply(next('/api/submissions/'+A.id),{...A,status:'generating',updated_at:'2026-10-15T00:00:00.000003+00:00'});
    assert.equal(polls.size,1);
    await reply(delayedAck,{...A,status:'generating',updated_at:'2026-10-15T00:00:00.000002Z'});
    assert.equal(polls.size,1,'An older operation acknowledgement must preserve polling for the current in-progress version');
    const [pollId,pollTick]=[...polls][0];polls.delete(pollId);await act(async()=>pollTick());
    await reply(next('/api/submissions/'+A.id),{...A,status:'needs_review',updated_at:'2026-10-15T00:00:00.000004+00:00'});
    assert.equal(document.querySelector('.submission-detail-heading .submission-status').textContent,'待人工评审');
    assert.equal(polls.size,0,'A terminal response stops the sequential poll loop');
    await stop();
  } finally {globalThis.setTimeout=originalSetTimeout;globalThis.clearTimeout=originalClearTimeout;}
  const {proposalDraftKey,proposalDraftOwner,readProposalDraft}=await server.ssrLoadModule('/src/proposalDrafts.ts');
  await start();
  await edit('论文标题','  Restore paper 🧪  ');await edit('论文链接','https://arxiv.org/abs/2002.02126');
  await edit('官方代码链接（可选）','https://github.com/kuandeng/LightGCN');await edit('练习重点（可选）','  exact focus\nnext line  ');
  const excerptToggle=[...document.querySelectorAll('summary')].find(el=>el.textContent.trim()==='补充论文摘录（可选）');await click(excerptToggle);
  const excerptInput=document.querySelector('.submission-form details textarea');
  await act(async()=>{Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype,'value').set.call(excerptInput,'\n```python\nreturn x\n```\n');excerptInput.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
  const draftRecord=readProposalDraft(dom.window.sessionStorage,'local').record;
  assert(draftRecord,'All editable proposal fields must persist before navigation');
  const savedDraft=draftRecord.draft;
  await stop();await start({keepStorage:true});
  assert.equal(field('论文标题').value,savedDraft.title);assert.equal(field('练习重点（可选）').value,savedDraft.focus);assert.equal(document.querySelector('.submission-form details textarea').value,savedDraft.excerpt);
  assert(document.querySelector('.submission-form [role="status"]').textContent.includes('已恢复'));assert(!requestQueue.some(r=>r.method==='POST'),'Restoring a draft never submits it');
  await act(async()=>document.querySelector('form.submission-form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
  const savingDraft=next('/api/submissions','POST');await edit('论文标题','Different edit');await edit('论文标题',savedDraft.title);
  await reply(savingDraft,item('e'.repeat(32),'Restored submission'));
  assert.equal(field('论文标题').value,savedDraft.title);assert(readProposalDraft(dom.window.sessionStorage,'local').record,'An ABA edit stays recoverable after the older source succeeds');
  await stop();await start({keepStorage:true});assert.equal(field('论文标题').value,savedDraft.title);
  await act(async()=>document.querySelector('form.submission-form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
  await reply(next('/api/submissions','POST'),{error:'Temporary API failure'},503);
  await stop();await start({keepStorage:true});assert.equal(field('论文标题').value,savedDraft.title,'A failed submission keeps the tab draft');
  await act(async()=>document.querySelector('form.submission-form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
  const exact=next('/api/submissions','POST');assert.deepEqual(Object.keys(JSON.parse(exact.body)).sort(),['code_url','excerpt','focus','paper_url','title'],'Draft owner/revision metadata never enters the API payload');assert(readProposalDraft(dom.window.sessionStorage,'local').record,'Do not clear before acknowledgement');
  await reply(exact,item('f'.repeat(32),'Acknowledged proposal'));
  assert.equal(readProposalDraft(dom.window.sessionStorage,'local').record,undefined);await stop();await start({keepStorage:true});assert.equal(field('论文标题').value,'','An acknowledged unchanged draft must not reappear');await stop();

  const aliceAccount={user:{id:'alice',role:'member'},health:{mode:'hosted'},capabilities:{}},bobAccount={...aliceAccount,user:{id:'bob',role:'member'}};
  await start({account:aliceAccount});await edit('论文标题','Alice private draft');
  globalThis.__proposalAccountFixture=bobAccount;await act(async()=>root.render(tree()));await hydrate(false,[B]);
  assert.equal(field('论文标题').value,'','Account switches remount before reading another owner’s draft');await edit('论文标题','Bob private draft');
  globalThis.__proposalAccountFixture=aliceAccount;await act(async()=>root.render(tree()));await hydrate(false,[A]);
  assert.equal(field('论文标题').value,'Alice private draft');assert.equal(readProposalDraft(dom.window.sessionStorage,proposalDraftOwner('hosted','bob')).record.draft.title,'Bob private draft');
  await stop();

  await start({account:aliceAccount});await edit('论文标题','Alice pending save');await edit('论文链接','https://arxiv.org/abs/2002.02126');
  await act(async()=>document.querySelector('form.submission-form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
  const oldOwnerRequest=next('/api/submissions','POST');
  globalThis.__proposalAccountFixture=bobAccount;await act(async()=>root.render(tree()));await hydrate(false,[B]);await edit('论文标题','Bob keeps editing');
  await reply(oldOwnerRequest,item('1'.repeat(32),'Alice confirmed elsewhere'));
  assert.equal(field('论文标题').value,'Bob keeps editing');assert.equal(readProposalDraft(dom.window.sessionStorage,proposalDraftOwner('hosted','alice')).record.draft.title,'Alice pending save','An unmounted account’s uncertain result does not clean a restored draft');await stop();

  await start();await edit('论文标题','Saved but cleanup blocked');await edit('论文链接','https://arxiv.org/abs/2002.02126');
  const removeItem=dom.window.Storage.prototype.removeItem;
  dom.window.Storage.prototype.removeItem=function(key){if(key===proposalDraftKey('local'))throw Error('Delete denied');return removeItem.call(this,key);};
  try {
    await act(async()=>document.querySelector('form.submission-form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
    await reply(next('/api/submissions','POST'),item('2'.repeat(32),'Saved but cleanup blocked'));
    await click(button('新提交'));assert(document.querySelector('.submission-form [role="status"]').textContent.includes('仍保留一份草稿'),'A failed cleanup must not claim the old draft was cleared');
    assert(readProposalDraft(dom.window.sessionStorage,'local').record);await stop();
  } finally {dom.window.Storage.prototype.removeItem=removeItem;}

  const storageDescriptor=Object.getOwnPropertyDescriptor(dom.window,'sessionStorage');
  Object.defineProperty(dom.window,'sessionStorage',{configurable:true,get(){throw Error('Storage blocked');}});
  try {
    await start({keepStorage:true});await edit('论文标题','In-memory draft');
    assert.equal(field('论文标题').value,'In-memory draft');assert(document.querySelector('.submission-form [role="status"]').textContent.includes('暂时无法保存'),'Blocked storage must never claim the draft is saved');
    await stop();
  } finally {Object.defineProperty(dom.window,'sessionStorage',storageDescriptor);}
  console.log('PASS proposal requests: latest selection, stale errors, new-form navigation, captured create payload, newer draft preservation, bilingual notices, exact saved-draft clearing, duplicate mutation gates, background updates, stale lists, sub-millisecond server versions, sequential ABA-safe polling, five-field refresh recovery, per-account drafts, failed/confirmed submissions and blocked-storage notices');
} finally {
  if(root)await act(async()=>root.unmount());
  await server.close();dom.window.close();
  for(const [key,descriptor] of Object.entries(previous)){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
