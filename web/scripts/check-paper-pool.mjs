import assert from 'node:assert/strict';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom = new JSDOM('<!doctype html><div id="root"></div>', {url:'http://127.0.0.1/#/submit'});
Object.defineProperty(dom.window.navigator, 'languages', {value:['en-US'], configurable:true});
const keys = ['window','document','location','sessionStorage','navigator','HTMLElement','MutationObserver','IS_REACT_ACT_ENVIRONMENT','fetch','__poolAccount'];
const previous = Object.fromEntries(keys.map(key => [key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for (const key of keys.slice(0,8)) Object.defineProperty(globalThis,key,{value:dom.window[key],configurable:true,writable:true});
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const React = await import('react');
const {createRoot} = await import('react-dom/client');
const {act} = React;
const server = await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
  name:'paper-pool-fixture',enforce:'pre',
  resolveId(source,importer) {
    if (source === './social' && /\/(?:PaperPool|Submissions)\.tsx$/.test(importer || '')) return '\0pool-account';
    if (source === './deferredFeature' && importer?.endsWith('/src/PaperPool.tsx')) return '\0pool-editor';
  },
  load(id) {
    if (id === '\0pool-account') return `import React from 'react'; export const useAccount=()=>globalThis.__poolAccount; export const errorMessage=e=>e.message; export const profileHref=u=>'#/u/'+u.username; export const Avatar=()=>React.createElement('span');`;
    if (id === '\0pool-editor') return `import React from 'react'; export const deferredFeature=()=>props=>React.createElement('textarea',{'aria-label':props.label,value:props.value,onChange:e=>props.onChange(e.target.value)});`;
  },
}]});
let root;
const requests=[];
const caps={can_submit:true,automation_enabled:true,generation:{ready:true},attempts:{ready:true}};
const alice={user:{id:'a'.repeat(32)},health:{mode:'hosted'}}, bob={user:{id:'b'.repeat(32)},health:{mode:'hosted'}};
const makeProposal=(id='c'.repeat(32),patch={})=>({id,arxiv_id:'2002.02126',arxiv_url:'https://arxiv.org/abs/2002.02126',status:'pool',created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',version:1,problem:{title:'Loss function',title_en:'Loss function',statement:'Return a scalar loss.',starter_code:'def loss(x):\n    pass\n',functions:[{name:'loss',signature:'loss(x)',returns:'A scalar float.'}]},review_summary:{approve:0,needs_changes:0,total:0,minimum_reviews:5,approval_ratio:.8,review_hours:48},viewer:{can_review:false,has_attempted:false,reason:'Try this version first.'},...patch});
const A=makeProposal(), B=makeProposal('d'.repeat(32),{problem:{title:'Other candidate',statement:'Return x.',starter_code:'return x'}});
const next=(url,method='GET')=>{const index=requests.findIndex(request=>request.url===url&&request.method===method);assert(index>=0,`Missing ${method} ${url}; queued ${requests.map(r=>r.method+' '+r.url)}`);return requests.splice(index,1)[0];};
const reply=async(request,data,status=200)=>{await act(async()=>request.resolve(new Response(JSON.stringify(data),{status})));};
const button=text=>[...document.querySelectorAll('button')].find(el=>el.textContent.trim()===text);
const click=async(el)=>{assert(el,'Missing control');await act(async()=>el.click());};
const edit=async(selector,value)=>{const el=document.querySelector(selector);assert(el,`Missing field ${selector}`);await act(async()=>{const type=el.tagName==='INPUT'?'HTMLInputElement':'HTMLTextAreaElement';Object.getOwnPropertyDescriptor(dom.window[type].prototype,'value').set.call(el,value);el.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});};
const submit=()=>act(async()=>document.querySelector('.pool-submission-form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
const navigate=async(hash)=>{await act(async()=>{dom.window.history.replaceState({},'',hash);dom.window.dispatchEvent(new dom.window.HashChangeEvent('hashchange'));});};
const listReply=()=>reply(next('/api/paper-pool'),{items:[A,B],capabilities:caps});
try {
  const {default:Submissions}=await server.ssrLoadModule('/src/Submissions.tsx');
  const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
  const {arxivPaperUrl,poolRoute}=await server.ssrLoadModule('/src/paperPoolApi.ts');
  const {signInHref,accountReturnHref}=await server.ssrLoadModule('/src/accountNavigation.ts');
  assert.equal(accountReturnHref(decodeURIComponent(signInHref(`submit?proposal=${A.id}`).slice(2))),`#/submit?proposal=${A.id}`,'Signing in returns to the exact candidate');
  assert.equal(signInHref('submit?proposal=evil&redirect=https://example.com'),'#/account');
  assert.equal(arxivPaperUrl('https://arxiv.org/pdf/2002.02126v3.pdf'),'https://arxiv.org/abs/2002.02126v3');
  assert.equal(arxivPaperUrl('https://export.arxiv.org/abs/2002.02126v2'),'https://arxiv.org/abs/2002.02126v2');
  assert.equal(arxivPaperUrl('http://www.arxiv.org/abs/hep-th/9901001v2'),'https://arxiv.org/abs/hep-th/9901001v2');
  for (const invalid of ['https://arxiv.org.evil.test/abs/2002.02126','https://u@arxiv.org/abs/2002.02126','https://arxiv.org:443/pdf/2002.02126','https://arxiv.org:4000/pdf/2002.02126','https://arxiv.org/html/2002.02126','https://arxiv.org/abs/2013.02126','https://arxiv.org/abs/2002.02126v0','https://arxiv.org/pdf/2002.02126v3.pdf?download=1','https://arxiv.org/abs/2002.02126#section','https://arxiv.org/abs/2002.02126.pdf','https://arxiv.org/abs/2002.02126/','https://arxiv.org/ABS/2002.02126','javascript:alert(1)']) assert.equal(arxivPaperUrl(invalid),undefined,invalid);
  assert.equal(poolRoute('#/submit?proposal=../secret').id,undefined);
  globalThis.fetch=(url,options={})=>new Promise((resolve,reject)=>requests.push({url,method:options.method||'GET',body:options.body,signal:options.signal,resolve,reject}));
  const tree=()=>React.createElement(LocaleProvider,null,React.createElement(Submissions,{onPublished(){}}));
  async function start({hash='#/submit',account=alice,keepStorage=false,capabilities=caps,items=[A,B],total=items.length}={}) {
    if(!keepStorage)sessionStorage.clear();
    requests.length=0;globalThis.__poolAccount=account;
    dom.window.history.replaceState({},'',hash);
    root=createRoot(document.querySelector('#root'));
    await act(async()=>root.render(tree()));
    await reply(next('/api/paper-pool'),{items,total,capabilities});
  }
  async function stop(){await act(async()=>root.unmount());root=undefined;}
  await start();
  assert.equal(document.querySelectorAll('.pool-submission-form input').length,1,'Hosted submission only asks for arXiv plus optional guidance');
  assert.equal(document.querySelectorAll('.pool-submission-form textarea').length,1);
  await edit('.pool-submission-form input','https://example.com/paper');await submit();
  assert(!requests.some(r=>r.method==='POST'));assert(document.querySelector('[role=alert]').textContent.includes('arXiv'));
  await edit('.pool-submission-form input','https://arxiv.org/pdf/2002.02126v3.pdf');
  await edit('.pool-submission-form textarea','  Focus on loss stability  ');
  await act(async()=>{document.querySelector('.pool-submission-form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true}));document.querySelector('.pool-submission-form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true}));});
  const create=next('/api/paper-pool','POST');assert.deepEqual(JSON.parse(create.body),{arxiv_url:A.arxiv_url+'v3',prompt:'Focus on loss stability',expected_user_id:alice.user.id});
  assert(!requests.some(r=>r.method==='POST'),'Double submit is gated synchronously');
  await reply(create,{proposal:A,duplicate:{kind:'proposal',proposal:A}});await listReply();
  assert(document.querySelector(`.pool-feedback a[href="#/submit?proposal=${A.id}"]`));assert.equal(document.querySelector('.pool-submission-form textarea').value,'  Focus on loss stability  ','Duplicates preserve guidance for revision');
  await submit();await reply(next('/api/paper-pool','POST'),{proposal:null,duplicate:{kind:'catalog',problems:[{id:'alignment/dpo',title:'DPO'}]}});await listReply();assert(document.querySelector('.pool-feedback a[href="#/alignment/dpo"]'));
  await submit();const pending=next('/api/paper-pool','POST');await edit('.pool-submission-form textarea','New guidance');await reply(pending,{proposal:A});await listReply();
  assert.equal(document.querySelector('.pool-submission-form textarea').value,'New guidance','A late save cannot clear newer edits');
  await submit();await reply(next('/api/paper-pool','POST'),{error:'Daily limit reached'},429);assert.equal(document.querySelector('.pool-submission-form textarea').value,'New guidance');
  await stop();await start({keepStorage:true});assert.equal(document.querySelector('.pool-submission-form textarea').value,'New guidance','Refresh restores owner-scoped guidance');
  await submit();const oldOwner=next('/api/paper-pool','POST');globalThis.__poolAccount=bob;await act(async()=>root.render(tree()));await listReply();
  assert.equal(document.querySelector('.pool-submission-form textarea').value,'');await edit('.pool-submission-form textarea','Bob draft');await reply(oldOwner,{proposal:A});assert.equal(document.querySelector('.pool-submission-form textarea').value,'Bob draft');assert(!document.querySelector('.pool-feedback'));
  await stop();

  await start({capabilities:{...caps,automation_enabled:false,generation:{ready:false,reason:'Worker not configured'}}});
  assert(document.querySelector('.pool-service-notice').textContent.includes('Automatic generation is not ready'));assert(!document.querySelector('.pool-service-notice').textContent.includes('Worker not configured'),'Availability uses localized product wording');assert(!button('Submit paper').disabled,'Proposals can be retained while automation is explicitly unavailable');await stop();

  const firstPage=Array.from({length:20},(_,index)=>makeProposal((index+1).toString(16).padStart(32,'0')));
  await start({hash:'#/submit?view=pool',items:firstPage,total:21});
  assert.equal(document.querySelectorAll('.pool-card').length,20);assert(button('Previous page').disabled);
  await act(async()=>{button('Next page').click();button('Next page').click();});
  const page2=next('/api/paper-pool?limit=20&offset=20');assert(!requests.some(r=>r.url.includes('offset=')),'Repeated next clicks issue only one page request');
  await reply(page2,{items:[B],total:21,capabilities:caps});
  assert.equal(document.querySelectorAll('.pool-card').length,1);assert(document.querySelector(`.pool-card[href="#/submit?proposal=${B.id}"]`),'Older candidates beyond the default 20 remain accessible');
  assert(document.querySelector('.pool-pagination').textContent.includes('21–21 / 21'));assert(button('Next page').disabled);
  await click(document.querySelector('button[aria-label="Refresh pool"]'));await reply(next('/api/paper-pool?limit=20&offset=20'),{items:[B],total:21,capabilities:caps});
  await click(button('Previous page'));await reply(next('/api/paper-pool'),{items:firstPage,total:21,capabilities:caps});assert.equal(document.querySelectorAll('.pool-card').length,20);
  await click(button('Next page'));await reply(next('/api/paper-pool?limit=20&offset=20'),{error:'Page temporarily unavailable'},503);assert(document.querySelector('[role=alert]').textContent.includes('Page temporarily unavailable'));
  await click(button('Retry'));await reply(next('/api/paper-pool?limit=20&offset=20'),{items:[B],total:21,capabilities:caps});assert.equal(document.querySelectorAll('.pool-card').length,1,'Retry preserves the failed page offset');await stop();

  await start({hash:`#/submit?proposal=${A.id}`});const stale=next('/api/paper-pool/'+A.id);await navigate(`#/submit?proposal=${B.id}`);await reply(next('/api/paper-pool/'+B.id),B);await reply(stale,A);
  assert.equal(document.querySelector('.pool-detail-heading h2').textContent,'Other candidate','Late details never replace another candidate');await stop();

  await start({hash:`#/submit?proposal=${A.id}`});await reply(next('/api/paper-pool/'+A.id),A);
  assert(button('Submit review').disabled,'A non-attempted version cannot be reviewed');
  await edit('textarea[aria-label="Trial code"]','def loss(x):\n    return 0.0');await click(button('Run trial tests'));
  const trial=next('/api/paper-pool/'+A.id+'/attempts','POST');assert.deepEqual(JSON.parse(trial.body),{version:1,code:'def loss(x):\n    return 0.0',expected_user_id:alice.user.id});
  const queued={id:'e'.repeat(32),version:1,status:'queued'};
  await reply(trial,{attempt:queued,proposal:A},202);await listReply();
  assert(button('Queued…').disabled);assert(button('Submit review').disabled,'Queue admission is not a completed trial');
  const firstPoll=next('/api/paper-pool/'+A.id+'/attempts/'+queued.id);
  await edit('textarea[aria-label="Trial code"]','def loss(x):\n    return 1.0');
  await stop();assert(firstPoll.signal.aborted,'Unmount aborts the pending trial poll');
  await start({hash:`#/submit?proposal=${A.id}`,keepStorage:true});await reply(next('/api/paper-pool/'+A.id),A);
  const restored=next('/api/paper-pool/'+A.id+'/attempts/'+queued.id);
  assert.equal(document.querySelector('textarea[aria-label="Trial code"]').value,'def loss(x):\n    return 1.0','The candidate version retains its local draft');
  const reviewed={...A,updated_at:'2026-10-15T00:00:01Z',viewer:{can_review:true,has_attempted:true,reason:'',eligibility_reason:null}};
  await reply(restored,{attempt:{...queued,status:'succeeded',report:{passed:false,tests:[{id:'test_loss',passed:false,message:'Expected 1.0'}]}},proposal:reviewed});await listReply();
  assert(document.querySelector('.pool-trial-result').textContent.includes('Tests did not pass'));assert(document.querySelector('.pool-trial-result').textContent.includes('last code you ran'));
  assert(!button('Submit review').disabled,'A valid failed trial can give useful feedback');
  await click(document.querySelector('input[value=needs_changes]'));
  await act(async()=>document.querySelector('.pool-review-form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
  assert(!requests.some(r=>r.method==='POST'),'Changes requested requires a reason');
  await edit('.pool-review-form textarea','The scalar return requirement is ambiguous.');
  await click(button('Submit review'));
  const vote=next('/api/paper-pool/'+A.id+'/votes','POST');assert.deepEqual(JSON.parse(vote.body),{version:1,vote:'needs_changes',reason:'The scalar return requirement is ambiguous.',expected_user_id:alice.user.id});
  await reply(vote,{...reviewed,updated_at:'2026-10-15T00:00:02Z',review_summary:{...A.review_summary,needs_changes:1,total:1},viewer:{...reviewed.viewer,vote:'needs_changes'}});await listReply();
  assert(button('Update review'));assert(document.querySelector('.pool-review').textContent.includes('Review saved.'));
  await stop();

  await start({hash:`#/submit?proposal=${A.id}`,capabilities:{...caps,attempts:{ready:false,reason:'Sandbox unavailable'}}});await reply(next('/api/paper-pool/'+A.id),{...A,viewer:{can_review:false,has_attempted:true,reason:'Authors cannot review their own candidates.'}});
  assert(button('Run trial tests').disabled);assert(button('Submit review').disabled);assert(document.querySelector('.pool-code-panel').textContent.includes('Sandbox unavailable'));
  assert(!document.querySelector('.pool-moderation'),'Member controls never expose maintainer actions');
  await stop();
  await start({hash:`#/submit?proposal=${A.id}`});await reply(next('/api/paper-pool/'+A.id),{...A,viewer:{can_review:true,has_attempted:true,can_manage:true,vote:'needs_changes',reason:'Keep my prior review reason.',eligibility_reason:null}});
  assert.equal(document.querySelector('.pool-review-form textarea').value,'Keep my prior review reason.','Reloading a review retains its reason');
  const moderation=document.querySelector('.pool-moderation form');assert(moderation);
  await act(async()=>moderation.dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
  assert(!requests.some(r=>r.method==='POST'),'Maintainer action requires an explanation');
  await edit('.pool-moderation textarea','Fix the reference return type before opening a new version.');
  await act(async()=>moderation.dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
  const moderationRequest=next('/api/paper-pool/'+A.id+'/moderate','POST');assert.deepEqual(JSON.parse(moderationRequest.body),{action:'request_changes',reason:'Fix the reference return type before opening a new version.',expected_user_id:alice.user.id});
  await reply(moderationRequest,{error:'Permission changed'},403);assert(document.querySelector('.pool-moderation [role=alert]').textContent.includes('Permission changed'));
  assert(![...document.querySelectorAll('.pool-moderation button')].some(el=>/publish/i.test(el.textContent)),'There is no shortcut around catalog publication review');
  await stop();
  await start({account:{user:null,health:{mode:'hosted'}}});assert(document.querySelector('.pool-submission-form a.button').textContent.includes('Sign in'));assert(document.querySelector('.pool-submission-form input').disabled);await stop();
  console.log('PASS paper pool: hosted arXiv-only flow, canonicalization, duplicate links, owner-scoped draft recovery, stale account/detail rejection, real capability gates, queued trial recovery, failed-trial review eligibility, version-bound reasoned votes and own-author restrictions');
} finally {
  if(root)await act(async()=>root.unmount());
  await server.close();dom.window.close();
  for(const key of keys){if(previous[key])Object.defineProperty(globalThis,key,previous[key]);else delete globalThis[key];}
}
