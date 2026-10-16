import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/'});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__writingFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
let guideLoads=0;
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
 name:'writing-fixture',enforce:'pre',
 resolveId(source,importer){if(source==='./social')return '\0writing-social';if(source==='./InterviewWritingGuide'&&importer?.endsWith('/InterviewComposer.tsx'))return '\0writing-guide';},
 load(id){
  if(id==='\0writing-social')return `import React from 'react';export const useAccount=()=>globalThis.__writingFixture.account;export const socialApi=(...args)=>globalThis.__writingFixture.request(...args);export const errorMessage=e=>e.message;`;
  if(id==='\0writing-guide'){guideLoads++;return `import React from 'react';import Guide from '/src/InterviewWritingGuide.tsx';export default function FixtureGuide(props){globalThis.__writingFixture.guideProps=props;if(globalThis.__writingFixture.failGuide)throw Error('Writing guide fixture failure');return React.createElement(Guide,props);}`;}
 }
}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Composer}=await server.ssrLoadModule('/src/InterviewComposer.tsx');
 const model=await server.ssrLoadModule('/src/interviewWriting.ts');
 const drafts=await server.ssrLoadModule('/src/interviewDrafts.ts');
 const receipts=await server.ssrLoadModule('/src/publicationRequests.ts');
 assert.equal(guideLoads,0,'Opening the composer does not load the optional prompt panel');
 assert.equal(Object.keys(model.interviewOutlines).length,5);
 for(const id of Object.keys(model.interviewOutlines))for(const locale of ['zh','en']){
  const outline=model.interviewOutlineText(id,locale);
  assert.equal((outline.match(/^## /gm)||[]).length,4);
  assert.equal(model.hasInterviewDraftContent(outline),false);
  assert.equal(model.hasInterviewDraftContent(outline+'Actual experience 😀'),true);
 }
 assert.equal(model.hasInterviewDraftContent('## 面试流程\n\n## 考察重点与我的思路\n\n## 复盘与准备建议\n'),false,'Legacy outlines remain recognizable');
 assert.equal(model.hasInterviewDraftContent('## My actual interview used a ranking example'),true,'Free-form headings are valid writing');
 assert.equal(model.insertInterviewOutline('x'.repeat(20000),'system','en'),undefined,'Length failures cannot truncate writing');
 const unicode=model.insertInterviewOutline('a😀b','technical','en',2);assert(unicode.after.startsWith('a😀\n\n## '),'Insertion cannot split a surrogate pair');assert(unicode.after.endsWith('b'));
 assert.equal(model.defaultInterviewOutline('assessment'),'technical');assert.equal(model.defaultInterviewOutline('research'),'research');assert.equal(model.defaultInterviewOutline('system-design'),'system');assert.equal(model.defaultInterviewOutline('behavioral'),'behavioral');
 const requests=[];
 globalThis.__writingFixture={account:{user:{id:'writer-alice',can_publish_community:true}},request(path,body){return new Promise((resolve,reject)=>requests.push({path,body,resolve,reject}));}};
 let owner='writer-alice',published=[],closed=0;
 function LocaleButtons(){const {setLocale}=useI18n();return React.createElement(React.Fragment,null,React.createElement('button',{className:'outside',onClick:()=>setLocale('zh')},'Chinese'),React.createElement('button',{onClick:()=>setLocale('en')},'English'));}
 const render=async()=>act(async()=>{if(!root)root=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(LocaleButtons),React.createElement(Composer,{key:owner,owner,company:'google',problems:[],onClose:()=>closed++,onPublished:post=>published.push(post)})));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const button=text=>[...document.querySelectorAll('button')].find(element=>element.textContent.trim()===text);
 const click=async element=>{assert(element,'Missing control');await act(async()=>element.click());};
 const edit=async(element,value)=>{assert(element,'Missing field');await act(async()=>{const type=element.tagName==='SELECT'?'HTMLSelectElement':element.tagName==='TEXTAREA'?'HTMLTextAreaElement':'HTMLInputElement';Object.getOwnPropertyDescriptor(dom.window[type].prototype,'value').set.call(element,value);element.dispatchEvent(new dom.window.Event(element.tagName==='SELECT'?'change':'input',{bubbles:true}));});};
 const field=label=>[...document.querySelectorAll('label')].find(element=>[...element.childNodes].filter(node=>node.nodeType===3).map(node=>node.textContent).join('').trim()===label)?.querySelector('input,select,textarea');
 const body=()=>document.querySelector('.interview-composer textarea'),publishButton=()=>document.querySelector('.form-footer > .button');
 const settle=async predicate=>{for(let attempt=0;attempt<80&&!predicate();attempt++)await act(async()=>new Promise(resolve=>setTimeout(resolve,20)));assert(predicate(),'Writing UI did not settle');};
 const openGuide=async()=>{await act(async()=>{const details=document.querySelector('.interview-writing-guide');details.open=true;details.dispatchEvent(new dom.window.Event('toggle'));});await settle(()=>document.querySelector('.interview-writing-panel'));};
 const submit=async()=>act(async()=>document.querySelector('.interview-composer form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
 await render();assert.equal(guideLoads,0);await edit(field('Title'),'My actual experience');const beforeOpen=drafts.readInterviewDraft(window.sessionStorage,owner);
 await openGuide();assert.equal(guideLoads,1);assert.deepEqual(drafts.readInterviewDraft(window.sessionStorage,owner),beforeOpen,'Reading prompts never edits or publishes a draft');assert.equal(requests.length,0);
 assert.equal(document.querySelectorAll('.interview-writing-panel ol > li').length,4);
 const choice=()=>document.querySelector('.interview-writing-panel select');
 for(const locale of ['en','zh']){
  await click(button(locale==='en'?'English':'Chinese'));
  for(const id of Object.keys(model.interviewOutlines)){
   await edit(choice(),id);const items=[...document.querySelectorAll('.interview-writing-panel ol > li')];assert.equal(items.length,4);
   assert.deepEqual(items.map(item=>item.querySelector('b').textContent),model.interviewOutlines[id].sections.map(section=>section[locale]));
   assert(items.every(item=>item.querySelector('p').textContent.length>12),'Every template section has a real localized writing prompt');
  }
 }
 await click(button('English'));assert.deepEqual(drafts.readInterviewDraft(window.sessionStorage,owner),beforeOpen,'Template and language previews do not rewrite the draft');
 await edit(field('Stage'),'system-design');assert.equal(choice().value,'system');assert.equal(body().value,'');
 await edit(choice(),'behavioral');assert.equal(field('Stage').value,'system-design','An outline choice cannot change interview metadata');
 await edit(field('Stage'),'research');assert.equal(choice().value,'research');
 const tip=document.querySelector('.interview-writing-panel li p').textContent;
 const original='Original notes 😀\nSecond paragraph.';await edit(body(),original);body().focus();body().setSelectionRange(0,8);body().scrollTop=23;
 await click(button('Insert outline at cursor'));const inserted=body().value;
 assert(inserted.startsWith(original.slice(0,8)));assert(inserted.endsWith(original.slice(8)),'Selected writing is preserved rather than replaced');assert(!inserted.includes(tip),'Prompts must never become purported interview answers');assert.equal(document.activeElement,body());
 await click(button('Undo outline insertion'));assert.equal(body().value,original);assert.equal(body().selectionStart,0);assert.equal(body().selectionEnd,8);assert.equal(body().scrollTop,23);assert.equal(field('Stage').value,'research');
 await click(button('Insert outline at cursor'));await edit(body(),body().value+'\nMy later reflection');const later=body().value;
 assert(button('Undo outline insertion').disabled);await click(button('Undo outline insertion'));assert.equal(body().value,later);assert(document.body.textContent.includes('Your text changed after insertion'));
 const mountedBody=body();await click(button('Preview'));assert.equal(body(),mountedBody);assert(body().closest('label').hidden);await click(button('Insert outline at cursor'));assert(!body().closest('label').hidden);assert(body().value.startsWith(later));assert.equal(document.activeElement,body());
 const translatedBody=body().value;await click(button('Chinese'));assert.equal(body().value,translatedBody);assert(document.querySelector('.interview-writing-panel').textContent.includes('实验'));await click(button('English'));assert.equal(body().value,translatedBody,'Locale changes never translate existing writing');
 await stop();await render();assert.equal(body().value,translatedBody);assert(!document.querySelector('.interview-writing-panel'),'Recovered drafts do not open optional prompts automatically');
 await openGuide();const pendingFocus=document.querySelector('.interview-writing-actions .button');
 await act(async()=>{pendingFocus.click();document.querySelector('.outside').focus();});assert.equal(document.activeElement,document.querySelector('.outside'),'Late insertion focus never steals a newer external focus');
 const aliceBody=body().value,staleProps=globalThis.__writingFixture.guideProps;
 owner='writer-bob';globalThis.__writingFixture.account={user:{id:owner,can_publish_community:true}};await render();assert.equal(body().value,'');
 await act(async()=>staleProps.onInsert('technical',pendingFocus));assert.equal(drafts.readInterviewDraft(window.sessionStorage,'writer-alice').body,aliceBody);assert.equal(body().value,'');assert.equal(requests.length,0);
 owner='writer-blank';globalThis.__writingFixture.account={user:{id:owner,can_publish_community:true}};await render();await edit(field('Title'),'A future reflection');await openGuide();await click(button('Insert outline'));
 assert(publishButton().disabled);assert(document.body.textContent.includes('Your outline is ready'));await submit();assert.equal(requests.length,0,'A manual form submission cannot publish a blank scaffold');
 const blank=body().value;await stop();await render();assert.equal(body().value,blank);assert(publishButton().disabled);
 await edit(body(),blank+'I discussed an approach and its limitations.');await submit();assert.equal(requests.length,1);const first=requests[0];assert.equal(first.body.expected_user_id,owner);assert.equal(first.path,'community/posts');assert(publishButton().disabled);
 await openGuide();assert(document.querySelector('.interview-writing-panel select').disabled);assert([...document.querySelectorAll('.interview-writing-actions button')].every(element=>element.disabled));
 const submittedBody=body().value;await act(async()=>globalThis.__writingFixture.guideProps.onInsert('system',document.querySelector('.interview-writing-actions .button')));assert.equal(body().value,submittedBody,'Even a retained callback cannot edit an in-flight publication');
 await act(async()=>first.reject(Error('Lost confirmation')));await openGuide();await click(button('Insert outline at cursor'));const newer=body().value;
 await click(button('Retry previous publication'));assert.equal(requests.length,2);assert.equal(requests[1].body.request_id,first.body.request_id);assert.equal(requests[1].body.body,first.body.body,'Retry confirms the captured payload, not the newer outline edit');
 await act(async()=>requests[1].resolve({id:'a'.repeat(32),author:{id:owner},interview:first.body.interview}));assert.equal(body().value,newer);assert.equal(published.length,0,'Acknowledging an older publication cannot discard newer writing');
 // Old versions allowed empty outlines. Their pending receipts must remain confirmable.
 owner='writer-legacy';globalThis.__writingFixture.account={user:{id:owner,can_publish_community:true}};
 const legacy={title:'Legacy pending report',body:model.interviewOutlineText('technical','en'),interview:{company:'google',role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:[]}};
 drafts.writeInterviewDraft(window.sessionStorage,owner,legacy);const receipt=receipts.preparePublication(window.sessionStorage,owner,'interview','general',drafts.interviewPublication(legacy));await render();assert.equal(publishButton().disabled,false);
 await submit();assert.equal(requests.at(-1).body.request_id,receipt.receipt.requestId);await act(async()=>requests.at(-1).resolve({id:'b'.repeat(32),author:{id:owner},interview:legacy.interview}));assert.equal(published.length,1);
 // Optional prompt failures leave the user's form and exact draft mounted.
 owner='writer-recovery';globalThis.__writingFixture.account={user:{id:owner,can_publish_community:true}};await render();await edit(field('Title'),'Keep this draft');await edit(body(),'Actual notes that must survive.');const fieldBefore=body(),savedBefore=fieldBefore.value;
 globalThis.__writingFixture.failGuide=true;const previousError=console.error;console.error=()=>{};
 try{await act(async()=>{const details=document.querySelector('.interview-writing-guide');details.open=true;details.dispatchEvent(new dom.window.Event('toggle'));});await settle(()=>document.querySelector('.interview-writing-failure'));}finally{console.error=previousError;}
 assert.equal(body(),fieldBefore);assert.equal(body().value,savedBefore);assert(!button('Reload page'));
 globalThis.__writingFixture.failGuide=false;await click(button('Retry writing prompts'));await settle(()=>document.querySelector('.interview-writing-panel'));assert.equal(document.activeElement,document.querySelector('.interview-writing-guide > summary'));assert.equal(body(),fieldBefore);
 await edit(body(),'x'.repeat(20000));const fullDraft=drafts.readInterviewDraft(window.sessionStorage,owner);await click(button('Insert outline at cursor'));assert.equal(body().value,fullDraft.body);assert.deepEqual(drafts.readInterviewDraft(window.sessionStorage,owner),fullDraft);assert(document.body.textContent.includes('near its length limit'),'A rejected insert leaves the exact draft recoverable');
 console.log('PASS interview writing: five bilingual stage-aware outlines, lazy prompt loading, explicit inserts, Unicode/length safety, exact selection/scroll undo, later-edit protection, draft/owner isolation, scaffold publication guard, legacy receipt recovery, captured retries and local failure recovery');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
 for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,{...descriptor});else delete globalThis[key];}
}
