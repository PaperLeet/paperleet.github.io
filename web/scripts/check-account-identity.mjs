import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

const dom = new JSDOM('<div id="root"></div>', { url: 'https://paperleet.test/#/account' });
const names = ['window','document','location','localStorage','IS_REACT_ACT_ENVIRONMENT','__identityFixture'];
const originals = new Map(names.map(name => [name,Object.getOwnPropertyDescriptor(globalThis,name)]));
Object.assign(globalThis,{window:dom.window,document:dom.window.document,location:dom.window.location,localStorage:dom.window.localStorage,IS_REACT_ACT_ENVIRONMENT:true});
localStorage.setItem('paperleet.locale','en');
const React = await import('react'); const { act } = React; const { createRoot } = await import('react-dom/client');
const server = await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
  name:'identity-fixture',enforce:'pre',resolveId(source,importer){if(source==='./social' && importer?.endsWith('/AccountIdentitySettings.tsx'))return '\0identity-fixture';},
  load(id){if(id==='\0identity-fixture')return `import React from 'react';export const useAccount=()=>globalThis.__identityFixture;export const socialApi=(...args)=>globalThis.__identityFixture.api(...args);export const errorMessage=e=>e.message;export const SocialNotice=({error})=>error?React.createElement('p',{role:'alert'},error):null;`;}
}]});
let root;
try {
  const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
  const {default:Settings}=await server.ssrLoadModule('/src/AccountIdentitySettings.tsx');
  const {default:History}=await server.ssrLoadModule('/src/AffiliationHistory.tsx');
  const school={id:'school-login',email:'private@school.edu',kind:'google',verified:true,enabled:true,is_primary:true};
  const record={id:'affiliation-1',name:'School',name_en:'School',domain:'school.edu',kind:'institution',position:'Student',start_date:'2024-01-01',end_date:'',status:'current',is_primary:true,verified_at:'2026-01-01T00:00:00Z',revision:1,email:'private@school.edu'};
  let user={id:'owner',email:school.email,login_methods:[school],affiliations:[record]};
  let result={challenge:'binding-challenge',retry_after:60},failure,late;
  const requests=[],updates=[];
  const fixture=globalThis.__identityFixture={capabilities:{account_identity_management:true,login_email_binding_available:true},
    setUser(next){updates.push(next);user=next;render();},async api(path,payload){requests.push({path,payload});if(failure)throw failure;if(late)return new Promise(resolve=>{late.resolve=resolve;});return result;}};
  function LocaleControl(){fixture.setLocale=useI18n().setLocale;return null;}
  const container=document.querySelector('#root');root=createRoot(container);
  function render(history=false){root.render(React.createElement(LocaleProvider,null,React.createElement(LocaleControl),React.createElement(history?History:Settings,history?{affiliations:user.affiliations}:{user})));}
  const text=()=>container.textContent;
  const findButton=(name,scope=container)=>[...scope.querySelectorAll('button')].find(button=>button.textContent===name);
  async function click(button){assert(button,'Expected action');await act(async()=>button.click());}
  async function input(selector,value){const element=container.querySelector(selector);assert(element,selector);const props=element[Object.keys(element).find(key=>key.startsWith('__reactProps$'))];await act(async()=>props.onChange({target:{value,checked:value}}));}
  async function submit(selector='form'){const form=container.querySelector(selector);await act(async()=>form.dispatchEvent(new window.Event('submit',{bubbles:true,cancelable:true})));}
  await act(async()=>render());
  assert.match(text(),/permanent personal email/);assert(!findButton('Disable sign-in'),'Primary method cannot be disabled');
  await input('input[type=email]','Personal@Gmail.com');await submit();
  assert.deepEqual(requests.at(-1),{path:'account/login-emails/send-code',payload:{email:'personal@gmail.com',expected_user_id:'owner'}});
  assert(container.querySelector('input[type=email]').disabled);assert([...container.querySelectorAll('button')].some(b=>b.textContent.includes('Resend in')&&b.disabled));
  await input('input[inputmode=numeric]','123456');
  const personal={...school,id:'personal-login',email:'personal@gmail.com',kind:'email',is_primary:false};
  result={user:{...user,login_methods:[school,personal]}};await submit();
  assert.deepEqual(requests.at(-1).payload,{email:'personal@gmail.com',code:'123456',challenge:'binding-challenge',expected_user_id:'owner'});
  assert.match(text(),/Login email linked/);
  result={user:{...user,email:personal.email,login_methods:[{...personal,is_primary:true},{...school,is_primary:false}]}};
  await click(findButton('Make primary'));
  assert.deepEqual(requests.at(-1),{path:'account/login-methods/change',payload:{id:'personal-login',action:'primary',expected_user_id:'owner'}});
  const count=requests.length;await click(findButton('Disable sign-in'));assert.equal(requests.length,count);assert.match(text(),/Affiliation history and account data remain/);
  await click(findButton('Cancel'));assert(!findButton('Confirm disable'));
  await click(findButton('Disable sign-in'));result={user:{...user,login_methods:user.login_methods.map(m=>m.id===school.id?{...m,enabled:false}:m)}};await click(findButton('Confirm disable'));
  assert.deepEqual(requests.at(-1).payload,{id:'school-login',action:'disable',expected_user_id:'owner'});
  assert.match(text(),/Login method disabled/);assert(!findButton('Disable sign-in'));
  await input('.affiliation-editor select','past');
  assert(!container.querySelector('.affiliation-editor input[type=checkbox]'));
  await input('.affiliation-editor input[type=date]:last-of-type','2025-01-01');
  failure=Error('This affiliation changed. Refresh before editing it again.');await submit('.affiliation-editor form');
  assert.equal(requests.at(-1).path,'account/affiliations/update');assert.equal(requests.at(-1).payload.is_primary,false);assert.equal(requests.at(-1).payload.status,'past');assert.equal(requests.at(-1).payload.revision,1);
  assert.match(text(),/This affiliation changed/);assert.equal(container.querySelector('.affiliation-editor select').value,'past');
  await act(async()=>fixture.setLocale('zh'));assert.match(text(),/机构与经历/);assert.equal(container.querySelector('.affiliation-editor select').value,'past');
  // Old-account responses must never replace a newer account after a switch.
  failure=undefined;late={};await input('input[type=email]','late@gmail.com');await submit();
  const before=updates.length;user={...user,id:'new-owner',email:'new@school.edu',login_methods:[],affiliations:[]};await act(async()=>render());
  await act(async()=>late.resolve({challenge:'old-account',retry_after:60}));assert.equal(updates.length,before);assert(!container.querySelector('input[inputmode=numeric]'));
  late=undefined;result={challenge:'next-binding',retry_after:60};await input('input[type=email]','next@gmail.com');await submit();await input('input[inputmode=numeric]','123456');
  const oldOwner=user;late={};await submit();user={...user,id:'third-owner'};await act(async()=>render());
  await act(async()=>late.resolve({user:{...oldOwner,email:'next@gmail.com'}}));assert.equal(updates.length,before);assert(!text().includes('next@gmail.com'));
  // The public history component receives no reason to display the private proof address.
  user={...user,affiliations:[{...record,name:'brown.edu',name_en:undefined,domain:'brown.edu',status:'past',is_primary:false,end_date:'2026-06-01'}]};await act(async()=>render(true));
  assert.match(text(),/历史机构/);assert(!text().includes(record.email));assert.match(text(),/2026-01-01/);
  assert.match(text(),/布朗大学/);
  console.log('PASS account identity UI: owner-bound email linking, cooldown, explicit primary/disable review, stale history revision and locale continuity, retired-account response isolation and private-proof-free public history');
} finally {
  if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
  for(const[name,descriptor]of originals){if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];}
}
