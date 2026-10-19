import assert from 'node:assert/strict';
import React, {act, useEffect, useState} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

// Component interactions use an isolated DOM; no running browser, account or app is accessed.
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/'});
const globals={window:dom.window,document:dom.window.document,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
const originalFetch=globalThis.fetch;
const originalResize=globalThis.ResizeObserver;
const originalRaf=globalThis.requestAnimationFrame, originalCancel=globalThis.cancelAnimationFrame;
globalThis.requestAnimationFrame=callback=>setTimeout(callback,0);
globalThis.cancelAnimationFrame=clearTimeout;
globalThis.ResizeObserver=class {observe(){}disconnect(){}};
dom.window.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){}});
Object.defineProperties(dom.window.HTMLElement.prototype,{clientWidth:{get:()=>1000},clientHeight:{get:()=>700}});
const user={id:'account-one',username:'researcher',display_name:'Researcher <script>',avatar_color:'#48766b',bio:'',created_at:'2026-10-14',email:'private@example.test',role:'user',show_institution:false,show_country:false,show_progress:true,institution:null,country:''};
let logoutRequests=0,finishLogout;
globalThis.fetch=async path=>{
  if(String(path).endsWith('/api/health'))return Response.json({mode:'hosted',auth_required:true,judge:{ready:true}});
  if(String(path).endsWith('/api/account/me'))return Response.json({user,capabilities:{auth_available:true,institutions:[],community_bookmarks:true}});
  if(String(path).endsWith('/api/account/logout')){logoutRequests++;return new Promise(resolve=>{finishLogout=resolve;});}
  throw Error(`Unexpected test request: ${path}`);
};
const server=await createServer({server:{middlewareMode:true},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify('hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify('0'),'import.meta.env.VITE_API_ORIGIN':JSON.stringify('')}});
let root;
try {
  const {createRoot}=await import('react-dom/client');
  const {AccountProvider,useAccount,AccountNav}=await server.ssrLoadModule('/src/social.tsx');
  const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
  const {default:LayoutMenu}=await server.ssrLoadModule('/src/WorkspaceLayoutMenu.tsx');
  const {default:Split}=await server.ssrLoadModule('/src/ResizableSplit.tsx');
  const h=React.createElement;
  const find=(selector)=>document.querySelector(selector);
  const button=(text)=>[...document.querySelectorAll('button')].find(button=>button.textContent.trim()===text);
  const click=async element=>{assert(element,'Expected an interactive control');await act(async()=>element.dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true})));};
  const settle=async predicate=>{for(let i=0;i<60&&!predicate();i++)await act(async()=>new Promise(resolve=>setTimeout(resolve,20)));assert(predicate(),'Component did not settle');};
  let account,localeControl,backupOpens=0,mounts=0,layoutIntent=()=>{};
  function Draft(){useEffect(()=>{mounts++;},[]);return h('textarea',{'aria-label':'Draft',defaultValue:'private draft αβ'});}
  function Harness(){
    account=useAccount();
    localeControl=useI18n();
    const [theme,setTheme]=useState('light'),[layout,setLayout]=useState(),[collapsed,setCollapsed]=useState(false),[lab,setLab]=useState(false);
    return h(React.Fragment,null,
      h('button',{className:'outside'},'Outside'),
      h(AccountNav,{theme,onTheme:setTheme,returnTo:'22',backupReady:true,onBackup(){backupOpens++;}}),
      h(LayoutMenu,{onLayout:(ratio,reset)=>{setLayout({ratio});if(reset)setCollapsed(false);layoutIntent();}}),
      h('button',{onClick:()=>setCollapsed(value=>!value)},'Toggle results'),
      h('button',{onClick:()=>setLab(value=>!value)},'Toggle lab'),
      h('div',{hidden:lab},h(Split,{axis:'vertical',storageKey:'test.layout',initial:.5,minFirst:100,minSecond:100,label:'Resize',firstId:'code',secondId:'results',layout,secondCollapsed:collapsed},h('div',{id:'code'},h(Draft)),h('div',{id:'results'},h('button',{hidden:collapsed},'Result')))));
  }
  root=createRoot(find('#root'));
  await act(async()=>root.render(h(LocaleProvider,null,h(AccountProvider,null,h(Harness)))));
  await settle(()=>account.user?.id===user.id);
  await settle(()=>find('.account-menu-trigger')?.getAttribute('aria-expanded')==='false');
  assert(!find('.header-signin-cta'),'Signed-in users do not see registration prompts');
  assert(!find('[role="dialog"]'),'The account panel starts closed');
  await click(find('.account-menu-trigger'));
  await settle(()=>document.activeElement===find('.account-menu-panel'));
  assert(find('[role="dialog"]').textContent.includes(user.display_name));
  assert(!find('.account-menu-panel script')&&!find('.account-menu-panel').textContent.includes(user.email),'Display names stay escaped and email stays private');
  assert.equal(find('.account-menu-shortcuts a').getAttribute('href'),'#/u/researcher');
  await click(button('EN'));
  assert.equal(localStorage.getItem('paperleet.locale'),'en');
  assert.equal(find('.account-menu-panel').getAttribute('aria-label'),'Personal menu');
  const savedLink=find('.account-menu-links a[href*="saved=1"]');assert(savedLink&&savedLink.textContent.includes('Saved interviews'));assert.equal(savedLink.getAttribute('href'),'#/interviews?saved=1&days=all');const navigated=new Promise(resolve=>window.addEventListener('hashchange',resolve,{once:true}));await click(savedLink);await act(async()=>navigated);assert(!find('.account-menu-panel'),'The saved-reading shortcut closes the account panel');await click(find('.account-menu-trigger'));
  await click(button('Dark'));
  assert.equal(button('Dark').getAttribute('aria-pressed'),'true');
  assert(find('.account-menu-panel'),'Changing preferences keeps the panel open');
  await act(async()=>find('.account-menu-panel').dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true})));
  assert(!find('.account-menu-panel'));assert.equal(document.activeElement,find('.account-menu-trigger'));
  await click(find('.account-menu-trigger'));
  await act(async()=>find('.outside').dispatchEvent(new dom.window.MouseEvent('pointerdown',{bubbles:true})));
  assert(!find('.account-menu-panel'),'Clicking outside closes the panel');
  await click(find('.account-menu-trigger'));
  // Test keyboard departure after the opening focus transition has completed.
  await settle(()=>document.activeElement===find('.account-menu-panel'));
  await act(async()=>find('.outside').focus());
  assert(!find('.account-menu-panel'),'Keyboard focus can leave the non-modal panel');
  await click(find('.account-menu-trigger'));
  assert(!button('Practice guide'), 'The personal menu has no generic guide action');
  await click(button('Code backups'));
  assert.equal(backupOpens,1);assert(!find('.account-menu-panel'));assert.equal(document.activeElement,find('.account-menu-trigger'));
  await click(find('.account-menu-trigger'));await settle(()=>find('.account-menu-signout'));
  await click(find('.account-menu-signout'));
  assert(find('.account-menu-signout').disabled);assert.equal(logoutRequests,1);
  await click(find('.account-menu-signout'));assert.equal(logoutRequests,1,'A pending logout cannot send duplicate requests');
  await act(async()=>finishLogout(Response.json({error:'Sign out unavailable'},{status:503})));
  await settle(()=>find('.account-menu-error'));
  assert.equal(account.user.id,user.id,'Failed logout preserves the signed-in account');
  assert(find('.account-menu-error').textContent.includes('Sign out unavailable'));
  await click(find('.account-menu-signout'));
  await act(async()=>finishLogout(Response.json({ok:true})));
  await settle(()=>!account.user&&!find('.account-menu-panel'));
  const join=find('.header-signin-cta');
  assert(join&&join.textContent==='Join / Sign in','Guests can reach sign-in without opening the personal menu');
  assert.equal(join.getAttribute('href'),'#/account/continue/22','Registration returns to the current practice route');
  assert.equal(logoutRequests,2,'Showing the registration entry does not create any account or submission request');
  await click(find('.account-menu-trigger'));
  assert(!find('.account-menu-panel').textContent.includes(user.display_name),'The next owner cannot see the previous account identity');
  assert(!find('.account-menu-links a[href*="saved=1"]'),'Guests do not inherit an account saved-reading shortcut');
  assert(find('.account-menu-signin').getAttribute('href').includes('account/continue/22'),'Sign-in preserves the practice route');
  await act(async()=>window.dispatchEvent(new dom.window.Event('hashchange')));
  assert(!find('.account-menu-panel'),'Route changes close the account panel');

  const draft=find('textarea');draft.value='edited private draft 🧪';draft.setSelectionRange(4,9);draft.scrollTop=17;
  await click(find('.workspace-layout-trigger'));await click(button('Balanced'));
  assert.equal(localStorage.getItem('test.layout'),'0.5');
  const separator=find('[role="separator"]');
  await act(async()=>separator.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'ArrowDown',bubbles:true,cancelable:true})));
  assert.equal(Number(localStorage.getItem('test.layout')),.52);
  await click(button('Toggle results'));
  assert.equal(find('.resizable-split').getAttribute('data-second-collapsed'),'true');assert(separator.hidden);assert(find('#results'),'Collapsing results retains the mounted content');
  await click(button('Toggle results'));assert(!separator.hidden);
  assert.equal(Number(localStorage.getItem('test.layout')),.52,'Collapsing results preserves the manually resized ratio');
  await click(button('Toggle lab'));await click(button('Toggle lab'));
  assert.equal(find('textarea'),draft);assert.equal(draft.value,'edited private draft 🧪');assert.equal(draft.selectionStart,4);assert.equal(draft.selectionEnd,9);assert.equal(draft.scrollTop,17);assert.equal(mounts,1);
  await click(find('.workspace-layout-trigger'));await act(async()=>button('Reading').focus());await click(button('Reading'));
  assert.equal(localStorage.getItem('test.layout'),'0.62');assert.equal(document.activeElement,find('.workspace-layout-trigger'));
  const layoutTrigger=find('.workspace-layout-trigger');
  const key=async(node,options)=>{const event=new dom.window.KeyboardEvent('keydown',{bubbles:true,cancelable:true,...options});await act(async()=>node.dispatchEvent(event));return event;};
  await act(async()=>layoutTrigger.focus());await key(layoutTrigger,{key:'ArrowDown'});
  await settle(()=>document.activeElement===button('Reading'));
  const panel=find('.workspace-layout-popover'),preset=button('Reading'),controls=layoutTrigger.getAttribute('aria-controls');
  await act(async()=>localeControl.setLocale('zh'));assert.equal(find('.workspace-layout-popover'),panel);assert.equal(document.activeElement,preset);assert.equal(preset.textContent.trim(),'阅读优先');assert(panel.textContent.includes('均分题面与代码')&&panel.textContent.includes('收起此面板'));
  assert.equal(find('.header-signin-cta').textContent,'注册 / 登录');
  await act(async()=>localeControl.setLocale('en'));assert.equal(document.activeElement,preset);assert.equal(layoutTrigger.getAttribute('aria-controls'),controls);assert(panel.textContent.includes('Equal pane widths')&&panel.textContent.includes('Close this panel'));
  for(const options of [{isComposing:true},{keyCode:229},{repeat:true},{altKey:true},{ctrlKey:true},{metaKey:true},{shiftKey:true}]){
    const event=await key(button('Reading'),{key:'Escape',...options});
    assert(find('.workspace-layout-popover'),'Composing, repeated or modified Escape must not close layout');assert(!event.defaultPrevented);
  }
  const consume=event=>event.preventDefault();button('Reading').addEventListener('keydown',consume);
  await key(button('Reading'),{key:'Escape'});assert(find('.workspace-layout-popover'),'A consumed key keeps its original action');button('Reading').removeEventListener('keydown',consume);
  let outerEscapes=0;const outer=event=>{if(event.key==='Escape')outerEscapes++;};document.addEventListener('keydown',outer);
  const beforeKeys=JSON.stringify({...localStorage});const escaped=await key(button('Reading'),{key:'Escape'});
  assert(escaped.defaultPrevented);assert(!find('.workspace-layout-popover'));assert.equal(document.activeElement,layoutTrigger);assert.equal(outerEscapes,0,'Closing layout must not also close an outer focus mode');document.removeEventListener('keydown',outer);assert.equal(JSON.stringify({...localStorage}),beforeKeys,'Reading the layout controls does not save a new layout');
  draft.setSelectionRange(2,8,'backward');draft.scrollTop=23;draft.scrollLeft=11;
  layoutIntent=()=>draft.focus();await click(layoutTrigger);await act(async()=>button('Coding').focus());await click(button('Coding'));
  assert.equal(document.activeElement,draft,'Applying a layout yields to a newer focus choice');assert.equal(localStorage.getItem('test.layout'),'0.36');
  const href=location.href;layoutIntent=()=>{location.hash='#/newer-work';};await click(layoutTrigger);await act(async()=>button('Balanced').focus());await click(button('Balanced'));
  assert.notEqual(document.activeElement,layoutTrigger,'Applying a layout yields to a newer route');assert.equal(localStorage.getItem('test.layout'),'0.5');location.href=href;layoutIntent=()=>{};
  await act(async()=>{layoutTrigger.focus();layoutTrigger.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'ArrowDown',bubbles:true,cancelable:true}));location.hash='#/newer-reading';});
  await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert.equal(document.activeElement,layoutTrigger,'A pending keyboard open cannot move focus after navigation');location.href=href;await click(layoutTrigger);
  assert.equal(find('textarea'),draft);assert.equal(draft.value,'edited private draft 🧪');assert.equal(draft.selectionStart,2);assert.equal(draft.selectionEnd,8);assert.equal(draft.selectionDirection,'backward');assert.equal(draft.scrollTop,23);assert.equal(draft.scrollLeft,11);assert.equal(mounts,1);
  console.log('PASS layout keyboard ownership: local Escape, IME/repeat/modifier/consumed exclusions, outer isolation, keyboard opening, newer route/focus priority and unchanged mounted draft/selection/direction/both scrolls');
  console.log('PASS isolated DOM: avatar panel focus/Escape/outside/Tab/route dismissal, live language/theme, private identity, backup handoff, logout failure/retry/duplicate guards, owner change, layout persistence, accessible collapse and mounted draft/cursor/scroll preservation');
} finally {
  if(root)await act(async()=>root.unmount());
  await server.close();dom.window.close();
  globalThis.fetch=originalFetch;globalThis.ResizeObserver=originalResize;globalThis.requestAnimationFrame=originalRaf;globalThis.cancelAnimationFrame=originalCancel;
  for(const [key,descriptor] of previous)if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];
}
