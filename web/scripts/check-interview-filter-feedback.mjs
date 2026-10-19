import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';

const readonly=process.argv.includes('--readonly');
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/interviews',pretendToBeVisual:true});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');sessionStorage.setItem('private-filter-feedback','KEEP');
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-filter-feedback-');
const server=await createServer({configFile:false,esbuild:{jsx:'automatic'},cacheDir,server:{watch:null,middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},define:{'import.meta.env.VITE_APP_EDITION':JSON.stringify(readonly?'local':'hosted'),'import.meta.env.VITE_PUBLIC_SITE':JSON.stringify(readonly?'1':'0')}});
let root;
try{
 const {createRoot}=await import('react-dom/client'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Browse}=await server.ssrLoadModule('/src/InterviewBrowseFilters.tsx'),nav=await server.ssrLoadModule('/src/interviewNavigation.ts');
 let filters={...nav.emptyInterviewFilters(),view:readonly?'questions':'reports',company:'google',role:'ml',days:'90'},allowed=true,loading=false,available=true,owner='alice',locale;
 const changes=[];
 function Frame(){
  locale=useI18n();const href=location.href,actor=owner;
  return React.createElement(React.Fragment,null,React.createElement('textarea',{'aria-label':'Private editor',defaultValue:'PRIVATE_FILTER_CODE'}),React.createElement(Browse,{key:owner,filters,loading,contextAvailable:available,outcomeAvailable:true,availabilityNotice:true,searchHelp:'search-help',outcomeHelp:'outcome-help',canAct:()=>allowed&&owner===actor&&location.href===href,onChange:patch=>{changes.push(patch);filters={...filters,...patch,page:1};location.hash=nav.interviewHref(filters);root.render(tree());}}));
 }
 const tree=()=>React.createElement(React.StrictMode,null,React.createElement(LocaleProvider,null,React.createElement(Frame)));
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(tree());});
 const panel=()=>document.querySelector('.interview-more-filters'),summary=()=>panel().querySelector('summary'),form=()=>panel().querySelector('form');
 const field=suffix=>form().querySelector(`input[id$="-${suffix}"]`),button=label=>[...panel().querySelectorAll('button')].find(node=>node.textContent===label);
 const toggle=async open=>act(async()=>{panel().open=open;panel().dispatchEvent(new window.Event('toggle'));});
 const input=async(suffix,value)=>act(async()=>{const node=field(suffix);node.focus();Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set.call(node,value);node.dispatchEvent(new window.Event('input',{bubbles:true}));});
 const click=async node=>{assert(node,'Expected filter action');await act(async()=>{node.focus();node.click();});};
 const submit=async()=>act(async()=>form().requestSubmit(form().querySelector('[type=submit]')));
 const key=async(node,extra={},handled=false)=>{const event=new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true,cancelable:true,...extra});if(handled)event.preventDefault();await act(async()=>node.dispatchEvent(event));return event;};
 location.hash=nav.interviewHref(filters);await render();await toggle(true);
 const editor=document.querySelector('textarea'),box=panel();editor.setSelectionRange(2,11,'backward');editor.scrollTop=71;editor.scrollLeft=19;
 const snapshot=[editor.value,editor.selectionStart,editor.selectionEnd,editor.selectionDirection,editor.scrollTop,editor.scrollLeft],storage=JSON.stringify({...sessionStorage});
 if(readonly){
  const select=panel().querySelector('select');select.focus();assert(!(await key(select)).defaultPrevented);assert(panel().open,'Native practice selectors keep Escape');
  summary().focus();assert((await key(summary())).defaultPrevented);assert(!panel().open);assert(document.activeElement===summary());assert.equal(changes.length,0);assert.equal(form(),null);
  console.log('PASS readonly filter keyboard: actual self-study controls, native select Escape ownership, explicit summary collapse/focus and zero changes or context form');
 }else{
  assert.equal(form().noValidate,true,'Localized field validation owns out-of-range month feedback');
  await input('location','x'.repeat(81));const before=location.href;await submit();
  assert(document.activeElement===field('location'),'Apply focuses the invalid location');assert.equal(field('location').getAttribute('aria-invalid'),'true');
  const errorId=form().querySelector('[role=alert]').id;assert(field('location').getAttribute('aria-describedby').split(' ').includes(errorId));assert(!field('from').hasAttribute('aria-invalid'));assert.equal(changes.length,0);assert.equal(location.href,before);
  await input('from','2026-09');assert.equal(field('location').getAttribute('aria-invalid'),'true','An unrelated edit keeps the invalid location visible');assert(document.activeElement===field('from'),'Typing never redirects focus');
  const nativeLocation=field('location');await act(async()=>locale.setLocale('zh'));assert.equal(field('location'),nativeLocation);assert(form().querySelector('[role=alert]').textContent.includes('80'));assert.equal(form().querySelector('[role=alert]').id,errorId);await act(async()=>locale.setLocale('en'));
  await input('location',' Remote / 上海 ');assert.equal(form().querySelector('[role=alert]'),null);await input('to','2100-01');await submit();
  assert(document.activeElement===field('to'));assert.equal(field('to').getAttribute('aria-invalid'),'true');assert(form().querySelector('[role=alert]').textContent.includes('1900-01'));
  await input('from','1899-01');assert.equal(field('from').getAttribute('aria-invalid'),'true');assert(!field('to').hasAttribute('aria-invalid'));await submit();assert(document.activeElement===field('from'));
  await input('from','2026-09');assert.equal(field('to').getAttribute('aria-invalid'),'true');await input('to','2026-01');await submit();assert(document.activeElement===field('to'));assert(form().querySelector('[role=alert]').textContent.includes('cannot be before'));assert.equal(changes.length,0);
  nativeLocation.focus();nativeLocation.setSelectionRange(2,7,'backward');nativeLocation.scrollTop=12;nativeLocation.scrollLeft=33;
  for(const extra of [{repeat:true},{ctrlKey:true},{metaKey:true},{altKey:true},{shiftKey:true},{isComposing:true},{keyCode:229}])assert(!(await key(nativeLocation,extra)).defaultPrevented);
  await act(async()=>nativeLocation.dispatchEvent(new window.CompositionEvent('compositionstart',{bubbles:true})));assert(!(await key(nativeLocation)).defaultPrevented);await act(async()=>nativeLocation.dispatchEvent(new window.CompositionEvent('compositionend',{bubbles:true})));
  assert((await key(nativeLocation,{},true)).defaultPrevented);assert(panel().open,'Already handled Escape leaves the panel open');
  for(const native of [field('from'),panel().querySelector('select')]){native.focus();assert(!(await key(native)).defaultPrevented);assert(panel().open);}
  const dialog=document.createElement('dialog'),dialogButton=document.createElement('button');dialog.open=true;dialog.append(dialogButton);form().append(dialog);dialogButton.focus();assert(!(await key(dialogButton)).defaultPrevented);assert(panel().open);dialog.remove();
  editor.focus();assert(!(await key(nativeLocation)).defaultPrevented);assert(document.activeElement===editor,'An unfocused field cannot claim Escape');
  nativeLocation.focus();assert((await key(nativeLocation)).defaultPrevented);assert(!panel().open);assert(document.activeElement===summary());assert(form().querySelector('[role=alert]'),'Collapse retains validation');assert.deepEqual([nativeLocation.selectionStart,nativeLocation.selectionEnd,nativeLocation.selectionDirection,nativeLocation.scrollTop,nativeLocation.scrollLeft],[2,7,'backward',12,33]);assert.equal(changes.length,0);assert.equal(location.href,before);
  editor.focus();await submit();assert(document.activeElement===editor,'A closed context form cannot claim validation focus');await toggle(true);assert.equal(field('location'),nativeLocation);
  for(const attribute of ['hidden','inert','aria-hidden']){
   panel().setAttribute(attribute,attribute==='aria-hidden'?'true':'');editor.focus();const count=changes.length;
   await submit();await act(async()=>button('Collapse filters').click());assert(panel().open);assert(document.activeElement===editor);assert.equal(changes.length,count);panel().removeAttribute(attribute);
  }
  const currentHref=location.href;window.history.replaceState(null,'','#/favorites');editor.focus();await submit();await act(async()=>button('Collapse filters').click());assert(panel().open);assert(document.activeElement===editor);assert.equal(changes.length,0);assert.equal(location.hash,'#/favorites');window.history.replaceState(null,'',currentHref);
  allowed=false;editor.focus();await submit();await act(async()=>button('Collapse filters').click());assert(panel().open);assert(document.activeElement===editor);allowed=true;
  await input('to','2026-10');assert(!form().querySelector('[role=alert]'));loading=true;await render();editor.focus();await submit();assert(document.activeElement===editor);assert.equal(changes.length,0);loading=false;available=false;await render();await submit();assert.equal(changes.length,0);available=true;await render();
  await input('from','');const nativeMonth=field('from');Object.defineProperty(nativeMonth,'validity',{configurable:true,value:{badInput:true}});await submit();assert.equal(changes.length,0);assert(document.activeElement===nativeMonth,'A partial native month cannot silently broaden the filter');assert.equal(nativeMonth.getAttribute('aria-invalid'),'true');delete nativeMonth.validity;await input('from','2026-09');assert(!form().querySelector('[role=alert]'));
  const apply=form().querySelector('[type=submit]');await click(apply);assert.equal(changes.length,1);assert.equal(filters.location,'Remote / 上海');assert.equal(filters.month_from,'2026-09');assert.equal(filters.month_to,'2026-10');assert.equal(filters.company,'google');assert.equal(filters.role,'ml');assert.equal(filters.days,'90');assert(document.activeElement===apply);assert.equal(field('location'),nativeLocation);
  await input('location','UNAPPLIED');const appliedScope=location.href;await click(button('Restore applied values'));assert.equal(field('location').value,'Remote / 上海');assert(document.activeElement===nativeLocation,'Restore focuses a surviving native field');assert.equal(changes.length,1);assert.equal(location.href,appliedScope);assert(!summary().textContent.includes('Unapplied edits'));
  await click(button('Clear location and months'));assert.equal(changes.length,2);assert.equal(filters.location,'');assert.equal(filters.month_from,'');assert.equal(filters.month_to,'');assert.equal(filters.company,'google');assert.equal(filters.role,'ml');assert.equal(filters.days,'90');assert(document.activeElement===nativeLocation);assert(!button('Clear location and months'));assert(!form().querySelector('[role=alert]'));
  await input('location','UNSENT_OWNER_NOTE');owner='bob';await render();assert(!box.isConnected);assert.notEqual(panel(),box);assert.equal(field('location').value,'');assert.equal(changes.length,2);
  console.log('PASS interview field feedback: actual localized native form, first invalid location/month focus and descriptions, range/order cascade, unrelated-edit diagnosis, no premature apply, stable locale fields, restore/clear focus and exact retained company/role/publication scope');
  console.log('PASS advanced filter keyboard and guards: Escape preserves draft/error/caret/two-scroll, native selects/months/dialogs and IME/modified/handled/unfocused keys keep ownership, closed/hidden/inert/stale/disallowed forms cannot focus or apply, pending support and owner retirement');
 }
 assert.deepEqual([editor.value,editor.selectionStart,editor.selectionEnd,editor.selectionDirection,editor.scrollTop,editor.scrollLeft],snapshot);assert.equal(JSON.stringify({...sessionStorage}),storage);assert.equal(document.querySelector('textarea'),editor);
 console.log('PASS filter workspace isolation: unchanged private editor identity/code/selection/direction/two-scroll and session draft storage');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();for(const[key,value]of previous)value?Object.defineProperty(globalThis,key,value):delete globalThis[key];
}
