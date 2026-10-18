import assert from 'node:assert/strict';
import React,{act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {mkdtempSync,rmSync} from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const dom=new JSDOM('<!doctype html><textarea id="private">Private implementation 😀</textarea><div id="root"></div>',{url:'http://paperleet.test/#/community/moderation',pretendToBeVisual:true});
let scrolls=0;dom.window.scrollTo=()=>scrolls++;dom.window.HTMLElement.prototype.scrollIntoView=()=>scrolls++;
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,Element:dom.window.Element,Node:dom.window.Node,location:dom.window.location,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');const cache=mkdtempSync(path.join(os.tmpdir(),'paperleet-moderation-comparison-'));
const server=await createServer({cacheDir:cache,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});let root;
try{
 const model=await server.ssrLoadModule('/src/moderationTextDiff.ts');
 function verify(before,after){
  const value=model.moderationTextDiff(before,after);assert(value);
  assert.equal(value.lines.filter(row=>row.kind!=='added').map(row=>row.text).join(''),before);
  assert.equal(value.lines.filter(row=>row.kind!=='removed').map(row=>row.text).join(''),after);
  let old=0,next=0;for(const row of value.lines){if(row.kind!=='added')assert.equal(row.before,++old);if(row.kind!=='removed')assert.equal(row.after,++next);}
  assert.equal(value.added,value.lines.filter(row=>row.kind==='added').length);assert.equal(value.removed,value.lines.filter(row=>row.kind==='removed').length);
  return value;
 }
 for(const [before,after]of [['',''],['','one\n'],['one\n',''],['a\n','a'],['a\n','a\r\n'],['a\rb\r','a\nb\n'],['\n\n','\n'],['中文😀 e\u0301\n\tindented\n','中文😀 e\u0301\n\tchanged\n'],['same\nold\nrepeated\nend\n','same\nnew\nrepeated\nend\n']])verify(before,after);
 let seed=71;const random=n=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed%n;},tokens=['a\n','b\n','\n','中文😀\r\n','\tcode\r','<img src=x>'];
 for(let iteration=0;iteration<100;iteration++){const source=()=>Array.from({length:random(14)},()=>tokens[random(tokens.length)]).join('');verify(source(),source());}
 const aligned=verify('head\none\nsame\ntwo\nend\n','head\nnew\nsame\nchanged\nend\n');assert.equal(aligned.mode,'lines');assert.equal(aligned.added,2);assert.equal(aligned.removed,2);
 const largeOld=Array.from({length:1200},(_,i)=>'old '+i+'\n').join(''),largeNew=Array.from({length:1200},(_,i)=>'new '+i+'\n').join('');
 const large=verify(largeOld,largeNew);assert.equal(large.mode,'block');const bounded=model.moderationDiffRows(large);assert(bounded.truncated);assert.equal(bounded.rows.filter(row=>row.kind!=='gap').length,model.MAX_DIFF_ROWS);
 const middle=Array.from({length:200},(_,i)=>'same '+i+'\n').join('');const gaps=model.moderationDiffRows(verify('first old\n'+middle+'last old','first new\n'+middle+'last new'));assert(gaps.rows.some(row=>row.kind==='gap'&&row.count>180));assert(!gaps.truncated);
 assert.equal(model.moderationTextDiff('x'.repeat(model.MAX_DIFF_SOURCE+1),'small'),undefined);
 assert.deepEqual(model.diffLineContent('a\r\n'),{text:'a',ending:'CRLF'});assert.deepEqual(model.diffLineContent('a'),{text:'a',ending:'EOF'});

 const h=React.createElement,{createRoot}=await import('react-dom/client'),{LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx'),{default:Comparison}=await server.ssrLoadModule('/src/ModerationSnapshotComparison.tsx');
 const base={id:'a'.repeat(32),title:'Reported title',body:'original\n<img src="https://untrusted.test/track" onerror="bad()">\n\tcode 😀',category:'questions',edit_version:'a'.repeat(64),updated_at:'2026-10-17T00:00:00Z',author:{display_name:'Anonymous',username:'',anonymous:true}};
 let before=base,after={...base,title:'Current title',body:'replacement\n<script>bad()</script>\n\tnew 😀\r\n',category:'career',edit_version:'b'.repeat(64)},setLocale;
 function Language(){setLocale=useI18n().setLocale;return null;}
 function App(){return h(LocaleProvider,null,h(Language),h(Comparison,{before,after}));}
 root=createRoot(document.querySelector('#root'));const render=()=>act(async()=>root.render(h(App)));
 const editor=document.querySelector('#private');editor.setSelectionRange(1,9,'backward');editor.scrollTop=31;editor.focus();
 await render();assert(document.querySelector('.moderation-diff-added'));assert(document.querySelector('.moderation-diff-removed'));
 assert.equal(document.querySelector('img,script,iframe'),null);assert(!document.querySelector('a[href*="untrusted"]'));
 const full=document.querySelector('.moderation-comparison-sources');full.open=true;const sources=()=>[...full.querySelectorAll('pre')];assert.deepEqual(sources().map(node=>node.textContent),[before.body,after.body]);sources()[0].scrollTop=41;
 const view=document.querySelector('.moderation-comparison-scroll');view.scrollTop=55;view.focus();await act(async()=>setLocale('zh'));assert(full===document.querySelector('.moderation-comparison-sources')&&full.open);assert(document.activeElement===view);assert.equal(view.scrollTop,55);assert(document.body.textContent.includes('新增'));
 after={...after,body:after.body+'latest exact source'};await render();assert(full===document.querySelector('.moderation-comparison-sources'));assert(full.open&&sources()[0].scrollTop===41);assert.equal(sources()[1].textContent,after.body);assert(document.activeElement===view);
 await act(async()=>setLocale('en'));before={...base,body:largeOld};after={...base,body:largeNew};await render();assert(document.body.textContent.includes('complete replacement'));assert.equal(document.querySelectorAll('.moderation-comparison tbody tr').length,model.MAX_DIFF_ROWS);assert(document.body.textContent.includes('first 400 rows'));assert.deepEqual(sources().map(node=>node.textContent),[largeOld,largeNew]);
 before=base;after={...base,edit_version:'c'.repeat(64)};await render();assert(document.body.textContent.includes('version changed'));assert(!document.querySelector('.moderation-comparison table'));
 before={...base,body:'x'.repeat(model.MAX_DIFF_SOURCE+1)};after=base;await render();assert(document.body.textContent.includes('exceeds the line comparison limit'));assert.equal(sources()[0].textContent,before.body);
 assert.deepEqual([editor.value,editor.selectionStart,editor.selectionEnd,editor.selectionDirection,editor.scrollTop],['Private implementation 😀',1,9,'backward',31]);assert.equal(scrolls,0);assert.equal(sessionStorage.length,0);assert.deepEqual({...localStorage},{'paperleet.locale':'en'});
 console.log('PASS moderation snapshot comparison: exact source reconstruction and line numbering across100repeat/Unicode/blank/newline cases, bounded LCS with honest large-block replacement, contextual gaps and400row preview with complete raw sources, literal hostile Markdown/HTML and zero remote assets, title/category/body scope and other-field notice, retained native source disclosure/scroll/focus across locale/current updates, no writes/storage/navigation or private code/caret changes; controlled DOM');
}finally{if(root)await act(async()=>root.unmount());await server.close();rmSync(cache,{recursive:true,force:true});dom.window.close();for(const [key,value]of previous)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}
