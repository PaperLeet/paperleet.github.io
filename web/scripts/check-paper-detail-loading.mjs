import assert from 'node:assert/strict';
import fs from 'node:fs';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

// Exercise the existing asynchronous boundary around the real detail renderer.
// Account bookmarks are independent optional controls and never make requests here.
const dom = new JSDOM('<div id="root"></div>', {url:'https://paper-detail.test/#/papers/transformer'});
const names = ['window','document','navigator','HTMLElement','localStorage','location','fetch','IS_REACT_ACT_ENVIRONMENT'];
const originals = new Map(names.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
for (const name of ['window','document','navigator','HTMLElement','localStorage','location']) {
  Object.defineProperty(globalThis,name,{value:dom.window[name],configurable:true,writable:true});
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
globalThis.fetch = async()=>{throw Error('Unexpected network request');};
const server = await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',
  optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
    name:'paper-detail-optional-account-fixture',enforce:'pre',
    resolveId(source,importer){
      if(importer?.endsWith('/src/PaperDetail.tsx')&&source==='./deferredPages')return '\0detail-bookmarks';
    },
    load(id){if(id==='\0detail-bookmarks')return 'export const ProblemFavoriteButton=()=>null;';},
  }],
});
let root;
try {
  const {createRoot}=await import('react-dom/client');
  const {default:PaperDetail}=await server.ssrLoadModule('/src/PaperDetail.tsx');
  const {PaperDetailFailure}=await server.ssrLoadModule('/src/PaperDetailFallback.tsx');
  const {deferredFeature}=await server.ssrLoadModule('/src/deferredFeature.tsx');
  const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
  const {emptyLibraryFilters,libraryHref,readLibraryRoute,withLibraryContext}=await server.ssrLoadModule('/src/libraryNavigation.ts');
  const {selectLibraryPractice}=await server.ssrLoadModule('/src/libraryPractice.ts');
  const catalog=JSON.parse(fs.readFileSync('public/data/catalog.json','utf8'));
  const paper=catalog.papers.find(p=>p.id==='transformer');
  const originalView=libraryHref(undefined,{...emptyLibraryFilters(),query:'Vaswani Sinusoidal'});
  const latestView=libraryHref('llm-basics',{...emptyLibraryFilters(),query:'Vaswani',difficulty:'M',sort:'title'});
  const scope=view=>selectLibraryPractice(catalog.papers,catalog.problems,{},readLibraryRoute(view)).problems;
  const caught=[],uncaught=[];
  const settle=async check=>{
    for(let n=0;n<100;n++){
      if(check())return;
      await act(async()=>{await new Promise(resolve=>setTimeout(resolve,5));});
    }
    assert(check(),'Expected asynchronous detail state');
  };
  for(const locale of ['zh','en']) {
    localStorage.clear();localStorage.setItem('paperleet.locale',locale);
    const draft=JSON.stringify({code:'# synthetic retained draft 😀\n',updated:'2026-10-16T10:00:00Z'});
    localStorage.setItem('paperleet.fixture.learner-draft',draft);
    const snapshot=JSON.stringify(Object.entries(localStorage).sort());
    let attempts=0,rejectImport,reloads=0;
    const failure={failure:props=>React.createElement(PaperDetailFailure,{returnTo:props.returnTo,recovery:props.recovery})};
    const Deferred=deferredFeature(()=>{
      attempts++;
      return new Promise((_resolve,reject)=>{rejectImport=reject;});
    },['论文详情','paper details'],failure);
    let CurrentDetail=Deferred;
    let props={paper,problems:catalog.problems,saved:{},practiceProblems:scope(originalView),returnTo:originalView,
      recovery:{blocked:false,reason:'',reload:()=>{reloads++;}}};
    const render=()=>React.createElement(LocaleProvider,null,
      React.createElement('header',null,React.createElement('a',{'data-shell-return':true,href:props.returnTo},'Library')),
      React.createElement(CurrentDetail,props));
    await act(async()=>{
      root=createRoot(document.querySelector('#root'),{onCaughtError:error=>caught.push(error),onUncaughtError:error=>uncaught.push(error)});
      root.render(render());
    });
    assert(document.querySelector('[aria-busy="true"]'));
    assert(document.body.textContent.includes(locale==='en'?'Loading paper details':'正在加载论文详情'));
    assert.equal(document.querySelector('.paper-detail'),null);
    assert.equal(attempts,1);

    // A slow import must not capture the old search or lose the surrounding shell.
    props={...props,practiceProblems:scope(latestView),returnTo:latestView};
    await act(async()=>root.render(render()));
    assert.equal(attempts,1,'A prop update must not restart the pending import');
    await act(async()=>rejectImport(Error('fixture detail transport failure')));
    await settle(()=>document.querySelector('[role="alert"]'));
    assert(document.body.textContent.includes(locale==='en'?'Paper details could not load':'论文详情暂时无法加载'));
    assert(!document.body.textContent.includes('fixture detail transport failure'),'Internal import errors stay out of the product UI');
    assert.equal(document.querySelector('[data-shell-return]').getAttribute('href'),latestView);
    assert(!document.body.textContent.includes(locale==='en'?'Retry loading':'重试加载'),'A cached failed module needs a fresh document, rather than an ineffective import retry');
    const reloadButton=()=>[...document.querySelectorAll('button')].find(button=>button.textContent===(locale==='en'?'Reload paper':'重新加载论文'));
    props={...props,recovery:{...props.recovery,blocked:true,reason:'Saving fixture draft'}};
    await act(async()=>root.render(render()));
    assert(reloadButton().disabled);
    assert(document.querySelector('[role="status"]').textContent.includes('Saving fixture draft'));
    await act(async()=>reloadButton().dispatchEvent(new window.MouseEvent('click',{bubbles:true})));
    assert.equal(reloads,0,'Pending saves cannot be interrupted by recovery');
    props={...props,recovery:{...props.recovery,blocked:false,reason:''}};
    await act(async()=>root.render(render()));
    assert(!reloadButton().disabled);
    assert.equal(document.querySelector('.feature-recovery-actions a').getAttribute('href'),latestView);
    await act(async()=>reloadButton().dispatchEvent(new window.MouseEvent('click',{bubbles:true})));
    assert.equal(reloads,1);
    // Browser verification also removes/restores the actual built script. Model
    // its fresh-document recovery here, retaining the latest route and storage.
    await act(async()=>root.unmount());root=undefined;
    CurrentDetail=deferredFeature(()=>{attempts++;return Promise.resolve({default:PaperDetail});},['论文详情','paper details'],failure);
    await act(async()=>{
      root=createRoot(document.querySelector('#root'),{onCaughtError:error=>caught.push(error),onUncaughtError:error=>uncaught.push(error)});
      root.render(render());
    });
    await settle(()=>document.querySelector('.paper-primary-action a'));
    const expected=withLibraryContext(27,latestView);
    assert.equal(document.querySelector('.paper-primary-action a').getAttribute('href'),expected,'Fresh-document recovery uses the newest collection/difficulty/sort selection');
    assert.equal(document.querySelector('.paper-library-link').getAttribute('href'),latestView);
    assert.equal(document.querySelectorAll('.paper-module-card').length,paper.module_ids.length);

    props={...props,practiceProblems:[],progressReady:false};
    await act(async()=>root.render(render()));
    assert.equal(document.querySelector('.paper-primary-action a'),null);
    assert(document.body.textContent.includes(locale==='en'?'Loading progress':'正在加载学习进度'));
    props={...props,progressReady:true};
    await act(async()=>root.render(render()));
    assert(document.body.textContent.includes(locale==='en'?'No modules match this library view':'当前题库筛选中没有匹配模块'));
    assert.equal(attempts,2,'Progress changes must retain the loaded route');
    assert.equal(JSON.stringify(Object.entries(localStorage).sort()),snapshot,'Loading, recovery and progress changes must not write learner storage');
    await act(async()=>root.unmount());root=undefined;
  }
  assert.equal(uncaught.length,0);
  assert.equal(caught.length,2);
  assert(caught.every(error=>error.message==='fixture detail transport failure'));
  console.log('PASS paper detail loading: bilingual pending/error/guarded reload, retained shell/storage, newest filtered recommendations and return context, full curriculum, loading/empty progress and no import restart');
} finally {
  if(root)await act(async()=>root.unmount());
  await server.close();dom.window.close();
  for(const [name,descriptor] of originals)descriptor?Object.defineProperty(globalThis,name,descriptor):delete globalThis[name];
}
