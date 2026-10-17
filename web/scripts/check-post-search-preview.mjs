import assert from 'node:assert/strict';
import React,{act} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'https://paperleet.test/#/community',pretendToBeVisual:true});
dom.window.matchMedia=()=>({matches:true});dom.window.HTMLElement.prototype.scrollIntoView=function(){};dom.window.scrollTo=()=>{};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__postSearchPreview:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));for(const[key,value]of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});localStorage.setItem('paperleet.locale','en');
const cacheDir=await mkdtemp(tmpdir()+'/paperleet-search-preview-');
const server=await createServer({cacheDir,server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
 name:'search-preview-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0search-preview-social';},load(name){if(name==='\0search-preview-social')return `import React from 'react';export const useAccount=()=>globalThis.__postSearchPreview.account;export const socialApi=(...args)=>globalThis.__postSearchPreview.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;`;}
}]});
let root;
try{
 const {postPreview}=await server.ssrLoadModule('/src/postExcerpt.ts');
 const literal=await server.ssrLoadModule('/src/postSearchModel.ts');
 const {default:Match}=await server.ssrLoadModule('/src/PostSearchMatch.tsx');
 const {default:Preview}=await server.ssrLoadModule('/src/PostPreview.tsx');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const h=React.createElement;
 const body='Opening context. '.repeat(40)+'Important masked_attention explanation near the end.';
 const result=postPreview(body,190,'[formula]','masked_attention');
 assert(result.text.includes('masked_attention'),'Search results need the matching passage rather than only the first paragraph');assert(result.text.startsWith('…'));assert.equal(result.search,'text');assert(result.truncated);assert(result.text.length<=192);
 assert.equal(postPreview('A short masked_attention paragraph.',190,'[formula]','MASKED_ATTENTION').text,'A short masked_attention paragraph.');
 for(const [source,kind,term]of [
  ['## masked_attention heading\n\nOrdinary prose.','text','masked_attention'],
  ['Intro\n\n```python\ndef masked_attention(q, k):\n    return q @ k.T\n```','code','masked_attention'],
  ['$$\n\\operatorname{softmax}(QK^T)\n$$','math','softmax'],
  ['| Metric | Value |\n| --- | --- |\n| Accuracy | 0.8 |','table','Accuracy'],
  ['Before $x^2$ after.','text','x^2'],
 ]){const value=postPreview(source,90,'[formula]',term);assert.equal(value.search,kind);assert(value.text.includes(term));}
 for(const term of ['%','_','[a-z]','.*','a+b','(x)','<img src=x onerror=boom()>']){
  const value=postPreview('Literal `'+term+'` content.',190,'[formula]',term);assert.equal(value.search,'text');assert(literal.postSearchRanges(value.text,term).length);
 }
 assert.equal(literal.postSearchIndex('É İ ß','é'),-1);assert.equal(literal.postSearchIndex('ABC_xyz','abc_XYZ'),0);assert.equal(literal.postSearchIndex('ＡＢＣ','ABC'),-1);
 assert.equal(postPreview('Only $x^2$ here.',190,'[formula]','formula').search,undefined,'Generated formula placeholders are not search evidence');
 assert.equal(postPreview('| A | B |\n| --- | --- |\n| 1 | 2 |',190,'[formula]','·').search,undefined,'Generated table separators are not search evidence');
 for(const source of ['[Useful label](https://tracker.invalid/hidden)', '![hidden alt](https://tracker.invalid/pixel)', '<script>hidden()</script>']){
  const value=postPreview(source,190,'[formula]','hidden');assert.equal(value.search,'source');assert(!value.text.includes('hidden'));assert(!value.text.includes('tracker.invalid'));
 }
 assert.equal(postPreview('x'.repeat(20000)+' target',190,'[formula]','target').search,'outside');assert.equal(postPreview('x'.repeat(40000)+' unbounded',190,'[formula]','unbounded').search,undefined);
 const large='needle'.repeat(30),wide=postPreview('before '+large+' after',30,'[formula]',large);assert(wide.text.includes(large),'A valid long literal query is not cut off by the nominal excerpt limit');
 assert.equal(literal.postSearchQuery('a'.repeat(201)),'');assert.equal(literal.postSearchQuery('\ud800'),'');assert.equal(literal.postSearchRanges('aaaa','').length,0);assert(literal.postSearchRanges('ab '.repeat(500),'a').length<=16);
 const family='👩🏽‍💻',accent='e\u0301';assert.deepEqual(literal.postSearchRanges(family,'🏽'),[{start:0,end:family.length}]);assert.deepEqual(literal.postSearchRanges(accent,'\u0301'),[{start:0,end:accent.length}]);
 const emoji=postPreview('x'.repeat(50)+family+' target '+family.repeat(20),18,'[formula]','🏽');assert(emoji.text.includes(family));assert(!/(?<![\ud800-\udbff])[\udc00-\udfff]|[\ud800-\udbff](?![\udc00-\udfff])/u.test(emoji.text));
 const clipped=literal.postSearchWindow('a'.repeat(40)+' … '+('b'.repeat(100)),'…',40);assert(clipped.text.startsWith('…'));const markHtml=renderToStaticMarkup(h(Match,{text:clipped.text,query:'…',range:clipped.searchRange}));assert.equal((markHtml.match(/<mark /g)||[]).length,1,'Only the real ellipsis is highlighted, not clipping markers');
 const escaped=renderToStaticMarkup(h(Match,{text:'<img src=x onerror=boom()>',query:'<img'}));assert(escaped.includes('&lt;img'));assert(!escaped.includes('<img '));assert(!escaped.includes('dangerouslySetInnerHTML'));
 const rendered=renderToStaticMarkup(h(LocaleProvider,null,h(Preview,{body:'```html\n<script>masked_attention()</script>\n```',query:'masked_attention'})));assert(rendered.includes('post-search-match'));assert(!/<script|<img|<iframe|<table|<pre|class="katex/.test(rendered));
 assert.deepEqual(postPreview(body),postPreview(body,210,'[formula]',''),'Empty search keeps the original narrative preview contract');
 assert.equal(postPreview('[Visible needle](https://tracker.invalid/hidden)',190,'[formula]','needle').search,'text');
 assert.equal(postPreview('Only visible text.',190,'[formula]','absent').search,undefined);
 const priorSegmenter=Intl.Segmenter;
 try{Intl.Segmenter=undefined;assert.deepEqual(literal.postSearchRanges('A😀Z','😀'),[{start:1,end:3}]);assert(literal.postSearchWindow('x'.repeat(60)+'😀 tail','😀',12).text.includes('😀'));}finally{Intl.Segmenter=priorSegmenter;}

 console.log('PASS search preview model: one bounded Markdown parse, literal ASCII-insensitive matching, prose/headings/code/math/table passages, full query/grapheme safety, bounded highlight nodes, honest source/outside fallback, no generated-marker hits or remote assets and unchanged empty-query summaries');

 const {createRoot}=await import('react-dom/client');
 const {default:Community,ProblemSolutions}=await server.ssrLoadModule('/src/Community.tsx');
 const nav=await server.ssrLoadModule('/src/communityNavigation.ts');
 const reading=await server.ssrLoadModule('/src/readingReturn.ts');
 const topic={id:'architecture/attention',number:1,title:'Attention'};
 const user={id:'alice',username:'alice',display_name:'Alice',can_publish_community:true,avatar_color:'#48766b',affiliation:{kind:'institution',domain:'research.edu',name:'测试研究院',name_en:'Research Institute'}};
 const account={user,loading:false,capabilities:{community_solutions:true,community_feed_period:true,community_feed_sorting:true,community_bookmarks:true},refresh:async()=>{}};
 const basePost={id:'a'.repeat(32),title:'MASKED_attention explained',body,category:'implementation',solution:true,problem_id:topic.id,problem:topic,author:user,created_at:'2026-10-17T10:00:00Z',updated_at:'2026-10-17T10:00:00Z',counts:{likes:2,replies:0,reposts:0},viewer:{bookmarked:true},replies:[]};
 let currentPost=basePost,embedded=false,postId,hold=false;const requests=[],pending=[],mutations=[];
 const response=request=>({items:[currentPost],total:42,limit:20,offset:Number(request.params.get('offset')),sort:request.params.get('sort')||'latest',solutions:request.params.get('solutions')==='true',...(request.params.get('days')?{days:request.params.get('days')}:{})});
 globalThis.__postSearchPreview={account,request(path,write,signal){
  if(write){mutations.push({path,write});throw Error('Search cannot publish or react');}
  const request={path,params:new URLSearchParams(path.split('?')[1]),signal};requests.push(request);
  if(path==='community/posts/'+basePost.id)return Promise.resolve(currentPost);
  if(hold)return new Promise(resolve=>pending.push({...request,resolve}));return Promise.resolve(response(request));
 }};
 function Locale(){const {locale,setLocale}=useI18n();return h('button',{id:'search-preview-locale',onClick:()=>setLocale(locale==='en'?'zh':'en')},'Language');}
 const render=async()=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(h(LocaleProvider,null,h(Locale),h('textarea',{'aria-label':'Private practice code',defaultValue:'PRIVATE_IMPLEMENTATION_KEEP'}),embedded?h(ProblemSolutions,{topic,onCode:()=>document.querySelector('[aria-label="Private practice code"]').focus()}):h(Community,{route:location.hash,...(postId?{postId}:{topicId:topic.id,topic}),topics:[topic]})));});
 const pause=()=>new Promise(resolve=>setTimeout(resolve,5));
 const navigate=async href=>{await act(async()=>{location.hash=href;await pause();});await render();};
 const click=async node=>{assert(node,'Missing search interaction');await act(async()=>{node.focus();node.click();});await render();};
 const change=async(node,value)=>{assert(node);await act(async()=>{Object.getOwnPropertyDescriptor(node.tagName==='TEXTAREA'?dom.window.HTMLTextAreaElement.prototype:node.tagName==='SELECT'?dom.window.HTMLSelectElement.prototype:dom.window.HTMLInputElement.prototype,'value').set.call(node,value);node.dispatchEvent(new dom.window.Event(node.tagName==='SELECT'?'change':'input',{bubbles:true}));});await render();};
 const search=()=>document.querySelector('.community-search input');
 const apply=async query=>{await change(search(),query);await act(async()=>search().form.dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));await render();};
 const named=text=>[...document.querySelectorAll('button')].find(node=>node.textContent===text);
 const settle=async test=>{for(let step=0;step<80&&!test();step++)await act(async()=>new Promise(resolve=>setTimeout(resolve,10)));assert(test(),'Search reading did not settle');};
 const originalScope=nav.communityFeedHref(topic.id,{...nav.emptyCommunityFeed(),days:'7',sort:'top',saved:true});await navigate(originalScope);
 assert(!document.querySelector('mark'));const opening=document.querySelector('.post-excerpt').textContent;
 const code=document.querySelector('[aria-label="Private practice code"]');code.setSelectionRange(3,10,'backward');code.scrollTop=51;code.scrollLeft=17;sessionStorage.setItem('private-search-draft','Unpublished content');
 const beforeTyping=requests.length;await change(search(),'masked_attention');assert.equal(requests.length,beforeTyping);assert.equal(document.querySelector('.post-excerpt').textContent,opening);assert(!document.querySelector('mark'));
 await apply('masked_attention');assert.equal(requests.at(-1).params.get('q'),'masked_attention');assert.equal(requests.at(-1).params.get('days'),'7');assert.equal(requests.at(-1).params.get('sort'),'top');assert.equal(requests.at(-1).params.get('saved'),'true');
 assert.equal(document.querySelector('.post-title mark').textContent,'MASKED_attention');assert(document.querySelector('.post-excerpt mark').textContent.includes('masked_attention'));assert(!document.querySelector('.post-card-meta mark'));assert(!document.querySelector('.inline-post-body .community-prose'));
 const scope=location.hash,href=document.querySelector('a.post-title').getAttribute('href');assert.equal(nav.communityPostReturn(href,basePost.id),scope);
 await change(search(),'unapplied replacement');assert.equal(document.querySelector('.post-title mark').textContent,'MASKED_attention');assert.equal(requests.at(-1).params.get('q'),'masked_attention');
 const beforeLocale=requests.length;await click(document.querySelector('#search-preview-locale'));assert.equal(location.hash,scope);assert.equal(requests.length,beforeLocale);assert(document.querySelector('.post-excerpt mark'));await click(document.querySelector('#search-preview-locale'));
 const beforeExpansion=requests.length;await click(document.querySelector('.inline-comment-toggle'));await settle(()=>document.querySelector('.inline-post-body .discussion-reading'));assert.equal(requests.length,beforeExpansion);assert.equal(document.querySelector('[aria-label="Private practice code"]'),code);await click(document.querySelector('.inline-comment-collapse'));assert(document.querySelector('.post-excerpt mark'));assert.equal(document.activeElement,document.querySelector('.inline-comment-toggle'));
 // A click on the highlighted child retains the actual post/reading return contract.
 await act(async()=>{const anchor=document.querySelector('a.post-title');anchor.focus();document.addEventListener('click',event=>event.preventDefault(),{once:true});anchor.querySelector('mark').dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true,cancelable:true,detail:0}));});assert.equal(reading.currentReadingReturn().href,scope);
 postId=basePost.id;await navigate(href);assert.equal(reading.currentReadingReturn().phase,'reading');assert.equal(document.querySelector('.discussion-navigation a').getAttribute('href'),scope);
 postId=undefined;await navigate(scope);assert.equal(reading.currentReadingReturn(),undefined);assert.equal(document.activeElement,document.querySelector('a.post-title'));assert(document.querySelector('[data-reading-resumed="true"]'));assert(document.querySelector('.post-title mark'));
 currentPost={...basePost,title:'Literal <img src=x onerror=boom()>',body:'![hidden alt](https://tracker.invalid/pixel)'};await apply('<img');assert(document.querySelector('.post-title').textContent.includes('<img'));assert(!document.querySelector('img[src="x"]'));assert(!document.querySelector('.post-excerpt mark'));
 await apply('hidden');assert(document.querySelector('[data-search-context="source"]'));assert(document.querySelector('.post-search-note').textContent.includes('formatting source'));assert(!document.querySelector('img[src^="https://tracker.invalid"]'));assert(!document.querySelector('.post-excerpt').textContent.includes('hidden'));
 // Embedded solutions use the same context while both private editors stay mounted.
 currentPost=basePost;embedded=true;await navigate('#/22?from=collections%2Fllm-basics');const practiceHash=location.hash;
 const editor=document.querySelector('[aria-label="Private practice code"]');editor.setSelectionRange(3,10,'backward');editor.scrollTop=51;editor.scrollLeft=17;
 await click(named('Share a solution'));const composer=document.querySelector('.community-composer'),field=composer.querySelector('textarea');await change(field,'PRIVATE_SOLUTION_DRAFT_KEEP');field.setSelectionRange(2,8,'backward');field.scrollTop=73;field.scrollLeft=11;const storage=JSON.stringify({...sessionStorage});
 await change(document.querySelector('.community-period select'),'30');await change(document.querySelector('.community-sort select'),'top');await apply('masked_attention');assert.equal(location.hash,practiceHash);assert(document.querySelector('.solution-reading-card .post-title mark'));assert(document.querySelector('.post-excerpt mark'));assert.equal(nav.readCommunityFeed(nav.communityPostReturn(document.querySelector('.solution-reading-page').getAttribute('href'),basePost.id)).state.days,'30');
 await click(document.querySelector('.solution-reading-card summary mark'));await settle(()=>document.querySelector('.solution-reading-content .discussion-reading'));const detail=document.querySelector('.solution-reading-card details');assert(detail.open);const beforeBack=requests.length;await click(document.querySelector('.solution-reading-code'));assert.equal(requests.length,beforeBack);assert.equal(document.activeElement,editor);assert(detail.open);
 await click(document.querySelector('.solution-reading-collapse'));assert(!detail.open);assert(document.querySelector('.post-excerpt mark'));assert.equal(document.activeElement,document.querySelector('.solution-reading-card summary'));
 currentPost={...basePost,body:'```python\ndef masked_attention():\n    return 1\n```'};await apply('MASKED_attention');assert.equal(document.querySelector('.post-excerpt').getAttribute('data-search-context'),'code');assert.equal(document.querySelector('.post-search-origin').textContent,'Code excerpt');assert(!document.querySelector('.solution-reading-card').querySelector('.highlight,pre,table,.katex'));
 const beforeChinese=requests.length;await click(document.querySelector('#search-preview-locale'));assert.equal(requests.length,beforeChinese);assert.equal(document.querySelector('.post-search-origin').textContent,'代码片段');await click(document.querySelector('#search-preview-locale'));
 await click(document.querySelector('.community-search-clear'));assert(!document.querySelector('mark'));assert.equal(requests.at(-1).params.get('q'),'');assert.equal(requests.at(-1).params.get('days'),'30');assert.equal(requests.at(-1).params.get('sort'),'top');assert.equal(location.hash,practiceHash);
 assert.equal(document.querySelector('.community-composer'),composer);assert.equal(composer.querySelector('textarea'),field);assert.equal(field.value,'PRIVATE_SOLUTION_DRAFT_KEEP');assert.equal(JSON.stringify({...sessionStorage}),storage);assert.equal(document.querySelector('[aria-label="Private practice code"]'),editor);assert.deepEqual([editor.value,editor.selectionStart,editor.selectionEnd,editor.selectionDirection,editor.scrollTop,editor.scrollLeft],['PRIVATE_IMPLEMENTATION_KEEP',3,10,'backward',51,17]);assert.deepEqual([field.selectionStart,field.selectionEnd,field.selectionDirection,field.scrollTop,field.scrollLeft],[2,8,'backward',73,11]);
 hold=true;await apply('old_query');const old=pending.shift();currentPost={...basePost,title:'new_query result',body:'The new_query passage'};await apply('new_query');const fresh=pending.shift();assert(old.signal.aborted);await act(async()=>old.resolve({...response(old),items:[{...basePost,title:'OLD_PRIVATE_COHORT'}]}));assert(!document.body.textContent.includes('OLD_PRIVATE_COHORT'));await act(async()=>fresh.resolve(response(fresh)));assert.equal(document.querySelector('.post-title mark').textContent,'new_query');
 await apply('outgoing_query');const outgoing=pending.shift();hold=false;account.user={...user,id:'bob',username:'bob'};await render();assert(outgoing.signal.aborted);await act(async()=>outgoing.resolve({...response(outgoing),items:[{...basePost,title:'OUTGOING_OWNER_PRIVATE'}]}));assert(!document.body.textContent.includes('OUTGOING_OWNER_PRIVATE'));assert(!document.querySelector('.community-composer'));assert.equal(mutations.length,0);assert(!JSON.stringify(requests.map(row=>row.path)).includes('PRIVATE_IMPLEMENTATION_KEEP'));assert(!JSON.stringify(requests.map(row=>row.path)).includes('PRIVATE_SOLUTION_DRAFT_KEEP'));
 console.log('PASS community search presentation: real applied-only title/body context highlights, source-safe fallback and native marked-link reading round trip, retained date/sort/saved scope, deferred in-place reading and solution/back-to-code actions, bilingual controls, private draft/editor/caret/both-scroll preservation, no social writes or detail prefetch, and stale query/owner isolation');
}finally{if(root)await act(async()=>root.unmount());await server.close();await rm(cacheDir,{recursive:true,force:true});dom.window.close();for(const[key,value]of previous)if(value)Object.defineProperty(globalThis,key,value);else delete globalThis[key];}
