import assert from 'node:assert/strict';
import React,{act} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
import {readFile} from 'node:fs/promises';

const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/'});
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
let root,failingServer,serverClosed=false;
try{
 const {postPreview}=await server.ssrLoadModule('/src/postExcerpt.ts');
 const {default:Preview}=await server.ssrLoadModule('/src/PostPreview.tsx');
 const {default:InlineBody}=await server.ssrLoadModule('/src/InlineCommentBody.tsx');
 const {default:Card}=await server.ssrLoadModule('/src/InterviewReportCard.tsx');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {interviewOutlines,interviewOutlineText}=await server.ssrLoadModule('/src/interviewWriting.ts');
 const {communityPostReturn}=await server.ssrLoadModule('/src/communityNavigation.ts');
 const {libraryReturnHref}=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const {emptyInterviewFilters,interviewHref}=await server.ssrLoadModule('/src/interviewNavigation.ts');
 for(const id of Object.keys(interviewOutlines))for(const locale of ['en','zh']){
  const outline=interviewOutlineText(id,locale);
  assert.equal(postPreview(outline).text,'','Blank writing outlines never become a misleading card summary');
  const body=outline.replace('\n\n','\n\nI clarified the input, tested empty batches, and revised the design.\n\n');
  assert.equal(postPreview(body).text,'I clarified the input, tested empty batches, and revised the design.');
 }
 const rich='## Interview process\n\nFirst **clarify** the goal and use `a*b` without losing operators. Read [the source][paper].\n\n[paper]: https://arxiv.org/abs/1706.03762 "Paper title"\n\n```python\nprivate_debug_code()\n```\n\n$$\n\\Delta W = \\frac{a}{r}BA\n$$\n\n| Metric | Value |\n| --- | --- |\n| Accuracy | 0.8 |\n\n## Reflection\n\n- [x] Verify the edge cases.\n> Keep assumptions explicit.\n';
 const result=postPreview(rich,400);
 assert.equal(result.text,'First clarify the goal and use a*b without losing operators. Read the source. Verify the edge cases. Keep assumptions explicit.');
 assert.deepEqual(result.kinds,['code','math','table']);assert.equal(result.truncated,false);
 assert.equal(postPreview('Before $x^2$ after.',210,'[公式]').text,'Before [公式] after.');
 assert.deepEqual(postPreview('Use `$x^2$` and `a_b * c`.').kinds,[]);assert.equal(postPreview('Use `$x^2$` and `a_b * c`.').text,'Use $x^2$ and a_b * c.');
 assert.equal(postPreview('Cost $5.').text,'Cost $5.');assert.equal(postPreview('Keep an unclosed $literal.').text,'Keep an unclosed $literal.');
 assert.equal(postPreview('https://tracker.invalid/full?secret=1\n\nwww.example.org/secret\n\n[Useful label](https://tracker.invalid/destination) and a [reference][r].\n\n[r]: https://tracker.invalid/reference').text,'Useful label and a reference.');
 assert.equal(postPreview('![private alt](https://tracker.invalid/pixel)\n\n<script>evil()</script>\n\nVisible <b>words</b> and &lt;literal&gt;.').text,'Visible words and <literal>.');
 assert.equal(postPreview('Heading only\n===\n\n---\n\n```\n```').text,'');assert.deepEqual(postPreview('```\n```').kinds,[]);
 for(const body of ['```js\ncodeOnly()\n```','$$\nx = y\n$$','| A | B |\n| - | - |\n| 1 | 2 |'])assert.equal(postPreview(body).text,'','Non-narrative posts use a truthful full-content prompt');
 assert.equal(postPreview('你好'.repeat(150),210).text,'你好'.repeat(105)+'…');
 const family='👨‍👩‍👧‍👦',accent='e\u0301';
 assert.equal(postPreview('ab'+family+accent+' tail',3).text,'ab'+family+'…','Truncation keeps a complete emoji grapheme');
 assert.equal(postPreview('ab'+accent+' tail',3).text,'ab'+accent+'…','Combining accents stay with their base character');
 assert.equal(postPreview('四个字好',4).text,'四个字好');assert.equal(postPreview('line one\nline two').text,'line one line two');
 assert.equal(postPreview('A plain reply.').rich,false);assert.equal(postPreview('一行\n自然换行').rich,true,'Whitespace preserved by the full renderer remains available inline');assert.equal(postPreview('Aligned  words').rich,true);
 for(const body of ['**emphasis**','[source](https://example.org)','`x * y`','## Heading','- First\n- Second','> Context','First\n\nSecond','![alt](https://example.org/image)','A hard  \nbreak'])assert.equal(postPreview(body).rich,true,'Formatted replies must retain an in-place full reading option: '+body);
 const bounded=postPreview('x'.repeat(20000)+'SECRET_AFTER_LIMIT',400);assert(!bounded.text.includes('SECRET_AFTER_LIMIT'));assert.equal(bounded.text.length,401);assert(bounded.truncated);
 const original=rich;postPreview(rich);assert.equal(rich,original);
 const staticPreview=body=>renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(Preview,{body})));
 const hostile=staticPreview('<script>evil()</script>\n\n![pixel](https://tracker.invalid/pixel)\n\n&lt;img src=x onerror=evil()&gt;');
 assert(!/<script|<img|href="https:\/\/tracker|src="https:\/\/tracker/i.test(hostile));assert(hostile.includes('&lt;img'),'Literal text remains escaped, not interpreted as HTML');
 const {createRoot}=await import('react-dom/client');
 function LocaleControl(){const {setLocale,locale}=useI18n();return React.createElement('button',{onClick:()=>setLocale(locale==='en'?'zh':'en')},'Change locale');}
 const render=async children=>act(async()=>{root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(LocaleControl),children));});
 await render(React.createElement(Preview,{body:rich}));assert.equal(document.querySelectorAll('.post-content-signals > li').length,3);assert.equal(document.querySelector('.post-content-signals').getAttribute('aria-label'),'Post contains');assert(!document.querySelector('pre,table,.katex,iframe,img'));
 await act(async()=>document.querySelector('button').click());assert.equal(document.querySelector('.post-content-signals').getAttribute('aria-label'),'内容包含');assert(document.querySelector('.post-content-signals').textContent.includes('代码片段'));
 await render(React.createElement(Preview,{body:'Before $x$ after.'}));assert(document.querySelector('.post-excerpt').textContent.includes('[公式]'));await act(async()=>document.querySelector('button').click());assert(document.querySelector('.post-excerpt').textContent.includes('[formula]'),'Changing language regenerates math placeholders without changing the post');await act(async()=>document.querySelector('button').click());
 await render(React.createElement(Preview,{body:'真正的新正文。'}));assert.equal(document.querySelector('.post-excerpt').textContent,'真正的新正文。');assert(!document.querySelector('.post-content-signals'),'Content updates cannot retain stale badges');
 // Inline reply reading reuses the bounded model but mounts the full renderer only on demand.
 const inlineStyle=document.createElement('style');inlineStyle.textContent=await readFile(new URL('../src/post-preview.css',import.meta.url),'utf8')+'\n'+await readFile(new URL('../src/community-card-actions.css',import.meta.url),'utf8');document.head.append(inlineStyle);
 const short='完整中文回复'.repeat(40);
 await render(React.createElement(InlineBody,{body:short}));assert.equal(document.querySelector('.post-excerpt').textContent,short);assert(!document.querySelector('.inline-comment-toggle'));assert.equal(window.getComputedStyle(document.querySelector('.post-excerpt')).display,'block','An untruncated reply cannot still be clipped by the four-line preview rule');
 const emojiReply=family.repeat(300),click=async node=>act(async()=>{node.focus();node.click();});
 await render(React.createElement(InlineBody,{body:emojiReply}));let disclosure=document.querySelector('.inline-comment-toggle');assert.equal(disclosure.getAttribute('aria-expanded'),'false');assert.equal(document.querySelector('.post-excerpt').textContent,family.repeat(280)+'…');assert.equal(window.getComputedStyle(document.querySelector('.inline-comment-controls')).order,'1');
 await click(disclosure);assert.equal(disclosure.getAttribute('aria-expanded'),'true');assert.equal(document.querySelector('.community-prose').textContent,emojiReply);assert.equal(window.getComputedStyle(document.querySelector('.inline-comment-controls')).order,'0');assert.equal(document.activeElement,disclosure);
 let scrollOptions;disclosure.getBoundingClientRect=()=>({top:1000,bottom:1040});disclosure.scrollIntoView=options=>{scrollOptions=options;};await click(document.querySelector('.inline-comment-collapse'));assert.equal(document.activeElement,disclosure);assert.deepEqual(scrollOptions,{block:'nearest',behavior:'instant'},'Bottom collapse keeps the compact control reachable without a smooth long jump');assert(!document.querySelector('.community-prose'));
 const readers=body=>React.createElement(React.Fragment,null,React.createElement(InlineBody,{key:'first',body}),React.createElement(InlineBody,{key:'second',body:rich}));
 await render(readers(rich));const controls=[...document.querySelectorAll('.inline-comment-toggle')];assert.equal(new Set(controls.map(node=>node.getAttribute('aria-controls'))).size,2,'Identical replies in different cards need independent region IDs');for(const control of controls)assert(document.getElementById(control.getAttribute('aria-controls')));
 await click(controls[0]);assert.equal(document.querySelectorAll('.community-prose').length,1);assert.equal(controls[1].getAttribute('aria-expanded'),'false');assert(controls[0].textContent.includes('收起回复'));await click(document.querySelector('button'));assert(controls[0].textContent.includes('Show less'));assert.equal(controls[0].getAttribute('aria-expanded'),'true','Changing language retains the current reading position');
 await render(readers('Changed **reply**'));disclosure=document.querySelector('.inline-comment-toggle');assert.equal(disclosure.getAttribute('aria-expanded'),'false');assert(!document.querySelector('.community-prose'),'Updated content cannot inherit a previously opened body');await click(disclosure);await act(async()=>{const bottom=document.querySelector('.inline-comment-collapse');bottom.focus();bottom.click();location.hash='#/elsewhere';});assert.notEqual(document.activeElement,disclosure,'A route change cancels the focus/scroll handoff');location.hash='';
 await click(document.querySelector('button'));inlineStyle.remove();
 await render(React.createElement(Preview,{body:'$$\nx=y\n$$'}));assert(document.querySelector('.post-excerpt').textContent.includes('打开阅读全文'));assert.equal(document.querySelectorAll('[data-content-kind="math"]').length,1);
 const filters={...emptyInterviewFilters(),saved:true,company:'google',role:'llm',stage:'technical',q:'reflection',page:3,days:'90',concept:'architecture/attention'},returnTo=interviewHref(filters);
 const post={id:'a'.repeat(32),title:'A useful reflection',body:rich,category:'career',created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',author:{id:'owner',username:'owner',display_name:'Author',avatar_color:'#48766b'},counts:{replies:3,likes:5,reposts:1},viewer:{bookmarked:false},interview:{company:'google',role:'llm',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:['architecture/attention']}};
 await render(React.createElement(Card,{post,owner:'owner',canSave:false,returnTo,savedView:false,onReload(){}}));
 const card=document.querySelector('.interview-report-card'),title=card.querySelector('.interview-report-title');assert.equal(card.getAttribute('aria-labelledby'),title.id);assert.equal(communityPostReturn(title.getAttribute('href'),post.id),returnTo,'Refined previews preserve exact saved/search/company/page returns');
 assert.equal(card.querySelector('.post-excerpt').textContent,result.text);assert(!card.querySelector('.post-excerpt').textContent.includes('Interview process'));assert.equal(card.querySelectorAll('.post-content-signals > li').length,3);assert.equal(card.querySelector('.interview-report-actions > a').href,title.href);assert.equal(card.querySelectorAll('.post-card-counts.card-actions').length,1);assert(card.querySelector('.card-actions>button[aria-expanded]').textContent.includes('3'));assert(card.querySelector('.card-actions>button[aria-pressed]').textContent.includes('5'));
 const catalog=[{id:'architecture/attention',title:'注意力',translations:{en:{title:'Attention'}}},{id:'rl/gae',title:'优势估计',translations:{en:{title:'Advantage estimation'}}},{id:'rl/ppo_clip',title:'策略裁剪',translations:{en:{title:'Clipped objective'}}}];
 const relatedPost={...post,interview:{...post.interview,questions:['architecture/attention','rl/gae','rl/ppo_clip','retrieval/unavailable']}};
 await render(React.createElement(Card,{post:relatedPost,problems:catalog,owner:'owner',canSave:false,returnTo,savedView:false,onReload(){}}));
 const topics=document.querySelector('.interview-report-topics');assert(topics);assert(topics.textContent.includes('注意力'));assert.equal(topics.querySelectorAll('a').length,2,'Collapsed cards show two practice links and an in-place expansion');for(const link of topics.querySelectorAll('a'))assert.equal(libraryReturnHref(link.getAttribute('href')),returnTo);const expand=document.querySelector('.interview-report-practice-toggle');assert(expand.textContent.includes('2'));assert.equal(expand.getAttribute('aria-expanded'),'false');await act(async()=>expand.click());assert.equal(expand.getAttribute('aria-expanded'),'true');assert.equal(topics.querySelectorAll('a').length,3);assert(topics.textContent.includes('retrieval/unavailable'));assert(!topics.querySelector('a[href*="unavailable"]'));
 await act(async()=>document.querySelector('button').click());assert(document.querySelector('.interview-report-topics').textContent.includes('Attention'));assert.equal(document.querySelector('.interview-report-topics').getAttribute('aria-label'),'Practice linked to this experience');
 await render(React.createElement(Card,{post:relatedPost,problems:[],owner:'owner',canSave:false,returnTo,savedView:false,onReload(){}}));assert.equal(document.querySelectorAll('.interview-report-topics a').length,0,'Unavailable catalog data cannot create speculative practice destinations');assert.equal(document.querySelectorAll('.interview-report-practice-missing').length,4);assert(document.querySelector('.interview-report-topics').textContent.includes('Details pending'));
 await server.close();serverClosed=true;
 failingServer=await createServer({ssr:{noExternal:['unified']},server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'failing-preview-parser',enforce:'pre',resolveId(source){if(source==='unified')return '\0failing-preview-parser';},load(id){if(id==='\0failing-preview-parser')return 'export const unified=()=>{const parser={use(){return parser},freeze(){return parser},parse(){throw new Error("Parser failed")}};return parser}';}}]});
 const failedModel=await failingServer.ssrLoadModule('/src/postExcerpt.ts');assert.equal(failedModel.postPreview('Kept only in the full post').unavailable,true,'A parser failure cannot crash the surrounding feed');
 const {default:FailedPreview}=await failingServer.ssrLoadModule('/src/PostPreview.tsx');const {LocaleProvider:FailedLocale}=await failingServer.ssrLoadModule('/src/i18n.tsx');localStorage.setItem('paperleet.locale','en');const failedMarkup=renderToStaticMarkup(React.createElement(FailedLocale,null,React.createElement(FailedPreview,{body:'Kept only in the full post'})));assert(failedMarkup.includes('The preview is unavailable. Open the post to keep reading.'));assert(!failedMarkup.includes('post-content-signals'));
 console.log('PASS post previews: narrative-only Markdown/GFM/math extraction, outline-free summaries, literal inline code, link labels without tracking destinations, raw HTML/image exclusion, exact content badges, bounded Unicode grapheme truncation, bilingual fallback/update isolation, per-post parser recovery and unchanged scoped reading links');
}finally{
 if(root)await act(async()=>root.unmount());if(!serverClosed)await server.close();if(failingServer)await failingServer.close();dom.window.close();for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
