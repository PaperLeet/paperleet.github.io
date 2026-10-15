import assert from 'node:assert/strict';
import React,{act} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

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
 const {default:Card}=await server.ssrLoadModule('/src/InterviewReportCard.tsx');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {interviewOutlines,interviewOutlineText}=await server.ssrLoadModule('/src/interviewWriting.ts');
 const {communityPostReturn}=await server.ssrLoadModule('/src/communityNavigation.ts');
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
 await render(React.createElement(Preview,{body:'$$\nx=y\n$$'}));assert(document.querySelector('.post-excerpt').textContent.includes('打开阅读全文'));assert.equal(document.querySelectorAll('[data-content-kind="math"]').length,1);
 const filters={...emptyInterviewFilters(),saved:true,company:'google',role:'llm',stage:'technical',q:'reflection',page:3,days:'90',concept:'architecture/attention'},returnTo=interviewHref(filters);
 const post={id:'a'.repeat(32),title:'A useful reflection',body:rich,category:'career',created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',author:{id:'owner',username:'owner',display_name:'Author',avatar_color:'#48766b'},counts:{replies:3,likes:5,reposts:1},viewer:{bookmarked:false},interview:{company:'google',role:'llm',stage:'technical',outcome:'unknown',interview_month:'',location:'',questions:['architecture/attention']}};
 await render(React.createElement(Card,{post,owner:'owner',canSave:false,returnTo,savedView:false,onReload(){}}));
 const card=document.querySelector('.interview-report-card'),title=card.querySelector('.interview-report-title');assert.equal(card.getAttribute('aria-labelledby'),title.id);assert.equal(communityPostReturn(title.getAttribute('href'),post.id),returnTo,'Refined previews preserve exact saved/search/company/page returns');
 assert.equal(card.querySelector('.post-excerpt').textContent,result.text);assert(!card.querySelector('.post-excerpt').textContent.includes('Interview process'));assert.equal(card.querySelectorAll('.post-content-signals > li').length,3);assert.equal(card.querySelector('.interview-report-actions > a').href,title.href);assert(card.querySelector('.interview-report-footer').textContent.includes('3'));
 await server.close();serverClosed=true;
 failingServer=await createServer({ssr:{noExternal:['unified']},server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'failing-preview-parser',enforce:'pre',resolveId(source){if(source==='unified')return '\0failing-preview-parser';},load(id){if(id==='\0failing-preview-parser')return 'export const unified=()=>{const parser={use(){return parser},freeze(){return parser},parse(){throw new Error("Parser failed")}};return parser}';}}]});
 const failedModel=await failingServer.ssrLoadModule('/src/postExcerpt.ts');assert.equal(failedModel.postPreview('Kept only in the full post').unavailable,true,'A parser failure cannot crash the surrounding feed');
 const {default:FailedPreview}=await failingServer.ssrLoadModule('/src/PostPreview.tsx');const {LocaleProvider:FailedLocale}=await failingServer.ssrLoadModule('/src/i18n.tsx');localStorage.setItem('paperleet.locale','en');const failedMarkup=renderToStaticMarkup(React.createElement(FailedLocale,null,React.createElement(FailedPreview,{body:'Kept only in the full post'})));assert(failedMarkup.includes('The preview is unavailable. Open the post to keep reading.'));assert(!failedMarkup.includes('post-content-signals'));
 console.log('PASS post previews: narrative-only Markdown/GFM/math extraction, outline-free summaries, literal inline code, link labels without tracking destinations, raw HTML/image exclusion, exact content badges, bounded Unicode grapheme truncation, bilingual fallback/update isolation, per-post parser recovery and unchanged scoped reading links');
}finally{
 if(root)await act(async()=>root.unmount());if(!serverClosed)await server.close();if(failingServer)await failingServer.close();dom.window.close();for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
}
