import assert from 'node:assert/strict';
import React,{act} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/community',pretendToBeVisual:true});
dom.window.HTMLElement.prototype.scrollIntoView=function(){};
const globals={window:dom.window,document:dom.window.document,navigator:dom.window.navigator,HTMLElement:dom.window.HTMLElement,localStorage:dom.window.localStorage,sessionStorage:dom.window.sessionStorage,location:dom.window.location,IS_REACT_ACT_ENVIRONMENT:true,__formatFixture:undefined};
const previous=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});
localStorage.setItem('paperleet.locale','en');
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{
 name:'format-account-fixture',enforce:'pre',resolveId(source){if(source==='./social')return '\0format-account';},load(id){if(id==='\0format-account')return `import React from 'react';export const useAccount=()=>globalThis.__formatFixture.account;export const socialApi=(...args)=>globalThis.__formatFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error{};export const Avatar=()=>null;export const PersonLink=({user})=>React.createElement('span',null,user.display_name);export const DateLabel=({value})=>React.createElement('time',null,value);export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);export const TestAccountBadge=()=>null;export const profileHref=user=>'#/profile/'+user.username;`;}
}]});
let root;
try{
 const {createRoot}=await import('react-dom/client');
 const {formatMarkdown}=await server.ssrLoadModule('/src/markdownFormatting.ts');
 const {default:Editor}=await server.ssrLoadModule('/src/MarkdownEditor.tsx');
 const {LocaleProvider,useI18n}=await server.ssrLoadModule('/src/i18n.tsx');
 const {default:Markdown}=await server.ssrLoadModule('/src/CommunityMarkdown.tsx');
 for(const english of [true,false])for(const format of ['bold','italic','code','codeblock','list','link','math']){
  const text='  中文 😀\nnotes `literal`\nend  ',result=formatMarkdown(text,format,2,7,20000,english);
  assert(result&&result.before===text&&result.start<=result.end);
  assert.equal(formatMarkdown(text,format,2,7,text.length,english),undefined);
  const empty=formatMarkdown('',format,0,0,20000,english);assert(empty.after.slice(empty.start,empty.end));
 }
 assert.equal(formatMarkdown('😀','bold',1,1,99).after,'**😀**');
 assert.equal(formatMarkdown('before\r\nline\r\nafter','bold',8,12,99).after,'before\r\n**line**\r\nafter');
 assert.equal(formatMarkdown('\nnext','list',0,0,99,true).after,'- list item\nnext','Do not consume the next untouched line');
 assert.equal(formatMarkdown('a\nb\nc','list',0,4,99,true).after,'- a\n- b\nc');
 assert.equal(formatMarkdown('a\r\nb','list',0,4,99,true).after,'- a\r\n- b');
 assert.equal(formatMarkdown('text','bold',NaN,0,99),undefined);
 const structured=[
  ['heading','Alpha 😀\nBeta','## Alpha 😀\n## Beta'],
  ['heading','# Alpha\n### Beta','## Alpha\n## Beta'],
  ['heading','## Alpha\n## Beta','Alpha\nBeta'],
  ['heading','Alpha\n\nBeta','## Alpha\n\n## Beta'],
  ['ordered','- First\n4) Second','1. First\n2. Second'],
  ['ordered','First\n\nSecond','1. First\n\n2. Second'],
  ['ordered','1. First\n2. Second','First\nSecond'],
  ['quote','Alpha\n\nBeta','> Alpha\n>\n> Beta'],
  ['quote','> Alpha\n>\n> Beta','Alpha\n\nBeta'],
  ['quote','>> Nested\n> Plain','> Nested\nPlain'],
  ['quote','Alpha> ','> Alpha> '],
 ];
 for(const [kind,original,expected] of structured){
  const edit=formatMarkdown(original,kind,0,original.length,20000,true);
  assert.equal(edit.after,expected,kind+': '+JSON.stringify(original));assert.equal(edit.after.slice(edit.start,edit.end),expected);
  assert.equal(formatMarkdown(original,kind,original.length,0,20000,true).after,expected,'Reverse selections use the same complete lines');
 }
 for(const english of [true,false])for(const kind of ['heading','ordered','quote']){
  const empty=formatMarkdown('',kind,0,0,20000,english);assert(empty.after.slice(empty.start,empty.end));
  assert.equal(formatMarkdown('Alpha',kind,0,5,5,english),undefined,'Structural edits obey the same publication length limit');
  assert.equal(formatMarkdown('Alpha',kind,Infinity,5,99,english),undefined);
  assert.equal(formatMarkdown('Alpha',kind,0,5,-1,english),undefined);
  const isolated=formatMarkdown('Before\nAlpha\nAfter',kind,8,10,20000,english);
  const originalNode=new JSDOM(renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(Markdown,{body:isolated.after}))));
  const paragraphs=[...originalNode.window.document.querySelector('.community-prose').children];assert.equal(paragraphs[0].outerHTML,'<p>Before</p>');assert.equal(paragraphs.at(-1).outerHTML,'<p>After</p>','Unselected paragraphs stay outside the new block');originalNode.window.close();
 }
 for(const eol of ['\n','\r\n'])for(const [kind,prefix] of [['heading','## '],['ordered','1. '],['quote','> ']]){
  const original=['Before','Alpha 😀','Beta','After'].join(eol),start=original.indexOf('Alpha')+2,end=original.indexOf('After');
  const edit=formatMarkdown(original,kind,start,end,20000,true),second=kind==='ordered'?'2. ':prefix;
  assert.equal(edit.after,['Before','',prefix+'Alpha 😀',second+'Beta','','After'].join(eol));
  assert.equal(edit.after.slice(edit.start,edit.end),[prefix+'Alpha 😀',second+'Beta'].join(eol),'Selection excludes separator lines');
  const back=formatMarkdown(edit.after,kind,edit.start,edit.end,20000,true);
  assert.equal(back.after,['Before','','Alpha 😀','Beta','','After'].join(eol),'A second click removes one block style without accumulating separators');
 }
 const heading=formatMarkdown('A clear approach','heading',4,8,999,true),steps=formatMarkdown('Read\nImplement\nCheck','ordered',0,20,999,true),quote=formatMarkdown('A source\n\nAnother paragraph','quote',0,29,999,true);
 const structuredHtml=renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(Markdown,{body:[heading.after,steps.after,quote.after].join('\n\n'),headingScope:'reading-'+ 'd'.repeat(32)})));
 const structuredDom=new JSDOM(structuredHtml);assert.equal(structuredDom.window.document.querySelector('h2').textContent,'A clear approach');assert.deepEqual([...structuredDom.window.document.querySelectorAll('ol>li')].map(node=>node.textContent),['Read','Implement','Check']);assert.equal(structuredDom.window.document.querySelectorAll('blockquote').length,1);assert.equal(structuredDom.window.document.querySelectorAll('blockquote>p').length,2);
 const {markdownOutline}=await server.ssrLoadModule('/src/markdownOutline.ts');assert.equal(markdownOutline(heading.after,'reading-'+ 'd'.repeat(32))[0].id,structuredDom.window.document.querySelector('h2').id);structuredDom.window.close();
 const code=formatMarkdown('return ``` + `value`','codeblock',0,21,999,true);assert(code.after.startsWith('````python\n'));assert(code.after.includes('return ``` + `value`'));
 const codeHtml=renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(Markdown,{body:code.after})));
 assert(codeHtml.includes('<pre>')&&codeHtml.includes('return ``` + `value`'),'Inserted fences render as a single literal code block');
 assert.equal(formatMarkdown('`x`','code',0,3,99).after,'`` `x` ``');
 assert.equal(formatMarkdown(' x ','code',0,3,99).after,'`  x  `');
 const link=formatMarkdown('[literal] \\','link',0,11,999,true);assert.equal(link.after.slice(link.start,link.end),'https://');assert(link.after.startsWith('[\\[literal\\] \\\\]('));
 let lastComponent,lastProps;
 const render=async(Component,props={})=>act(async()=>{lastComponent=Component;lastProps=props;root??=createRoot(document.querySelector('#root'));root.render(React.createElement(LocaleProvider,null,React.createElement(Component,props)));});
 const stop=async()=>{if(root)await act(async()=>root.unmount());root=undefined;};
 const click=async element=>{assert(element,'Missing control');await act(async()=>element.click());};
 const named=name=>[...document.querySelectorAll('button')].find(element=>(element.getAttribute('aria-label')||element.textContent.trim())===name);
 const input=async(element,value)=>act(async()=>{assert(element);Object.getOwnPropertyDescriptor(dom.window[element.tagName==='TEXTAREA'?'HTMLTextAreaElement':'HTMLInputElement'].prototype,'value').set.call(element,value);element.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});
 let setText,submitted=0;
 function Harness({disabled=false,limit=20000}){
  const [value,setValue]=React.useState('Notes 😀\nNext line'),[preview,setPreview]=React.useState(false);setText=setValue;
  const {setLocale}=useI18n();
  return React.createElement('form',{onSubmit:event=>{event.preventDefault();submitted++;}},React.createElement('button',{type:'button',id:'outside',onClick:()=>setLocale('zh')},'Outside'),React.createElement(Editor,{value,onChange:setValue,preview,onPreviewChange:setPreview,label:'Body',modeLabel:'Editor mode',maxLength:limit,disabled}));
 }
 await render(Harness);let field=document.querySelector('textarea');field.focus();field.setSelectionRange(0,5);field.scrollTop=48;
 await click(named('Bold'));assert.equal(field.value,'**Notes** 😀\nNext line');assert.equal(document.activeElement,field);assert.equal(field.selectionStart,2);assert.equal(field.selectionEnd,7);assert.equal(field.scrollTop,48);assert.equal(submitted,0);
 await click(named('Undo formatting'));assert.equal(field.value,'Notes 😀\nNext line');assert.equal(field.selectionStart,0);assert.equal(field.selectionEnd,5);assert.equal(field.scrollTop,48);
 await click(named('Preview'));assert.equal(document.querySelector('textarea'),field);assert(field.closest('label').hidden);assert(document.querySelector('[aria-label="Content preview"]').textContent.includes('Notes'));
 await click(named('Write'));assert.equal(document.activeElement,field);assert.equal(field.selectionEnd,5);assert.equal(field.scrollTop,48);assert(!field.closest('label').hidden);
 await act(async()=>field.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'b',ctrlKey:true,bubbles:true,cancelable:true})));assert.equal(field.value,'**Notes** 😀\nNext line');
 const undo=named('Undo formatting');undo.focus();await act(async()=>{undo.click();undo.blur();});assert.equal(document.activeElement,field,'Keyboard undo restores editing when its button becomes disabled');
 await act(async()=>field.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'b',metaKey:true,bubbles:true,cancelable:true})));
 const composed=field.value;await act(async()=>field.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'b',ctrlKey:true,isComposing:true,bubbles:true,cancelable:true})));assert.equal(field.value,composed,'IME composition is never formatted');
 await input(field,field.value+' later');assert(named('Undo formatting').disabled);const later=field.value;await click(named('Undo formatting'));assert.equal(field.value,later);
 await render(Harness,{disabled:true});assert([...document.querySelectorAll('.markdown-editor button')].every(button=>button.disabled));await act(async()=>field.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'k',metaKey:true,bubbles:true,cancelable:true})));assert.equal(field.value,later);
 await render(Harness,{limit:field.value.length});await click(named('Display math'));assert.equal(field.value,later);assert(document.querySelector('.markdown-format-error'));
 await render(Harness);await act(async()=>setText('exact source'));field.focus();field.setSelectionRange(0,5);
 await act(async()=>{named('Bold').click();document.querySelector('#outside').focus();});assert.equal(document.activeElement,document.querySelector('#outside'));assert.equal(field.value,'**exact** source');await click(document.querySelector('#outside'));assert(document.body.textContent.includes('撤销格式'));for(const name of ['小标题','编号步骤','引用'])assert(named(name));await stop();localStorage.setItem('paperleet.locale','en');
 const owner='alice',postId='a'.repeat(32),replyId='b'.repeat(32),version='c'.repeat(64);
 const user={can_publish_community:true,id:owner,username:owner,display_name:'Alice',avatar_color:'#48766b',affiliation:'student',institution:'Example'};
 const post={id:postId,title:'Actual discussion',body:'Existing post',category:'questions',problem:null,problem_id:null,created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',edit_version:version,author:user,counts:{likes:0,reposts:0,replies:0},viewer:{can_edit:true,can_delete:true},replies:[]};
 const reply={id:replyId,post_id:postId,parent_id:null,body:'Existing reply',author:user,can_edit:true,edit_version:version},mutations=[];
 globalThis.__formatFixture={account:{user,loading:false,capabilities:{community_bookmarks:true,community_solutions:true},refresh:async()=>{}},request:async(path,payload)=>{if(payload){mutations.push({path,payload});throw Error('No publication during formatting');}return path.includes('?')?{items:[],total:0,solutions:true}:post;}};
 const {default:Community}=await server.ssrLoadModule('/src/Community.tsx'),{default:Interview}=await server.ssrLoadModule('/src/InterviewComposer.tsx');
 const {default:PostEditor}=await server.ssrLoadModule('/src/PostEditor.tsx'),{default:ReplyEditor}=await server.ssrLoadModule('/src/ReplyEditor.tsx');
 const {default:CardActions}=await server.ssrLoadModule('/src/CommunityCardActions.tsx');
 const drafts=await server.ssrLoadModule('/src/communityDrafts.ts'),replies=await server.ssrLoadModule('/src/replyDrafts.ts'),interviews=await server.ssrLoadModule('/src/interviewDrafts.ts'),postEdits=await server.ssrLoadModule('/src/communityEdits.ts'),replyEdits=await server.ssrLoadModule('/src/replyEdits.ts');
 for(const kind of ['post','solution','reply','interview','edit-post','edit-reply','inline-reply']){
  sessionStorage.clear();
  if(kind==='post'){await render(Community,{route:'community?category=questions'});await click(named('New post'));}
  if(kind==='solution'){await render(Community,{route:'community?solutions=1',topicId:'architecture/attention',topic:{id:'architecture/attention',number:1,title:'注意力',title_en:'Attention'}});await click(named('Share a solution'));}
  if(kind==='reply')await render(Community,{postId,route:'community/'+postId});
  if(kind==='interview')await render(Interview,{owner,company:'google',problems:[],onClose(){},onPublished(){}});
  if(kind==='edit-post')await render(PostEditor,{owner,post,disabled:false,onSaved(){},onClose(){},onRefresh:async()=>true});
  if(kind==='edit-reply')await render(ReplyEditor,{owner,reply,disabled:false,onSaved(){},onClose(){},onRefresh:async()=>true});
  if(kind==='inline-reply'){await render(CardActions,{post});await click(named('Write comment'));}
  field=document.querySelector('.markdown-editor textarea');assert(field,kind);assert.equal(field.getAttribute('aria-keyshortcuts'),'Control+Enter Meta+Enter',kind+' opts in to native validated shortcut submission');assert(field.getAttribute('aria-describedby').split(' ').some(id=>document.getElementById(id)?.classList.contains('markdown-submit-shortcut')));await input(field,'A concrete reflection 😀');field.focus();field.setSelectionRange(2,10);field.scrollTop=17;
  await click(named('Bold'));assert.equal(field.value,'A **concrete** reflection 😀');
  const readDraft=()=>kind==='post'?drafts.readPostDraft(sessionStorage,owner,'general'):kind==='solution'?drafts.readPostDraft(sessionStorage,owner,'architecture/attention:solutions'):['reply','inline-reply'].includes(kind)?replies.readReplyDraft(sessionStorage,owner,postId):kind==='interview'?interviews.readInterviewDraft(sessionStorage,owner):kind==='edit-post'?postEdits.readPostEdit(sessionStorage,owner,postId):replyEdits.readReplyEdit(sessionStorage,owner,replyId);
  const draft=readDraft();
  assert.equal(draft.body,field.value,'Existing scoped draft path: '+kind);
  const beforeLive=JSON.stringify({...sessionStorage});field.scrollLeft=9;
  await click(named('Live preview'));assert.equal(document.querySelector('.markdown-editor textarea'),field);assert(!field.closest('label').hidden);assert(document.querySelector('.markdown-editor.has-live-preview .compose-preview').textContent.includes('concrete'));assert.equal(field.selectionStart,4);assert.equal(field.selectionEnd,12);assert.equal(field.scrollTop,17);assert.equal(field.scrollLeft,9);assert.equal(JSON.stringify({...sessionStorage}),beforeLive,'Changing the reading layout does not rewrite the '+kind+' draft');
  await click(named('Preview'));await click(named('Write'));assert.equal(document.querySelector('textarea'),field);assert.equal(field.selectionStart,4);assert.equal(field.selectionEnd,12);assert.equal(field.scrollTop,17);
  await click(named('Undo formatting'));assert.equal(field.value,'A concrete reflection 😀');assert.equal(mutations.length,0);
  field.focus();field.setSelectionRange(2,10,'backward');field.scrollTop=17;field.scrollLeft=9;
  const tablePicker=document.querySelector('.markdown-table-picker');await click(tablePicker.querySelector('summary'));assert(tablePicker.open);await click([...tablePicker.querySelectorAll('button')].find(button=>button.textContent.trim()==='3 columns'));
  assert.equal(document.querySelector('.markdown-editor textarea'),field);assert.equal(document.activeElement,field);assert.equal(readDraft().body,field.value,kind+' retains its inserted table in the existing scoped draft');assert(field.value.endsWith('concrete reflection 😀'));
  await click(named('Preview'));assert.equal(document.querySelectorAll('.compose-preview th').length,3);assert.equal(document.querySelectorAll('.compose-preview tbody>tr').length,2);await click(named('Write'));await click(named('Undo formatting'));
  assert.equal(field.value,'A concrete reflection 😀');assert.equal(readDraft().body,field.value);assert.deepEqual([field.selectionStart,field.selectionEnd,field.selectionDirection,field.scrollTop,field.scrollLeft],[2,10,'backward',17,9]);assert.equal(mutations.length,0);
  for(const [tool,prefix,selector] of [['Section heading','## ','h2'],['Numbered steps','1. ','ol>li'],['Block quote','> ','blockquote>p']]){
   field.focus();field.setSelectionRange(2,10,'backward');field.scrollTop=17;field.scrollLeft=9;
   named(tool).focus();await click(named(tool));assert.equal(document.querySelector('.markdown-editor textarea'),field);assert.equal(field.value,prefix+'A concrete reflection 😀');assert.equal(document.activeElement,field);assert.equal(field.scrollTop,17);assert.equal(field.scrollLeft,9);
   const currentDraft=readDraft();
   assert.equal(currentDraft.body,field.value,kind+' saves '+tool+' in its existing draft');
   await click(named('Preview'));assert.equal(document.querySelector('.compose-preview '+selector).textContent,'A concrete reflection 😀');assert(field.closest('label').hidden);assert(document.querySelector('.markdown-format-tools').hidden);
   await click(named('Write'));named('Undo formatting').focus();await click(named('Undo formatting'));assert.equal(field.value,'A concrete reflection 😀');assert.equal(document.activeElement,field);assert.equal(field.selectionStart,2);assert.equal(field.selectionEnd,10);assert.equal(field.selectionDirection,'backward');assert.equal(field.scrollTop,17);assert.equal(field.scrollLeft,9);assert.equal(mutations.length,0);
  }
  if(['post','solution','interview'].includes(kind))await input(document.querySelector(`input[maxlength="${kind==='interview'?180:160}"]`),'A complete publishing title');
  const form=field.closest('form'),publish=form.querySelector('button.button:not([type="button"])');
  assert(publish&&!publish.disabled,kind+' starts with an eligible publish control');
  user.can_publish_community=false;await render(lastComponent,lastProps);
  assert(document.querySelector('.publishing-affiliation-notice'),kind+' explains affiliation verification');
  assert.equal(document.querySelector('textarea'),field,kind+' preserves the mounted draft editor');
  assert(!field.disabled,kind+' can keep editing an unpublished draft');
  assert(publish.disabled,kind+' blocks publication and edits without affiliation');
  await act(async()=>form.dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
  await act(async()=>field.dispatchEvent(new dom.window.KeyboardEvent('keydown',{key:'Enter',metaKey:true,bubbles:true,cancelable:true})));
  assert.equal(mutations.length,0,kind+' protects direct form submission and the keyboard shortcut');
  assert.equal(field.value,'A concrete reflection 😀');
  assert.equal(document.querySelector('.publishing-affiliation-actions a').getAttribute('href'),'#/account');
  user.can_publish_community=true;await render(lastComponent,lastProps);
  assert(!document.querySelector('.publishing-affiliation-notice'));assert(!publish.disabled,kind+' resumes after account verification');
  await stop();
 }
 const receipts=await server.ssrLoadModule('/src/publicationRequests.ts');
 const submitForm=async()=>act(async()=>document.querySelector('.community-composer form').dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true})));
 for(const [category,label,hint] of [['admissions','School','MS / PhD'],['career','Job search','Internships']]){
  sessionStorage.clear();location.hash='#/community?category='+category;
  await render(Community,{route:'community?category='+category});await click(named('New post'));
  let categorySelect=document.querySelector('.community-composer select');
  assert.equal(categorySelect.value,category,'The selected board is the initial publication category');
  assert.equal(categorySelect.selectedOptions[0].textContent,label);
  assert(document.getElementById(categorySelect.getAttribute('aria-describedby')).textContent.includes(hint));
  await input(document.querySelector('input[maxlength="160"]'),'A concrete '+category+' question');
  await input(document.querySelector('.markdown-editor textarea'),'My actual experience and specific question.');
  await submitForm();
  const attempt=mutations.at(-1);assert.equal(attempt.path,'community/posts');assert.equal(attempt.payload.category,category);
  assert.equal(drafts.readPostDraft(sessionStorage,owner,'general').category,category);
  await stop();await render(Community,{route:'community?category='+category});await click(named('Resume draft'));
  assert.equal(document.querySelector('.community-composer select').value,category,'A failed post restores its board with the draft');
  assert(document.querySelector('.publication-recovery'));
  await submitForm();
  assert.equal(mutations.at(-1).payload.request_id,attempt.payload.request_id,'Retrying the restored board publication keeps its original request ID');
  receipts.clearPublication(sessionStorage,receipts.readPublication(sessionStorage,owner,'post','general').receipt);
  await stop();sessionStorage.clear();

  await render(PostEditor,{owner,post:{...post,category},disabled:false,onSaved(){},onClose(){},onRefresh:async()=>true});
  categorySelect=document.querySelector('.community-composer select');assert.equal(categorySelect.value,category);
  assert(document.getElementById(categorySelect.getAttribute('aria-describedby')).textContent.includes(hint));
  const other=category==='admissions'?'career':'admissions';
  await act(async()=>{categorySelect.value=other;categorySelect.dispatchEvent(new dom.window.Event('change',{bubbles:true}));});
  assert.equal(postEdits.readPostEdit(sessionStorage,owner,postId).category,other);
  assert.equal(document.querySelector('.markdown-editor textarea').value,post.body,'Changing boards preserves the current edited body');
  await submitForm();assert.equal(mutations.at(-1).path,'community/posts/'+postId+'/edit');assert.equal(mutations.at(-1).payload.category,other);
  await stop();
 }
 console.log('PASS admissions/job-search authoring: selected labels and purpose hints, exact publication categories, failed-response draft/remount/idempotent recovery, and recategorized edits without losing content');
 console.log('PASS table writing surfaces: actual post, linked solution, interview, reply, inline reply and post/reply editors preserve their existing scoped drafts, render native GFM table rows in preview, restore backward selection and both scrolls on undo, and make no publication request');
 console.log('PASS Markdown editing: ten bilingual tools, whole-line heading/numbered-step/quote toggles with isolated prose and LF/CRLF, real rendered blocks/outline parity, exact source/Unicode/fence/length safety, shortcuts, mounted preview selection/both-scroll, backward-selection undo/focus and saved structural drafts across seven real authoring surfaces including linked solutions, affiliation/direct/shortcut publication gates and verified-account recovery');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
 for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,{...descriptor});else delete globalThis[key];}
}
