import assert from 'node:assert/strict';
import React,{act} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://paperleet.test/#/community',pretendToBeVisual:true});
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
 await act(async()=>{named('Bold').click();document.querySelector('#outside').focus();});assert.equal(document.activeElement,document.querySelector('#outside'));assert.equal(field.value,'**exact** source');await click(document.querySelector('#outside'));assert(document.body.textContent.includes('撤销格式'));await stop();localStorage.setItem('paperleet.locale','en');
 const owner='alice',postId='a'.repeat(32),replyId='b'.repeat(32),version='c'.repeat(64);
 const user={can_publish_community:true,id:owner,username:owner,display_name:'Alice',avatar_color:'#48766b',affiliation:'student',institution:'Example'};
 const post={id:postId,title:'Actual discussion',body:'Existing post',category:'questions',problem:null,problem_id:null,created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z',edit_version:version,author:user,counts:{likes:0,reposts:0,replies:0},viewer:{can_edit:true,can_delete:true},replies:[]};
 const reply={id:replyId,post_id:postId,parent_id:null,body:'Existing reply',author:user,can_edit:true,edit_version:version},mutations=[];
 globalThis.__formatFixture={account:{user,loading:false,capabilities:{community_bookmarks:true},refresh:async()=>{}},request:async(path,payload)=>{if(payload){mutations.push({path,payload});throw Error('No publication during formatting');}return path.includes('?')?{items:[],total:0}:post;}};
 const {default:Community}=await server.ssrLoadModule('/src/Community.tsx'),{default:Interview}=await server.ssrLoadModule('/src/InterviewComposer.tsx');
 const {default:PostEditor}=await server.ssrLoadModule('/src/PostEditor.tsx'),{default:ReplyEditor}=await server.ssrLoadModule('/src/ReplyEditor.tsx');
 const {default:CardActions}=await server.ssrLoadModule('/src/CommunityCardActions.tsx');
 const drafts=await server.ssrLoadModule('/src/communityDrafts.ts'),replies=await server.ssrLoadModule('/src/replyDrafts.ts'),interviews=await server.ssrLoadModule('/src/interviewDrafts.ts'),postEdits=await server.ssrLoadModule('/src/communityEdits.ts'),replyEdits=await server.ssrLoadModule('/src/replyEdits.ts');
 for(const kind of ['post','reply','interview','edit-post','edit-reply','inline-reply']){
  sessionStorage.clear();
  if(kind==='post'){await render(Community,{route:'community?category=questions'});await click(named('New post'));}
  if(kind==='reply')await render(Community,{postId,route:'community/'+postId});
  if(kind==='interview')await render(Interview,{owner,company:'google',problems:[],onClose(){},onPublished(){}});
  if(kind==='edit-post')await render(PostEditor,{owner,post,disabled:false,onSaved(){},onClose(){},onRefresh:async()=>true});
  if(kind==='edit-reply')await render(ReplyEditor,{owner,reply,disabled:false,onSaved(){},onClose(){},onRefresh:async()=>true});
  if(kind==='inline-reply'){await render(CardActions,{post});await click(named('Write comment'));}
  field=document.querySelector('.markdown-editor textarea');assert(field,kind);assert.equal(field.getAttribute('aria-keyshortcuts'),'Control+Enter Meta+Enter',kind+' opts in to native validated shortcut submission');assert(field.getAttribute('aria-describedby').split(' ').some(id=>document.getElementById(id)?.classList.contains('markdown-submit-shortcut')));await input(field,'A concrete reflection 😀');field.focus();field.setSelectionRange(2,10);field.scrollTop=17;
  await click(named('Bold'));assert.equal(field.value,'A **concrete** reflection 😀');
  const draft=kind==='post'?drafts.readPostDraft(sessionStorage,owner,'general'):['reply','inline-reply'].includes(kind)?replies.readReplyDraft(sessionStorage,owner,postId):kind==='interview'?interviews.readInterviewDraft(sessionStorage,owner):kind==='edit-post'?postEdits.readPostEdit(sessionStorage,owner,postId):replyEdits.readReplyEdit(sessionStorage,owner,replyId);
  assert.equal(draft.body,field.value,'Existing scoped draft path: '+kind);
  const beforeLive=JSON.stringify({...sessionStorage});field.scrollLeft=9;
  await click(named('Live preview'));assert.equal(document.querySelector('.markdown-editor textarea'),field);assert(!field.closest('label').hidden);assert(document.querySelector('.markdown-editor.has-live-preview .compose-preview').textContent.includes('concrete'));assert.equal(field.selectionStart,4);assert.equal(field.selectionEnd,12);assert.equal(field.scrollTop,17);assert.equal(field.scrollLeft,9);assert.equal(JSON.stringify({...sessionStorage}),beforeLive,'Changing the reading layout does not rewrite the '+kind+' draft');
  await click(named('Preview'));await click(named('Write'));assert.equal(document.querySelector('textarea'),field);assert.equal(field.selectionStart,4);assert.equal(field.selectionEnd,12);assert.equal(field.scrollTop,17);
  await click(named('Undo formatting'));assert.equal(field.value,'A concrete reflection 😀');assert.equal(mutations.length,0);
  if(kind==='post'||kind==='interview')await input(document.querySelector(`input[maxlength="${kind==='post'?160:180}"]`),'A complete publishing title');
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
 console.log('PASS Markdown editing: seven bilingual tools, exact source/Unicode/fence/length safety, shortcuts, mounted preview selection/scroll, guarded undo/focus, affiliation gates across six real authoring surfaces, direct/shortcut publish prevention, preserved editable drafts and verified-account recovery');
}finally{
 if(root)await act(async()=>root.unmount());await server.close();dom.window.close();
 for(const [key,descriptor] of previous){if(descriptor)Object.defineProperty(globalThis,key,{...descriptor});else delete globalThis[key];}
}
