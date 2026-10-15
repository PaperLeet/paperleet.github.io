import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

export async function checkCommunityEditing(server) {
  const edits=await server.ssrLoadModule('/src/communityEdits.ts');
  const values=new Map(), storage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
  const replyEdits=await server.ssrLoadModule('/src/replyEdits.ts');
  const reply={id:'reply',body:'Original',author:{id:'alice'},can_edit:true,edit_version:'a'.repeat(64)};
  assert(replyEdits.canEditReply(reply,'alice'));assert(!replyEdits.canEditReply(reply,'bob'));
  assert(!replyEdits.canEditReply({...reply,can_edit:undefined},'alice'));
  const replyDraft={body:'  中文 🧪\n```python\nx = 1\n```',baseVersion:reply.edit_version};
  assert(replyEdits.writeReplyEdit(storage,'alice','reply',replyDraft));
  assert.deepEqual(replyEdits.readReplyEdit(storage,'alice','reply'),replyDraft);
  assert.equal(replyEdits.readReplyEdit(storage,'bob','reply'),undefined);
  assert.equal(replyEdits.readReplyEdit(storage,'alice','other'),undefined);
  const newerReply={...replyDraft,body:'New work'};
  replyEdits.writeReplyEdit(storage,'alice','reply',newerReply);
  assert(!replyEdits.clearSavedReplyEdit(storage,'alice','reply',replyDraft));
  assert(replyEdits.clearSavedReplyEdit(storage,'alice','reply',newerReply));
  assert(!replyEdits.writeReplyEdit(storage,'alice','reply',{...replyDraft,body:'x'.repeat(10001)}));
  assert(!replyEdits.writeReplyEdit(undefined,'alice','reply',replyDraft));
  const post={id:'post',title:'Original',body:'Published text',category:'questions',edit_version:'a'.repeat(64),author:{id:'alice'},viewer:{can_edit:true}};
  assert(edits.canEditPost(post,'alice'));
  for(const target of [{...post,viewer:{}},{...post,viewer:{can_delete:true}},{...post,edit_version:'bad'},{...post,author:{id:'bob'}}])assert(!edits.canEditPost(target,'alice'),'Only an explicit supported author permission enables editing');
  assert(!edits.canEditPost(post,'bob'));assert(!edits.canEditPost(post,''));
  const draft={...edits.newPostEdit(post),title:'x'.repeat(180),body:'  中文 🧪\n\n```python\nx = "<script>"\n```\n'};
  assert(edits.writePostEdit(storage,'alice','post',draft));
  assert.deepEqual(edits.readPostEdit(storage,'alice','post'),draft);
  assert.equal(edits.readPostEdit(storage,'bob','post'),undefined);assert.equal(edits.readPostEdit(storage,'alice','other'),undefined);
  assert.notEqual(edits.postEditKey('a:b','c'),edits.postEditKey('a','b:c'));
  storage.setItem(edits.postEditKey('bob','post'),storage.getItem(edits.postEditKey('alice','post')));
  assert.equal(edits.readPostEdit(storage,'bob','post'),undefined);
  storage.setItem(edits.postEditKey('alice','other'),storage.getItem(edits.postEditKey('alice','post')));
  assert.equal(edits.readPostEdit(storage,'alice','other'),undefined,'A copied envelope cannot move an edit to another post');
  for(const change of [{title:'x'.repeat(181)},{body:'x'.repeat(20001)},{category:'unknown'},{baseVersion:'wrong'},{body:4}]) {
    assert(!edits.writePostEdit(storage,'alice','post',{...draft,...change}));
  }
  const newer={...draft,body:'Newer draft after returning to the thread'};
  assert(edits.writePostEdit(storage,'alice','post',newer));
  assert(!edits.clearSavedPostEdit(storage,'alice','post',draft));
  assert.deepEqual(edits.readPostEdit(storage,'alice','post'),newer);
  assert(edits.clearSavedPostEdit(storage,'alice','post',newer));assert.equal(edits.readPostEdit(storage,'alice','post'),undefined);
  for(const raw of ['null','{bad',JSON.stringify({version:2,owner:'alice',post:'post',draft}),'x'.repeat(200001)]) {
    storage.setItem(edits.postEditKey('alice','post'),raw);assert.equal(edits.readPostEdit(storage,'alice','post'),undefined);
  }
  const blocked={getItem(){throw Error('blocked');},setItem(){throw Error('quota');},removeItem(){throw Error('blocked');}};
  assert.equal(edits.readPostEdit(blocked,'alice','post'),undefined);assert(!edits.writePostEdit(blocked,'alice','post',draft));
  const {default:PostEditor}=await server.ssrLoadModule('/src/PostEditor.tsx');
  const {default:ReplyEditor}=await server.ssrLoadModule('/src/ReplyEditor.tsx');
  const {LocaleProvider}=await server.ssrLoadModule('/src/i18n.tsx');
  const previous=globalThis.window;
  const previousLocale=globalThis.localStorage;
  try {
    globalThis.window={sessionStorage:storage};
    replyEdits.writeReplyEdit(storage,'alice','reply',replyDraft);
    const replyHtml=renderToStaticMarkup(React.createElement(ReplyEditor,{reply:{...reply,body:'New server reply',edit_version:'b'.repeat(64)},owner:'alice',disabled:false,onSaved(){},onClose(){},onRefresh:async()=>true}));
    assert(replyHtml.includes('New server reply')&&replyHtml.includes('较早版本')&&replyHtml.includes('中文 🧪'));
    assert(/disabled="">保存修改/.test(replyHtml));
    edits.writePostEdit(storage,'alice','post',draft);
    const latest={...post,title:'Newer public title',body:'Newer public body',edit_version:'b'.repeat(64)};
    const html=renderToStaticMarkup(React.createElement(PostEditor,{post:latest,owner:'alice',disabled:false,onSaved(){},onClose(){},onRefresh:async()=>true}));
    assert(html.includes('Newer public title')&&html.includes('Newer public body')&&html.includes('较早版本'));
    assert(html.includes('以此版本继续编辑')&&html.includes(draft.title),'A recovered draft keeps its old base and exposes current content for deliberate review');
    assert(/disabled="">保存修改/.test(html),'Saving a stale recovered draft must be disabled until reviewed');
    assert(!html.includes('<script>'),'Draft text remains escaped');
    globalThis.localStorage={getItem:()=> 'en'};
    const english=renderToStaticMarkup(React.createElement(LocaleProvider,null,React.createElement(PostEditor,{post:latest,owner:'alice',disabled:false,onSaved(){},onClose(){},onRefresh:async()=>true})));
    assert(english.includes('Latest post content')&&english.includes('Reviewed; continue from this version')&&english.includes('Save changes'));
  } finally {
    if(previous===undefined)delete globalThis.window;else globalThis.window=previous;
    if(previousLocale===undefined)delete globalThis.localStorage;else globalThis.localStorage=previousLocale;
  }
  console.log('PASS author post editing: backend capability gate, exact bounded drafts, owner/post isolation, stale-version review, blocked storage and late acknowledgement preservation');
}
