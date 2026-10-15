import assert from 'node:assert/strict';
import React from 'react';
import { localizedRenderer } from "./render-locale.mjs";

export async function checkPublicationRequests(server){
 const renderToString = await localizedRenderer(server, "zh");
 const {preparePublication,readPublication,clearPublication,postPublication,replyPublication,samePublication,publicationKey}=await server.ssrLoadModule('/src/publicationRequests.ts');
 const values=new Map(),storage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
 const owner='publication-alice',scope='architecture/attention';
 const draft={title:' Pending 🧪 ',body:' <script>literal</script>\n```python\nx = 1\n``` ',category:'questions'};
 const payload=postPublication(draft,scope);
 const first=preparePublication(storage,owner,'post',scope,payload);
 assert(first.durable&&/^[a-f0-9]{32}$/.test(first.receipt.requestId));
 assert.equal(preparePublication(storage,owner,'post',scope,{...payload}).receipt.requestId,first.receipt.requestId,'An unacknowledged retry must reuse the original ID');
 assert.equal(samePublication(first.receipt,Object.fromEntries(Object.entries(payload).reverse())),true,'Property order is not a content edit');
 const fresh=await server.ssrLoadModule('/src/publicationRequests.ts?fresh-publication-document');
 assert.equal(fresh.readPublication(storage,owner,'post',scope).receipt.requestId,first.receipt.requestId,'Refresh must recover the persisted request identity');
 const edited={...payload,body:'A newer draft'};
 assert(!samePublication(first.receipt,edited));
 const second=preparePublication(storage,owner,'post',scope,edited);
 assert.notEqual(second.receipt.requestId,first.receipt.requestId,'An explicit new-content publication has a new ID');
 assert.equal(clearPublication(storage,first.receipt),false,'A late acknowledgement must not erase a newer request');
 assert.equal(readPublication(storage,owner,'post',scope).receipt.requestId,second.receipt.requestId);
 assert.throws(()=>preparePublication(storage,owner,'post',scope,payload,first.receipt),/pending publication changed/);
 assert(clearPublication(storage,second.receipt));
 assert.equal(readPublication(storage,owner,'post',scope),undefined);
 assert.notEqual(preparePublication(storage,owner,'post',scope,payload).receipt.requestId,first.receipt.requestId,'After acknowledgement, a new intentional post may use the same source');
 const bob=preparePublication(storage,'publication-bob','post',scope,payload);
 assert.notEqual(bob.receipt.requestId,first.receipt.requestId);
 const post='a'.repeat(32),parent='b'.repeat(32),reply=replyPublication({body:' reply ',parentId:parent});
 assert.deepEqual(reply,{body:'reply',parent_id:parent});
 const r=preparePublication(storage,owner,'reply',post,reply);
 assert.notEqual(preparePublication(storage,owner,'reply','c'.repeat(32),reply).receipt.requestId,r.receipt.requestId);
 assert(!samePublication(r.receipt,replyPublication({body:'reply'})),'Changing the reply target is new content');
 values.set(publicationKey('copied-owner','reply',post),JSON.stringify(r.receipt));
 assert.equal(fresh.readPublication(storage,'copied-owner','reply',post),undefined);
 for(const raw of ['null','{bad',JSON.stringify({...r.receipt,owner:'malformed',requestId:['a'.repeat(32)]}),JSON.stringify({...r.receipt,owner:'malformed',payload:{body:'reply',parent_id:'wrong'}})]){
  values.set(publicationKey('malformed','reply',post),raw);assert.equal(fresh.readPublication(storage,'malformed','reply',post),undefined);
 }
 const blocked={getItem(){throw Error('blocked');},setItem(){throw Error('blocked');},removeItem(){throw Error('blocked');}};
 const volatile=preparePublication(blocked,'volatile','reply',post,reply);
 assert(!volatile.durable);
 assert.equal(preparePublication(blocked,'volatile','reply',post,reply).receipt.requestId,volatile.receipt.requestId);
 assert.equal(fresh.readPublication(blocked,'volatile','reply',post),undefined,'Blocked storage has no reload guarantee');
 const {default:PublicationRecovery}=await server.ssrLoadModule('/src/PublicationRecovery.tsx');
 const html=renderToString(React.createElement(PublicationRecovery,{pending:first,payload:edited,busy:false,onRetry(){}}));
 assert(html.includes('重试此前发布')&&html.includes('当前草稿已经改动'));
 assert(html.includes('&lt;script&gt;')&&!html.includes('<script>'));
 assert(!renderToString(React.createElement(PublicationRecovery,{pending:first,payload,busy:true,onRetry(){}})));
 console.log('PASS publication requests: durable/volatile retry identity, explicit new content, owner/post/parent isolation, stale acknowledgements, malformed storage and safe recovery review');
}
