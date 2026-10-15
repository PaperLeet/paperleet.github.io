import assert from 'node:assert/strict';
import {createServer} from 'vite';
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
try {
 const nav=await server.ssrLoadModule('/src/interviewNavigation.ts');
 const models=await server.ssrLoadModule('/src/interviewModels.ts');
 const library=await server.ssrLoadModule('/src/libraryNavigation.ts');
 const community=await server.ssrLoadModule('/src/communityNavigation.ts');
 const account=await server.ssrLoadModule('/src/accountNavigation.ts');
 const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
 for(const q of ['数学 & optimization 😀','#/%? +=','注意力，复盘与准备']){
  const filters={...nav.emptyInterviewFilters(),company:'amazon',role:'ml',stage:'technical',q,days:'90',sort:'helpful',page:3,track:'llm',saved:true,concept:'architecture/attention'};
  const href=nav.interviewHref(filters);assert.deepEqual(nav.readInterviewRoute(href),filters);
  const practice=library.withLibraryContext('architecture/attention',href);assert.equal(library.libraryReturnHref(practice),href);
  const thread=community.communityPostHref('a'.repeat(32),href,'b'.repeat(32));assert.equal(community.communityPostReturn(thread,'a'.repeat(32)),href);
  for(const destination of [href,practice,thread])assert.equal(account.accountReturnHref(decodeProblemHash(account.signInHref(destination))),destination);
  assert(!community.communityShareUrl('https://paperleet.test/#/interviews','a'.repeat(32),'b'.repeat(32)).includes('from='));
 }
 for(const route of ['interviews?saved=true','interviews?saved=1&saved=1','interviews?concept=bad','interviews?concept=../evil','interviews?track=unknown','interviews?track=','interviews?track=llm&track=training','interviews?from=interviews','interviews?page=0','interviews?page=5001','interviews?company=amazon&company=google','interviews?view=other','interviews?q=%FF','interviews?q=%ZZ','interviews?q=%00','interviews?q=%7f','https://evil.test/interviews','interviews?days=7','interviews?q='+encodeURIComponent('x'.repeat(201))])assert.equal(nav.readInterviewRoute(route),undefined,route);
 assert.equal(library.libraryReturnHref('#/22?from='+encodeURIComponent('interviews?from=interviews')),undefined);
 assert.equal(account.signInHref('#/22?from=https%3A%2F%2Fevil.test'),'#/account');
 const details={company:'amazon',role:'ml',stage:'technical',outcome:'unknown',interview_month:'',location:'Remote',questions:['finetuning/lora','architecture/attention']};
 assert.deepEqual(models.readInterviewDetails(details).questions,['architecture/attention','finetuning/lora']);
 for(const change of [{company:'fake'},{interview_month:'2099-01'},{questions:['architecture/attention','architecture/attention']},{questions:[null]},{location:'bad\ncity'}])assert.equal(models.readInterviewDetails({...details,...change}),undefined);
 const drafts=await server.ssrLoadModule('/src/interviewDrafts.ts'),receipts=await server.ssrLoadModule('/src/publicationRequests.ts');
 const map=new Map(),storage={getItem:k=>map.get(k)??null,setItem:(k,v)=>map.set(k,v),removeItem:k=>map.delete(k)};
 const draft={title:' My interview ',body:' Shared reflection 😀 ',interview:details};
 assert(drafts.writeInterviewDraft(storage,'model-alice',draft));assert.deepEqual(drafts.readInterviewDraft(storage,'model-alice'),draft);assert.equal(drafts.readInterviewDraft(storage,'model-bob'),undefined);
 const first=receipts.preparePublication(storage,'model-alice','interview','general',drafts.interviewPublication(draft));
 assert.equal(receipts.preparePublication(storage,'model-alice','interview','general',drafts.interviewPublication({...draft,interview:{...details,questions:[...details.questions].reverse()}})).receipt.requestId,first.receipt.requestId,'Canonical metadata keeps a retry identical');
 const changed={...draft,interview:{...details,company:'google'}},second=receipts.preparePublication(storage,'model-alice','interview','general',drafts.interviewPublication(changed));
 assert.notEqual(second.receipt.requestId,first.receipt.requestId);assert.equal(receipts.clearPublication(storage,first.receipt),false);
 assert(drafts.writeInterviewDraft(storage,'model-alice',changed));assert.equal(drafts.clearInterviewDraft(storage,'model-alice',draft),false);assert.deepEqual(drafts.readInterviewDraft(storage,'model-alice'),changed);
 assert.throws(()=>receipts.preparePublication(storage,'model-bob','interview','general',{...drafts.interviewPublication(draft),interview:'{}'}));
 const blocked={getItem(){throw Error('blocked');},setItem(){throw Error('blocked');},removeItem(){throw Error('blocked');}};
 assert.equal(drafts.writeInterviewDraft(blocked,'volatile-owner',draft),false);assert.deepEqual(drafts.readInterviewDraft(blocked,'volatile-owner'),draft);assert.equal(drafts.clearInterviewDraft(blocked,'volatile-owner',draft),false);
 const volatile=receipts.preparePublication(blocked,'volatile-owner','interview','general',drafts.interviewPublication(draft));assert.equal(volatile.durable,false);assert.equal(receipts.preparePublication(undefined,'volatile-owner','interview','general',drafts.interviewPublication(draft)).receipt.requestId,volatile.receipt.requestId);
 map.set(drafts.interviewDraftKey('copied-owner'),JSON.stringify({version:1,owner:'model-alice',draft}));assert.equal(drafts.readInterviewDraft(storage,'copied-owner'),undefined);
 const edits=await server.ssrLoadModule('/src/communityEdits.ts');
 const post={id:'a'.repeat(32),title:'Interview',body:'Details',category:'career',author:{id:'model-alice'},viewer:{can_edit:true},edit_version:'d'.repeat(64),interview:models.readInterviewDetails(details)};
 const edit=edits.newPostEdit(post);edit.interview.questions.push('rl/dpo');assert.equal(post.interview.questions.length,2);
 assert(edits.writePostEdit(storage,'model-alice',post.id,edit));assert.deepEqual(edits.readPostEdit(storage,'model-alice',post.id),edit);assert.equal(edits.readPostEdit(storage,'model-bob',post.id),undefined);
 assert.equal(edits.samePostEdit(edit,{...edit,interview:{...edit.interview,company:'google'}}),false);
 const summary={reports:0,contributors:0,basis:'published-community-reports',companies:[],questions:[],days:'365',generated_at:new Date().toISOString()};
 assert(models.readInterviewSummary(summary));assert.equal(models.readInterviewSummary({...summary,reports:-1}),undefined);assert.equal(models.readInterviewSummary({...summary,companies:[{company:'amazon',reports:0,contributors:1}]}),undefined);
 assert(models.readInterviewFeed({items:[],total:0,limit:20,offset:0}));assert.equal(models.readInterviewFeed({items:[{interview:details}],total:1,limit:20,offset:0}),undefined);
 console.log('PASS interview models: Unicode filter/thread/reply/practice/login round trips, strict local routes, canonical metadata, owner-isolated drafts, blocked storage, idempotent publication recovery, newer request protection, versioned edit metadata and malformed response rejection');
} finally {await server.close();}
