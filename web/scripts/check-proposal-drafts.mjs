import assert from 'node:assert/strict';
import {createServer} from 'vite';
const server=await createServer({server:{middlewareMode:true,hmr:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]}});
try {
  const {emptyProposalDraft,proposalDraftKey,proposalDraftOwner,proposalDraftLimits,readProposalDraft,writeProposalDraft,clearSubmittedProposalDraft}=await server.ssrLoadModule('/src/proposalDrafts.ts');
  const values=new Map();
  const storage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};
  const alice=proposalDraftOwner('hosted','alice'),bob=proposalDraftOwner('hosted','bob');
  assert.equal(proposalDraftOwner('local'),'local');assert.equal(proposalDraftOwner('hosted'),undefined);assert.equal(proposalDraftOwner(undefined,'alice'),undefined);
  assert.notEqual(proposalDraftKey(alice),proposalDraftKey(bob));assert.notEqual(proposalDraftKey(alice),proposalDraftKey('local'));assert.notEqual(proposalDraftKey('a:b'),proposalDraftKey('a%3Ab'));
  const draft={title:'  中文 paper 🧪  ',paperUrl:'https://arxiv.org/abs/',codeUrl:'https://github.com/example/',excerpt:'\n\\alpha + beta\n```python\nreturn x\n```\n',focus:'Preserve exact whitespace.\n'};
  const first=writeProposalDraft(storage,alice,draft);assert(first.saved&&first.revision);
  assert.deepEqual(readProposalDraft(storage,alice).record.draft,draft);assert.equal(readProposalDraft(storage,bob).record,undefined);
  storage.setItem(proposalDraftKey(bob),storage.getItem(proposalDraftKey(alice)));assert.equal(readProposalDraft(storage,bob).record,undefined,'Copied envelopes cannot change owners');
  values.delete(proposalDraftKey(bob));
  const second=writeProposalDraft(storage,alice,{...draft,focus:'Newer work'});assert(second.saved&&second.revision!==first.revision);
  assert.equal(clearSubmittedProposalDraft(storage,alice,first.revision,draft),'kept');assert.equal(readProposalDraft(storage,alice).record.draft.focus,'Newer work');
  const aba=writeProposalDraft(storage,alice,draft);assert(aba.revision!==first.revision);assert.equal(clearSubmittedProposalDraft(storage,alice,first.revision,draft),'kept','Returning to the same source is still a new edit revision');
  assert.equal(clearSubmittedProposalDraft(storage,alice,aba.revision,{...draft,title:'Different'}),'kept');
  assert.equal(clearSubmittedProposalDraft(storage,alice,aba.revision,draft),'cleared');assert.equal(readProposalDraft(storage,alice).record,undefined);
  assert.equal(writeProposalDraft(storage,undefined,draft).saved,false);assert.equal(readProposalDraft(storage,undefined).available,false);
  const blocked={getItem(){throw Error('blocked');},setItem(){throw Error('blocked');},removeItem(){throw Error('blocked');}};
  assert.equal(readProposalDraft(blocked,alice).available,false);assert.equal(writeProposalDraft(blocked,alice,draft).saved,false);assert.equal(clearSubmittedProposalDraft(blocked,alice,first.revision,draft),'unavailable');
  assert.equal(writeProposalDraft(undefined,alice,draft).saved,false);
  const cannotDelete={...storage,removeItem(){throw Error('blocked');}};
  const kept=writeProposalDraft(storage,alice,draft);assert.equal(clearSubmittedProposalDraft(cannotDelete,alice,kept.revision,draft),'unavailable');assert.deepEqual(readProposalDraft(storage,alice).record.draft,draft);
  for(const [field,limit] of Object.entries(proposalDraftLimits)) {
    const maximum={...emptyProposalDraft(),[field]:'\0'.repeat(limit)};
    assert(writeProposalDraft(storage,alice,maximum).saved,field);assert.deepEqual(readProposalDraft(storage,alice).record.draft,maximum);
    assert.equal(writeProposalDraft(storage,alice,{...maximum,[field]:'x'.repeat(limit+1)}).saved,false,field);
  }
  for(const raw of ['{bad','null','[]','a'.repeat(500001),JSON.stringify({version:2,owner:alice,revision:'r',draft}),JSON.stringify({version:1,owner:bob,revision:'r',draft}),JSON.stringify({version:1,owner:alice,revision:3,draft}),JSON.stringify({version:1,owner:alice,revision:'r',draft:{...draft,title:42}})]) {
    storage.setItem(proposalDraftKey(alice),raw);assert.equal(readProposalDraft(storage,alice).record,undefined);assert.equal(readProposalDraft(storage,alice).available,true);
  }
  assert(writeProposalDraft(storage,alice,{...draft,role:'admin',acceptedReport:{passed:true}}).saved);
  assert.deepEqual(Object.keys(readProposalDraft(storage,alice).record.draft).sort(),Object.keys(emptyProposalDraft()).sort(),'Only editable proposal fields persist');
  assert(writeProposalDraft(storage,bob,{...draft,title:'Bob draft'}).saved);
  assert(writeProposalDraft(storage,alice,emptyProposalDraft()).saved);assert.equal(readProposalDraft(storage,alice).record,undefined);assert.equal(readProposalDraft(storage,bob).record.draft.title,'Bob draft');
  console.log('PASS proposal draft storage: exact Unicode/whitespace, owner/local separation, bounded editable fields, malformed data, revision/ABA-safe cleanup, blocked storage and no grade/role persistence');
} finally {await server.close();}
