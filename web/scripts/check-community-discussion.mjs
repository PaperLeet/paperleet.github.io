import assert from 'node:assert/strict';

export async function checkCommunityDiscussion(server) {
  const {withConfirmedReply} = await server.ssrLoadModule('/src/communityDiscussion.ts');
  const reply={id:'new',post_id:'thread',parent_id:'parent',body:'  中文 🧪\n```python\nx = "<script>"\n```\n',author:{id:'alice'},can_delete:true,created_at:'2026-10-14T12:00:00Z'};
  const post={id:'thread',title:'Original post',body:'Keep original source',problem_id:'architecture/attention',viewer:{liked:true,reposted:false,can_delete:false},counts:{replies:1,likes:4,reposts:2},replies:[{...reply,id:'parent',parent_id:null,author:{id:'bob'},body:'Parent'}]};
  const before=structuredClone(post),original=structuredClone(reply);
  const confirmed=withConfirmedReply(post,reply);
  assert.equal(confirmed.replies.length,2,'A confirmed POST must be visible before any follow-up GET');
  assert.equal(confirmed.replies[1],reply,'Keep the actual server row, including its author, source and reply target');
  assert.deepEqual(confirmed.counts,{replies:2,likes:4,reposts:2});
  assert.equal(confirmed.viewer,post.viewer);
  assert.equal(confirmed.title,post.title);
  assert.equal(confirmed.problem_id,post.problem_id);
  assert.equal(withConfirmedReply(confirmed,reply),confirmed,'Reapplying the same acknowledgement must not duplicate or recount the reply');
  assert.equal(withConfirmedReply(post,{...reply,post_id:'other-thread'}),post,'A late reply from another thread must never enter the visible thread');
  assert.equal(withConfirmedReply(undefined,reply),undefined,'An acknowledgement cannot fabricate a missing post');
  assert.deepEqual(post,before);assert.deepEqual(reply,original);
  console.log('PASS confirmed discussion replies: visible before refresh, exact server row, duplicate acknowledgements, thread isolation and immutable counts/context');
}
