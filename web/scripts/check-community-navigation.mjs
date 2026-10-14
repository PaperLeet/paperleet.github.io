import assert from 'node:assert/strict';
import React from 'react';
import { renderToString } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({server:{middlewareMode:true},appType:'custom'});
try {
  const {communityFeedHref,communityPostHref,communityPostReturn,readCommunityFeed,communityContinuation,emptyCommunityFeed}=await server.ssrLoadModule('/src/communityNavigation.ts');
  const {signInHref,accountReturnHref}=await server.ssrLoadModule('/src/accountNavigation.ts');
  const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
  const {default:Community}=await server.ssrLoadModule('/src/Community.tsx');
  const id='a'.repeat(32), topics=[undefined,'architecture/attention','retired/module'];
  for (const topic of topics) {
    for (const search of ['', 'mask gradient', '张量 & mask=1? 100%_% # + /', '中文 😀']) {
      const state={category:'questions',following:true,search,page:3};
      const href=communityFeedHref(topic,state);
      assert.deepEqual(readCommunityFeed(href),{topicId:topic,state});
      assert.deepEqual(readCommunityFeed(decodeProblemHash(href)),{topicId:topic,state});
      assert.equal(accountReturnHref(decodeProblemHash(signInHref(href))),href);
      assert.equal(accountReturnHref(decodeProblemHash(signInHref(decodeProblemHash(href)))),href);
      const post=communityPostHref(id,href);
      assert.equal(communityPostReturn(post,id),href);
      assert.equal(communityPostReturn(decodeProblemHash(post),id),href);
      assert.equal(communityPostReturn(post,'b'.repeat(32)),undefined);
      assert.equal(accountReturnHref(decodeProblemHash(signInHref(post))),post);
      assert.equal(accountReturnHref(decodeProblemHash(signInHref(decodeProblemHash(post)))),post);
    }
  }
  assert.equal(communityFeedHref(undefined,emptyCommunityFeed()),'#/community');
  assert.equal(communityFeedHref(undefined,{...emptyCommunityFeed(),code:'private code',reply:'private reply',owner:'private account'}),'#/community');
  assert.equal(communityPostHref(id),'#/community/'+id);
  assert.equal(communityContinuation('community?following=true&page=2'),'community?following=1&page=2');
  const badFeeds=['https://evil.test/','//evil.test/','javascript:alert(1)','community/problem/../attention','community/problem/architecture/attention/extra',
    'account/continue/community','community?from=community','community?unknown=1','community?category=private','community?page=0','community?page=-2','community?page=1.5',
    'community?page=9007199254740991','community?q='+encodeURIComponent('x'.repeat(201)),'community?q=a&q=b','community?following=admin','community%252Fproblem%252Farchitecture%252Fattention'];
  for(const route of badFeeds) {
    assert.equal(readCommunityFeed(route),undefined,route);
    assert.equal(communityContinuation(route),undefined,route);
    assert.equal(signInHref(route),'#/account',route);
    assert.equal(communityPostHref(id,route),'#/community/'+id);
  }
  for(const from of ['https://evil.test/','//evil.test/','account/continue/community',`community/${id}`,`community/${id}?from=community`]) {
    const post=`#/community/${id}?${new URLSearchParams({from})}`;
    assert.equal(communityPostReturn(post,id),undefined);
    assert.equal(signInHref(post),'#/account');
  }
  assert.equal(communityPostReturn(`community/${id}?from=community&from=community`,id),undefined);
  assert.equal(accountReturnHref('account/continue/#/community?category=questions'),undefined);
  const route=communityFeedHref('architecture/attention',{category:'questions',following:true,search:'mask & gradient',page:2});
  const markup=renderToString(React.createElement(Community,{route:decodeProblemHash(route),topicId:'architecture/attention'}));
  assert(markup.includes('value="mask &amp; gradient"') && markup.includes('maxLength="200"'));
  assert(markup.includes('aria-pressed="true"') && markup.includes(signInHref(route).replaceAll('&','&amp;')));
  console.log('PASS community navigation: scoped filter/page URLs, reserved Unicode search text, post return context, login round trips, public-only targets and malformed/recursive/external rejection');
} finally { await server.close(); }
