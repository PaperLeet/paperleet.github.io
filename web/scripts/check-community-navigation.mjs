import assert from 'node:assert/strict';
import React from 'react';
import { localizedRenderer } from './render-locale.mjs';
import { createServer } from 'vite';

const server = await createServer({server:{middlewareMode:true},appType:'custom'});
try {
  const renderToString=await localizedRenderer(server,"zh");
  const {communityFeedHref,communityPostHref,communityPostReturn,communityReplyTarget,communityShareUrl,readCommunityFeed,communityContinuation,emptyCommunityFeed}=await server.ssrLoadModule('/src/communityNavigation.ts');
  const {signInHref,accountReturnHref}=await server.ssrLoadModule('/src/accountNavigation.ts');
  const {decodeProblemHash}=await server.ssrLoadModule('/src/reports.ts');
  const {default:Community}=await server.ssrLoadModule('/src/Community.tsx');
  const {default:PostBookmark}=await server.ssrLoadModule('/src/PostBookmark.tsx');
  const {confirmedBookmark}=await server.ssrLoadModule('/src/communityBookmarks.ts');
  const id='a'.repeat(32), topics=[undefined,'architecture/attention','retired/module'];
  for (const topic of topics) {
    for (const category of ['questions','admissions','career']) for (const search of ['', 'mask gradient', '张量 & mask=1? 100%_% # + /', '中文 😀']) {
      const state={category,following:true,search,page:3};
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
  for(const topic of topics) {
    const state={category:'questions',following:false,saved:true,search:'数学 & mask 😀',page:2};
    const feed=communityFeedHref(topic,state), post=communityPostHref(id,feed,'b'.repeat(32));
    assert.deepEqual(readCommunityFeed(feed),{topicId:topic,state});
    assert.equal(communityPostReturn(post,id),feed);
    assert.equal(accountReturnHref(decodeProblemHash(signInHref(feed))),feed);
    assert.equal(accountReturnHref(decodeProblemHash(signInHref(post))),post);
    assert.equal(communityShareUrl('https://paperleet.test/?private=query'+post,id),`https://paperleet.test/#/community/${id}`);
  }
  for(const query of ['saved=other','saved=1&saved=0']) {
    assert.equal(readCommunityFeed('community?'+query),undefined);
    assert.equal(signInHref('community?'+query),'#/account');
  }
  for(const sort of ['active','top']) for(const topic of topics) {
    const state={category:'papers',following:false,saved:true,sort,search:'中文 & mask 😀',page:3};
    const feed=communityFeedHref(topic,state),post=communityPostHref(id,feed);
    assert.deepEqual(readCommunityFeed(feed),{topicId:topic,state});
    assert.equal(communityPostReturn(post,id),feed);
    assert.equal(accountReturnHref(decodeProblemHash(signInHref(post))),post);
  }
  for(const query of ['sort=invalid','sort=TOP','sort=top&sort=active','sort=top&sort=']) {
    assert.equal(readCommunityFeed('community?'+query),undefined);
    assert.equal(signInHref('community?'+query),'#/account');
  }
  const owner='viewer-a',ack={owner_id:owner,post_id:id,bookmarked:true};
  assert.deepEqual(confirmedBookmark(ack,owner,id,true),ack);
  assert.deepEqual(confirmedBookmark({...ack,bookmarked:false},owner,id),{...ack,bookmarked:false});
  for(const value of [null,[],{}, {...ack,owner_id:'viewer-b'}, {...ack,post_id:'b'.repeat(32)}, {...ack,bookmarked:'true'}, {...ack,bookmarked:false}]) {
    assert.equal(confirmedBookmark(value,owner,id,true),undefined);
  }
  const props={postId:id,owner,bookmarked:false,enabled:true,returnTo:communityPostHref(id),disabled:false,onChange(){},onBusyChange(){}};
  const html=renderToString(React.createElement(PostBookmark,props));
  assert(html.includes('aria-label="收藏帖子"')&&html.includes('aria-pressed="false"'));
  assert(renderToString(React.createElement(PostBookmark,{...props,bookmarked:true})).includes('aria-label="取消收藏"'));
  assert(renderToString(React.createElement(PostBookmark,{...props,disabled:true})).includes('disabled=""'));
  assert.equal(renderToString(React.createElement(PostBookmark,{...props,enabled:false})), '');
  assert.equal(renderToString(React.createElement(PostBookmark,{...props,bookmarked:undefined})), '');
  const guest=renderToString(React.createElement(PostBookmark,{...props,owner:''}));
  assert(guest.includes('登录收藏')&&guest.includes(signInHref(props.returnTo))&&!guest.includes('<button'));
  assert.equal(communityFeedHref(undefined,{...emptyCommunityFeed(),code:'private code',reply:'private reply',owner:'private account'}),'#/community');
  assert.equal(communityPostHref(id),'#/community/'+id);
  const reply='b'.repeat(32);
  for(const back of [undefined,communityFeedHref('architecture/attention',{category:'questions',following:true,search:'中文 & mask # 😀',page:2})]) {
    const linked=communityPostHref(id,back,reply);
    assert.equal(communityReplyTarget(linked,id),reply);
    assert.equal(communityReplyTarget(decodeProblemHash(linked),id),reply);
    assert.equal(communityPostReturn(linked,id),back);
    assert.equal(accountReturnHref(decodeProblemHash(signInHref(linked))),linked);
    assert.equal(accountReturnHref(decodeProblemHash(signInHref(decodeProblemHash(linked)))),linked);
  }
  assert.equal(communityShareUrl('https://user:secret@paperleet.test/preview/?token=private#/community?q=private',id,reply),`https://paperleet.test/preview/#/community/${id}?reply=${reply}`);
  assert.equal(communityShareUrl('http://localhost:8954/#/anything',id),`http://localhost:8954/#/community/${id}`);
  assert.throws(()=>communityShareUrl('javascript:alert(1)',id));
  for(const bad of ['', '../escape', 'https://evil.test', 'c'.repeat(33), 'B'.repeat(32)])assert.throws(()=>communityPostHref(id,undefined,bad));
  for(const query of [`reply=${reply}&reply=${reply}`,`reply=../escape`,`reply=${reply}&from=https%3A%2F%2Fevil.test`,`reply=${reply}&unknown=secret`,`reply=`]) {
    const bad=`community/${id}?${query}`;
    assert.equal(communityReplyTarget(bad,id),undefined);
    assert.equal(communityContinuation(bad),undefined);
    assert.equal(signInHref(bad),'#/account');
  }
  assert.equal(communityContinuation('community?following=true&page=2'),'community?following=1&page=2');
  const badFeeds=['https://evil.test/','//evil.test/','javascript:alert(1)','community/problem/../attention','community/problem/architecture/attention/extra',
    'account/continue/community','community?from=community','community?unknown=1','community?category=../private','community?page=0','community?page=-2','community?page=1.5',
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
  assert(markup.includes('value="mask &amp; gradient"') && markup.includes('maxLength="400"') && markup.includes('最多200个字符') && markup.includes('type="search"'));
  assert(markup.includes('aria-pressed="true"') && markup.includes(signInHref(route).replaceAll('&','&amp;')));
  for(const [locale,admissions,career] of [['zh','求学','求职'],['en','School','Job search']]){
    const render=await localizedRenderer(server,locale);
    const html=render(React.createElement(Community,{route:'community?category=admissions'}));
    assert(html.includes(admissions)&&html.includes(career),'Both distinct boards must be visible in '+locale);
    assert(!html.includes('学术与职业')&&!html.includes('Research &amp; careers'),'The combined board label is retired');
  }
  console.log('PASS community navigation: scoped filter/page URLs, reserved Unicode search text, post return context, login round trips, public-only targets and malformed/recursive/external rejection');
} finally { await server.close(); }
