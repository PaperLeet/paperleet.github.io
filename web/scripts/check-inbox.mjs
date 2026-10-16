import assert from 'node:assert/strict';
import React, {act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom = new JSDOM('<!doctype html><div id="root"></div>', {url: 'http://paperleet.test/#/messages/bob', pretendToBeVisual: true});
const values = {window: dom.window, document: dom.window.document, navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage, location: dom.window.location, CustomEvent: dom.window.CustomEvent, requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window), cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window), IS_REACT_ACT_ENVIRONMENT: true, __inboxFixture: undefined};
const previous = new Map(Object.keys(values).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(values)) Object.defineProperty(globalThis, key, {value, writable: true, configurable: true});
localStorage.setItem('paperleet.locale', 'en');
const server = await createServer({cacheDir: 'node_modules/.vite-inbox-check', server: {middlewareMode: true, hmr: false}, appType: 'custom', optimizeDeps: {noDiscovery: true, include: []}, plugins: [{name: 'inbox-account-fixture', enforce: 'pre', resolveId(source) {if (source === './social') return '\0inbox-account';}, load(name) {if (name === '\0inbox-account') return `import React from 'react';export const useAccount=()=>globalThis.__inboxFixture.account;export const socialApi=(...args)=>globalThis.__inboxFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error {constructor(message,status,code){super(message);this.status=status;this.code=code;}}export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const DateLabel=({value})=>React.createElement('time',null,value);`;}}]});
let root;
try {
  const {createRoot} = await import('react-dom/client');
  const {default: Bell} = await server.ssrLoadModule('/src/InboxBell.tsx');
  const {default: Panel} = await server.ssrLoadModule('/src/InboxPanel.tsx');
  const {default: Messages, mergeMessages} = await server.ssrLoadModule('/src/Messages.tsx');
  const {inboxRequest, inboxChanged} = await server.ssrLoadModule('/src/inbox.ts');
  const {SocialError} = await server.ssrLoadModule('\0inbox-account');
  const alice = {id: 'alice', username: 'alice', display_name: 'Alice', avatar_color: '#48766b'};
  const bob = {id: 'bob', username: 'bob', display_name: 'Bob', avatar_color: '#48766b'};
  const account = {user: alice, loading: false, error: '', capabilities: {inbox_notifications: true, direct_messages: true}, refresh: async () => {}};
  const count = {user_id: 'alice', unread_count: 2, notification_unread_count: 1, message_unread_count: 1};
  const notification = {id: 'likes:post', kind: 'like', latest_event_id: 8, created_at: '2026-10-15T10:00:00Z', unread: true, actor_count: 3, actors: [bob], post: {id: 'a'.repeat(32), title: '<script>paper</script>'}, reply_id: null, preview: ''};
  let snapshot = {...count, observed_event_id: 8, notifications: [notification], notifications_next_before: null, conversations: [], conversations_next_before: null};
  let thread = {user_id: 'alice', peer: bob, conversation_id: null, messages: [], next_before: null, observed_message_id: 0, unread_count: 0, can_send: true, awaiting_reply: false, is_mutual_follow: false};
  let failSend = false, acknowledgeLost = false, sendRejection, deferRead = false, readRelease, deferThread = false, threadRelease, failThread = false;
  const calls = [], sent = [];
  globalThis.__inboxFixture = {account, request: async (path, body, signal) => {
    calls.push({path, body, signal});
    if (path === 'inbox?summary=1') return {...snapshot};
    if (path.startsWith('inbox?')) return structuredClone(snapshot);
    if (path === 'inbox/read') {
      assert.equal(body.expected_user_id, account.user.id);
      if (deferRead) await new Promise(resolve => {readRelease = resolve;});
      const covered = snapshot.notifications.every(item => item.latest_event_id <= body.through_event_id);
      snapshot = {...snapshot, notification_unread_count: covered ? 0 : 1, unread_count: covered ? 1 : 2, notifications: snapshot.notifications.map(item => item.latest_event_id <= body.through_event_id ? {...item, unread: false} : item)};
      return {...snapshot};
    }
    if (path.endsWith('/read')) {assert.equal(body.expected_user_id, account.user.id); thread = {...thread, unread_count: 0}; return {...count, unread_count: 1, message_unread_count: 0};}
    if (path.startsWith('messages/') && !body) {
      if (failThread) throw Error('Conversation unavailable');
      if (deferThread) return new Promise(resolve => {threadRelease = resolve;});
      return structuredClone(thread);
    }
    if (path.startsWith('messages/') && body) {
      assert.equal(body.expected_user_id, account.user.id); sent.push(body);
      if (sendRejection) throw sendRejection;
      if (failSend) {
        if (acknowledgeLost) thread = {...thread, messages: [{id: 1, sender_id: 'alice', body: body.body, request_id: body.request_id, created_at: '2026-10-15T11:00:00Z'}], observed_message_id: 1, can_send: false, awaiting_reply: true};
        throw Error('Connection lost');
      }
      if (!thread.can_send && !thread.messages.some(item => item.request_id === body.request_id)) throw new SocialError('Wait for a reply', 409, 'awaiting_reply');
      thread = {...thread, messages: [{id: 1, sender_id: 'alice', body: body.body, request_id: body.request_id, created_at: '2026-10-15T11:00:00Z'}], observed_message_id: 1, can_send: false, awaiting_reply: true};
      return structuredClone(thread);
    }
    throw Error(`Unexpected request ${path}`);
  }};
  const render = async (Component, props = {}) => act(async () => {root ??= createRoot(document.querySelector('#root')); root.render(React.createElement(Component, props));});
  const stop = async () => {if (root) await act(async () => root.unmount()); root = undefined;};
  const click = async element => {assert(element); await act(async () => {element.focus(); element.click();});};
  const button = text => [...document.querySelectorAll('button')].find(element => element.textContent === text);
  const edit = async text => act(async () => {const element = document.querySelector('textarea'); Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, 'value').set.call(element, text); element.dispatchEvent(new dom.window.Event('input', {bubbles: true}));});
  const submit = async () => act(async () => document.querySelector('form').dispatchEvent(new dom.window.Event('submit', {bubbles: true, cancelable: true})));

  await render(Bell);
  assert(document.querySelector('.inbox-unread-dot'));
  assert(calls.some(call => call.path === 'inbox?summary=1'));
  assert(!document.querySelector('.inbox-popover'));
  assert(!calls.some(call => call.path === 'inbox?limit=20'), 'Closed bell fetches counts only');
  snapshot = {...snapshot, unread_count: 0, notification_unread_count: 0, message_unread_count: 0};
  await act(async () => inboxChanged('alice'));
  assert(!document.querySelector('.inbox-unread-dot'), 'No red dot when the confirmed count is zero');
  account.loading = true; await render(Bell); assert(!document.querySelector('.inbox-bell'), 'Account checks hide private notification UI');
  account.loading = false; await stop();

  snapshot = {...snapshot, ...count};
  await render(Panel, {owner: 'alice', notifications: true, messages: true, onCounts() {}});
  assert(document.body.textContent.includes('Bob and 2 others'));
  assert(document.body.textContent.includes('<script>paper</script>'));
  assert(!document.querySelector('script'));
  assert(!calls.some(call => call.path === 'inbox/read'), 'Opening the inbox does not mark unseen notifications read');
  deferRead = true; await click(button('Mark all as read'));
  const mark = calls.findLast(call => call.path === 'inbox/read');
  assert.equal(mark.body.through_event_id, 8);
  snapshot = {...snapshot, observed_event_id: 9, notifications: [{...notification, latest_event_id: 9}]};
  await act(async () => readRelease());
  assert(document.querySelector('.inbox-row-dot'), 'A notification arriving after the displayed cutoff remains unread');
  deferRead = false; await stop();

  await render(Messages, {username: 'bob'});
  assert(document.body.textContent.includes('You can send one message until they reply.'));
  await edit('<img src=x onerror=alert(1)> hello');
  failSend = true; await submit();
  const first = sent.at(-1);
  assert.equal(first.body, '<img src=x onerror=alert(1)> hello');
  assert.match(first.request_id, /^[a-f0-9]{32}$/);
  assert(document.querySelector('textarea').readOnly);
  assert(button('Retry send') && !button('Retry send').disabled);
  failSend = false; await submit();
  assert.equal(sent.at(-1).request_id, first.request_id, 'A retry uses the original idempotency key');
  assert.equal(sent.length, 2);
  assert.equal(document.querySelector('textarea').value, '');
  assert(document.querySelector('textarea').disabled, 'Strangers wait after their first message');
  assert(document.body.textContent.includes('<img src=x onerror=alert(1)> hello'));
  assert(!document.querySelector('img'), 'Messages render as escaped plain text');
  assert.equal(localStorage.length, 1); assert.equal(sessionStorage.length, 0, 'Private message drafts never enter browser storage');
  await stop();

  thread = {...thread, messages: [], observed_message_id: 0, can_send: true, awaiting_reply: false};
  failSend = true; acknowledgeLost = true;
  await render(Messages, {username: 'bob'}); await edit('Delivered but response lost'); await submit();
  assert.equal(document.querySelector('textarea').value, '', 'Polling reconciles a delivered request by its outgoing request_id');
  assert(!button('Retry send')); assert.equal(document.querySelectorAll('.message-list li').length, 1);
  failSend = false; acknowledgeLost = false; await stop();

  for (const [status, code] of [[400, undefined], [403, undefined], [404, undefined], [409, 'message_changed']]) {
    thread = {...thread, messages: [], observed_message_id: 0, can_send: true, awaiting_reply: false};
    sendRejection = new SocialError('Message was rejected', status, code);
    await render(Messages, {username: 'bob'}); await edit('Message needing correction'); await submit();
    const rejected = sent.at(-1);
    assert.equal(document.querySelector('textarea').value, 'Message needing correction', 'A definitive rejection preserves the draft');
    assert.equal(document.querySelector('textarea').readOnly, false, 'A definitive rejection allows editing');
    assert(!button('Retry send'));
    sendRejection = undefined; await edit('Corrected message'); await submit();
    assert.notEqual(sent.at(-1).request_id, rejected.request_id, 'Edited rejected messages use a fresh request key');
    assert.equal(sent.at(-1).body, 'Corrected message');
    await stop();
  }

  thread = {...thread, is_mutual_follow: true, can_send: true, awaiting_reply: false, messages: [{id: 2, sender_id: 'bob', body: 'Private Alice-only message', created_at: '2026-10-15T11:01:00Z'}], observed_message_id: 2, unread_count: 1};
  await render(Messages, {username: 'bob'});
  assert(calls.some(call => call.path === 'messages/bob/read' && call.body.through_message_id === 2));
  await edit('x'.repeat(2001)); assert(button('Send').disabled);
  await edit('Mutual followers can reply freely'); assert(!button('Send').disabled);
  deferThread = true; await act(async () => inboxChanged('alice'));
  const staleCall = calls.findLast(call => call.path.startsWith('messages/') && !call.body);
  account.user = {...bob}; await render(Messages, {username: 'alice'});
  assert(!document.body.textContent.includes('Private Alice-only message'));
  assert.equal(staleCall.signal.aborted, true, 'Account switches abort old private reads');
  // A mismatched owner can never become a private snapshot for a newly signed-in account.
  await assert.rejects(() => inboxRequest('bob', 'inbox?summary=1'), /account changed/);
  assert.deepEqual(mergeMessages([{id: 2, body: 'old'}], [{id: 1, body: 'earlier'}, {id: 2, body: 'updated'}]).map(item => item.body), ['earlier', 'updated']);
  console.log('PASS inbox summary/red dot, grouped/escaped notifications, snapshot-safe reads, idempotent messages and lost-response reconciliation, editable rejected drafts, stranger/mutual send rules, limits, plain text, no draft storage, account privacy and aborts');
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); dom.window.close();
  for (const [key, descriptor] of previous) {if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];}
}
