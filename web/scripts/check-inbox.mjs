import assert from 'node:assert/strict';
import React, {act} from 'react';
import {JSDOM} from 'jsdom';
import {createServer} from 'vite';

const dom = new JSDOM('<!doctype html><div id="root"></div>', {url: 'http://paperleet.test/#/messages/bob', pretendToBeVisual: true});
const values = {window: dom.window, document: dom.window.document, navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage, location: dom.window.location, CustomEvent: dom.window.CustomEvent, requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window), cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window), IS_REACT_ACT_ENVIRONMENT: true, __inboxFixture: undefined};
const previous = new Map(Object.keys(values).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(values)) Object.defineProperty(globalThis, key, {value, writable: true, configurable: true});
localStorage.setItem('paperleet.locale', 'en');
Object.defineProperty(dom.window.HTMLElement.prototype, 'clientHeight', {configurable: true, get() {return this.classList.contains('message-history') ? 300 : 0;}});
const server = await createServer({cacheDir: 'node_modules/.vite-inbox-check', server: {middlewareMode: true, hmr: false}, appType: 'custom', optimizeDeps: {noDiscovery: true, include: []}, plugins: [{name: 'inbox-account-fixture', enforce: 'pre', resolveId(source) {if (source === './social') return '\0inbox-account';}, load(name) {if (name === '\0inbox-account') return `import React from 'react';export const useAccount=()=>globalThis.__inboxFixture.account;export const socialApi=(...args)=>globalThis.__inboxFixture.request(...args);export const errorMessage=e=>e.message;export class SocialError extends Error {constructor(message,status,code){super(message);this.status=status;this.code=code;}}export const Avatar=({user})=>React.createElement('span',null,user.display_name);export const profileHref=user=>'#/u/'+user.username;export const DateLabel=({value})=>React.createElement('time',null,value);`;}}]});
let root;
try {
  const {createRoot} = await import('react-dom/client');
  const {default: Bell} = await server.ssrLoadModule('/src/InboxBell.tsx');
  const {default: Panel} = await server.ssrLoadModule('/src/InboxPanel.tsx');
  const {default: Messages, mergeMessages} = await server.ssrLoadModule('/src/Messages.tsx');
  const {inboxRequest, inboxChanged, applyInboxReadAll} = await server.ssrLoadModule('/src/inbox.ts');
  const {LocaleProvider} = await server.ssrLoadModule('/src/i18n.tsx');
  const {SocialError} = await server.ssrLoadModule('\0inbox-account');
  const alice = {id: 'alice', username: 'alice', display_name: 'Alice', avatar_color: '#48766b'};
  const bob = {id: 'bob', username: 'bob', display_name: 'Bob', avatar_color: '#48766b'};
  const account = {user: alice, loading: false, error: '', capabilities: {inbox_notifications: true, direct_messages: true}, refresh: async () => {}};
  const count = {user_id: 'alice', unread_count: 2, notification_unread_count: 1, message_unread_count: 1};
  const notification = {id: 'likes:post', kind: 'like', latest_event_id: 8, created_at: '2026-10-15T10:00:00Z', unread: true, actor_count: 3, actors: [bob], post: {id: 'a'.repeat(32), title: '<script>paper</script>'}, reply_id: null, preview: ''};
  let snapshot = {...count, observed_event_id: 8, observed_message_id: 40, notifications: [notification], notifications_next_before: null, conversations: [], conversations_next_before: null};
  let thread = {user_id: 'alice', peer: bob, conversation_id: null, messages: [], next_before: null, observed_message_id: 0, unread_count: 0, can_send: true, awaiting_reply: false, is_mutual_follow: false};
  let failSend = false, acknowledgeLost = false, sendRejection, deferRead = false, readRelease, deferThread = false, threadRelease, failThread = false;
  let deferInbox = false, inboxRelease, deferReadAll = false, readAllRelease, failReadAll = false, loseReadAllResponse = false;
  const calls = [], sent = [];
  globalThis.__inboxFixture = {account, request: async (path, body, signal) => {
    calls.push({path, body, signal});
    if (path === 'inbox?summary=1') return {...snapshot};
    if (path.startsWith('inbox?')) {
      if (deferInbox) return new Promise(resolve => {inboxRelease = resolve;});
      return structuredClone(snapshot);
    }
    if (path === 'inbox/read-all') {
      assert.equal(body.expected_user_id, account.user.id);
      if (deferReadAll) await new Promise(resolve => {readAllRelease = resolve;});
      if (failReadAll) throw Error('Could not mark inbox read');
      const notificationUnread = snapshot.observed_event_id > body.through_event_id ? 1 : 0;
      const messageUnread = snapshot.observed_message_id > body.through_message_id ? 1 : 0;
      snapshot = applyInboxReadAll(snapshot, {...count, notification_unread_count: notificationUnread, message_unread_count: messageUnread, unread_count: notificationUnread + messageUnread}, body);
      if (loseReadAllResponse) { loseReadAllResponse = false; throw Error('Read acknowledgement response lost'); }
      return {...snapshot};
    }
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
  assert(!button('Mark all as read'), 'Older servers keep the notification-only action without advertising all-inbox support');
  deferRead = true; await click(button('Mark notifications as read'));
  const mark = calls.findLast(call => call.path === 'inbox/read');
  assert.equal(mark.body.through_event_id, 8);
  snapshot = {...snapshot, observed_event_id: 9, notifications: [{...notification, latest_event_id: 9}]};
  await act(async () => readRelease());
  assert(snapshot.notification_unread_count===1&&button('Show latest'), 'A notification arriving after the displayed cutoff remains unread and waits behind an explicit refresh');
  await click(button('Show latest'));assert(document.querySelector('.inbox-row-dot'));
  deferRead = false; await stop();

  account.capabilities.inbox_read_all = true;
  const panelProps = {owner: 'alice', notifications: true, messages: true, onCounts() {}};
  deferInbox = true; await render(Panel, panelProps);
  assert(button('Mark all as read')?.disabled, 'The all-read control is visible but disabled while its snapshot loads');
  deferInbox = false; await act(async () => inboxRelease(structuredClone(snapshot)));
  assert(!button('Mark all as read').disabled);
  const allReadControl = button('Mark all as read');
  assert(allReadControl.closest('.inbox-panel-heading'), 'All-read belongs to the shared header');
  await click([...document.querySelectorAll('[role=tab]')].find(node => node.textContent.startsWith('Messages')));
  assert.equal(button('Mark all as read'), allReadControl, 'The same all-read action remains visible in the message tab');
  deferReadAll = true; await click(allReadControl);
  const allMark = calls.findLast(call => call.path === 'inbox/read-all');
  assert.deepEqual(allMark.body, {expected_user_id: 'alice', through_event_id: 9, through_message_id: 40});
  assert(button('Marking…').disabled, 'Repeated all-read requests are disabled in flight');
  assert.equal(snapshot.unread_count, 2, 'Pending acknowledgement never clears unread counters optimistically');
  snapshot = {...snapshot, observed_event_id: 10, observed_message_id: 41, notifications: [{...notification, latest_event_id: 10}]};
  await act(async () => readAllRelease());
  assert.equal(snapshot.notification_unread_count, 1); assert.equal(snapshot.message_unread_count, 1);
  assert(document.body.textContent.includes('New messages remain unread.'));
  assert(!button('Mark all as read').disabled);
  deferReadAll = false;
  failReadAll = true; await click(button('Mark all as read'));
  const failedCutoff = calls.findLast(call => call.path === 'inbox/read-all').body;
  assert(document.querySelector('.inbox-read-all-feedback[role=alert]'));
  assert.equal(snapshot.unread_count, 2, 'Failed writes leave unread counts intact');
  snapshot = {...snapshot, observed_event_id: 11, observed_message_id: 42};
  await act(async () => inboxChanged('alice'));
  failReadAll = false; await click(document.querySelector('.inbox-read-all-feedback button'));
  assert.deepEqual(calls.findLast(call => call.path === 'inbox/read-all').body, failedCutoff, 'A retry keeps both original cutoffs despite later arrivals');
  await click(button('Show latest'));
  await click(button('Mark all as read'));
  assert.equal(snapshot.unread_count, 0);
  assert(button('Mark all as read').disabled, 'An empty inbox keeps a visible disabled all-read action');
  assert(document.body.textContent.includes('All marked as read'));
  await stop();

  snapshot = {...snapshot, ...count, observed_event_id: 12, observed_message_id: 43, notifications: [{...notification, latest_event_id: 12}]};
  await render(Panel, panelProps);
  const writesBeforeLostResponse = calls.filter(call => call.path === 'inbox/read-all').length;
  loseReadAllResponse = true; await click(button('Mark all as read'));
  assert.equal(snapshot.unread_count, 0);
  assert(button('Mark all as read').disabled);
  assert(!document.querySelector('.inbox-read-all-feedback[role=alert]'), 'A later zero-unread snapshot clears an uncertain write error');
  assert(document.body.textContent.includes('All marked as read'), 'Confirmed completion replaces a lost-response retry with success');
  assert.equal(calls.filter(call => call.path === 'inbox/read-all').length, writesBeforeLostResponse + 1, 'Recovery needs no duplicate write');
  await stop();

  // Full-snapshot cutoffs, rather than the displayed rows, cover all conversation pages.
  snapshot = {...snapshot, ...count, observed_event_id: 30, observed_message_id: 80, conversations_next_before: 4, conversations: [{id: 1, peer: bob, last_message: {id: 4, sender_id: 'bob', body: 'Visible preview', created_at: '2026-10-15T10:00:00Z'}, unread_count: 1}]};
  await render(Messages);
  assert(button('Mark all as read').closest('.messages-sidebar-heading'));
  await click(button('Mark all as read'));
  assert.equal(calls.findLast(call => call.path === 'inbox/read-all').body.through_message_id, 80);
  assert.equal(snapshot.message_unread_count, 0);
  await stop();

  snapshot = {...snapshot, ...count}; delete snapshot.observed_message_id;
  await render(Panel, panelProps);
  assert(button('Mark all as read').disabled, 'Mixed-version responses without a message cutoff cannot mark all read');
  await stop(); snapshot.observed_message_id = 80;
  localStorage.setItem('paperleet.locale', 'zh');
  await act(async () => {root = createRoot(document.querySelector('#root')); root.render(React.createElement(LocaleProvider, null, React.createElement(Panel, panelProps)));});
  assert(button('全部标为已读'), 'The shared action is localized');
  await stop(); localStorage.setItem('paperleet.locale', 'en');
  deferReadAll = true; await render(Messages); await click(button('Mark all as read'));
  const oldRead = calls.findLast(call => call.path === 'inbox/read-all');
  account.user = bob; await render(Messages);
  assert(oldRead.signal.aborted, 'Switching accounts cancels an in-flight all-read acknowledgement');
  await stop(); account.user = alice; deferReadAll = false;
  await act(async () => readAllRelease());
  snapshot = {...snapshot, ...count, conversations: [], conversations_next_before: null};

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
  const history = document.querySelector('.message-history');
  Object.defineProperties(history, {scrollHeight: {value: 1000, configurable: true}, clientHeight: {value: 300, configurable: true}});
  history.scrollTop = 100;
  await act(async () => history.dispatchEvent(new dom.window.Event('scroll', {bubbles: true})));
  const latestControl = button('Back to latest messages'); assert(latestControl, 'Older history retains a direct way back to the newest content');
  thread = {...thread, messages: [...thread.messages, {id: 3, sender_id: 'bob', body: 'New message below the viewport', created_at: '2026-10-15T11:02:00Z'}], observed_message_id: 3, unread_count: 1};
  await act(async () => inboxChanged('alice'));
  assert(!calls.some(call => call.path === 'messages/bob/read' && call.body.through_message_id === 3), 'New messages remain unread while browsing older history');
  assert.equal(history.scrollTop, 100, 'Incoming messages do not move an older reading position');
  const draftEditor = document.querySelector('textarea');
  await edit('Keep this unfinished reply'); draftEditor.setSelectionRange(3, 9);
  const jump = button('New messages · Jump to latest'); assert(jump, 'New unread content has an explicit jump action');
  assert.equal(jump, latestControl, 'An incoming message updates the existing control without replacing its focus target');
  await click(jump);
  assert(calls.some(call => call.path === 'messages/bob/read' && call.body.through_message_id === 3), 'Jumping to the newest message marks only the observed snapshot read');
  assert.equal(document.activeElement, history, 'The jump leaves keyboard focus in message history');
  assert.equal(history.getAttribute('tabindex'), '0', 'Message history remains reachable for keyboard scrolling');
  assert.equal(document.querySelector('textarea'), draftEditor);
  assert.equal(draftEditor.value, 'Keep this unfinished reply');
  assert.equal(draftEditor.selectionStart, 3); assert.equal(draftEditor.selectionEnd, 9);
  assert(!button('New messages · Jump to latest'), 'The cue disappears after reaching the newest message');
  history.scrollTop = 100;
  await act(async () => history.dispatchEvent(new dom.window.Event('scroll', {bubbles: true})));
  thread = {...thread, messages: [...thread.messages, {id: 4, sender_id: 'bob', body: 'Unread while hidden', created_at: '2026-10-15T11:03:00Z'}], observed_message_id: 4, unread_count: 1};
  await act(async () => inboxChanged('alice'));
  Object.defineProperty(document, 'visibilityState', {configurable: true, value: 'hidden'});
  history.scrollTop = 700;
  await act(async () => history.dispatchEvent(new dom.window.Event('scroll', {bubbles: true})));
  assert(!calls.some(call => call.path === 'messages/bob/read' && call.body.through_message_id === 4), 'Hidden conversations cannot mark messages read');
  Object.defineProperty(document, 'visibilityState', {configurable: true, value: 'visible'});
  await act(async () => {history.dispatchEvent(new dom.window.Event('scroll', {bubbles: true})); history.dispatchEvent(new dom.window.Event('scroll', {bubbles: true}));});
  assert.equal(calls.filter(call => call.path === 'messages/bob/read' && call.body.through_message_id === 4).length, 1, 'Repeated scroll events share the in-flight read acknowledgement');
  Object.defineProperty(history, 'scrollHeight', {value: 1300, configurable: true});
  thread = {...thread, messages: [...thread.messages, {id: 5, sender_id: 'bob', body: 'Visible after following the live tail', created_at: '2026-10-15T11:04:00Z'}], observed_message_id: 5, unread_count: 1};
  await act(async () => inboxChanged('alice'));
  assert(calls.some(call => call.path === 'messages/bob/read' && call.body.through_message_id === 5), 'Following the live tail scrolls to new content before marking it read');
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
  await stop();
  account.user = null; const guestReads = calls.length;
  await render(Messages, {username: 'bob'});
  assert(document.querySelector('.messages-welcome'), 'Guests receive a complete conversation entry state');
  const signIn = [...document.querySelectorAll('a')].find(link => link.textContent === 'Sign in');
  assert.equal(signIn.getAttribute('href'), '#/account/continue/messages%2Fbob');
  assert.equal(document.querySelector('.messages-welcome-back').getAttribute('href'), '#/community');
  assert.equal(calls.length, guestReads, 'The guest introduction never reads private messages');
  console.log('PASS inbox summary/red dot, grouped/escaped notifications, snapshot-safe notification reads, newest-visible-message acknowledgements, explicit new-message jump with draft/selection and keyboard focus preservation, guest entry/login returns, preserved history scroll, hidden/pending-read guards, idempotent messages and lost-response reconciliation, editable rejected drafts, stranger/mutual send rules, limits, plain text, no draft storage, account privacy and aborts');
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); dom.window.close();
  for (const [key, descriptor] of previous) {if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];}
}
