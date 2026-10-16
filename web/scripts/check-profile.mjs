import assert from "node:assert/strict";
import React, { act } from "react";
import { JSDOM } from "jsdom";
import { createServer } from "vite";

const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'http://paperleet.test/#/u/alice', pretendToBeVisual: true });
dom.window.matchMedia = () => ({ matches: false });
const scrolls = [];
dom.window.HTMLElement.prototype.scrollIntoView = function (options) { scrolls.push({ element: this, options }); };
const globals = {
  window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
  HTMLElement: dom.window.HTMLElement, localStorage: dom.window.localStorage,
  sessionStorage: dom.window.sessionStorage, location: dom.window.location,
  requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
  IS_REACT_ACT_ENVIRONMENT: true, __profileFixture: undefined,
};
const previous = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
const timezone = process.env.TZ;
localStorage.setItem('paperleet.locale', 'en');
const errors = [];
dom.window.addEventListener('error', event => errors.push(event.error));
const server = await createServer({
  server: { middlewareMode: true, hmr: false }, appType: "custom",
  optimizeDeps: { noDiscovery: true, include: [] },
  plugins: [{
    name: 'profile-connections-fixture', enforce: 'pre',
    resolveId(source) {
      if (source === './social') return '\0profile-social';
      if (source === './ProfileProgress' || source === './ProfileAchievements') return '\0profile-above-fold';
    },
    load(id) {
      if (id === '\0profile-above-fold') return `import React from 'react';export default function Progress(){return React.createElement('section',{'data-profile-progress':true,style:{height:600}},'Practice progress');}`;
      if (id === '\0profile-social') return `import React from 'react';
        export const useAccount=()=>globalThis.__profileFixture.account;
        export const socialApi=(...args)=>globalThis.__profileFixture.request(...args);
        export const errorMessage=e=>e.message;export class SocialError extends Error{}
        export const Avatar=({user})=>React.createElement('span',null,user.display_name);
        export const profileHref=user=>'#/u/'+user.username;
        export const PersonLink=({user})=>React.createElement('a',{href:profileHref(user)},user.display_name);
        export const DateLabel=({value})=>React.createElement('time',null,value);
        export const SocialNotice=({error})=>React.createElement('p',{role:'alert'},error);
        export const TestAccountBadge=()=>null;`;
    },
  }],
});
let root;
try {
  const { calendarCells } = await server.ssrLoadModule("/src/profile-progress.ts");
  const accepted = { date: "2024-02-29", submissions: 4, attempted: 2, solved: 1, newly_solved: 1 };
  const leap = calendarCells("2024-01-01", "2024-12-31", [accepted]);
  assert.equal(leap.filter((cell) => cell.inRange).length, 366, "Leap day must be included");
  assert.equal(new Set(leap.map((cell) => cell.date)).size, leap.length, "Each date must occupy exactly one cell");
  assert.equal(leap.length % 7, 0, "The calendar must use complete weeks");
  assert.equal(new Date(`${leap[0].date}T00:00:00Z`).getUTCDay(), 0, "Calendar columns start on Sunday");
  assert.deepEqual(leap.find((cell) => cell.date === accepted.date).day, accepted, "Repeated submissions and unique problems must remain distinct");
  assert.equal(leap.find((cell) => cell.date === "2024-03-01").day.submissions, 0, "Inactive dates must not inherit adjacent activity");
  assert.equal(leap[0].inRange, false, "Padding in another year must not become an interactive day");

  for (const timezone of ["America/New_York", "Asia/Shanghai", "Pacific/Auckland"]) {
    process.env.TZ = timezone;
    const cells = calendarCells("2026-03-07", "2026-03-10", []);
    assert.deepEqual(cells.filter((cell) => cell.inRange).map((cell) => cell.date),
      ["2026-03-07", "2026-03-08", "2026-03-09", "2026-03-10"],
      `Date-only cells must not skip or repeat at DST boundaries in ${timezone}`);
  }
  const firstDay = calendarCells("2026-01-01", "2026-01-01", []);
  assert.equal(firstDay.filter((cell) => cell.inRange).length, 1, "A new-year calendar must allow only the current date");
  assert.deepEqual(calendarCells("2026-02-01", "2026-01-01", []), [], "Invalid ranges must not generate activity");
  console.log("PASS profile calendar leap years, week alignment, daily dedup fields, date bounds and DST independence");

  const { createRoot } = await import('react-dom/client');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  const { UserProfile } = await server.ssrLoadModule('/src/Community.tsx');
  const alice = { id: 'alice', username: 'alice', display_name: 'Alice', created_at: '2026-01-01', avatar_color: '#123456' };
  const member = username => ({ ...alice, id: username, username, display_name: username, viewer: { is_self: false, following: false } });
  const profile = { ...alice, counts: { posts: 0, replies: 0, likes: 0, reposts: 0, following: 21, followers: 1, likes_received: 0, reposts_received: 0 }, viewer: { is_self: true, following: false } };
  const requests = [];
  let held, holdKind = '', emptyFollowers = false;
  globalThis.__profileFixture = {
    account: { user: alice, capabilities: {}, loading: false },
    request: async (path, payload) => {
      assert.equal(payload, undefined, 'Opening profile lists must not mutate relationships');
      requests.push(path);
      const url = new URL(path, 'https://fixture.test/');
      if (url.pathname.endsWith('/connections')) {
        const kind = url.searchParams.get('kind'), offset = Number(url.searchParams.get('offset'));
        assert(['following', 'followers'].includes(kind));
        if (kind === holdKind) return new Promise((resolve, reject) => { held = { resolve, reject }; });
        return kind === 'following'
          ? { items: [member(offset ? 'following_last_page' : 'following_bob')], total: 21 }
          : { items: emptyFollowers ? [] : [member('follower_carol')], total: emptyFollowers ? 0 : 1 };
      }
      if (url.pathname.endsWith('/activity')) return { items: [], total: 0 };
      assert.equal(url.pathname, '/community/profiles/alice');
      return profile;
    },
  };
  const render = async () => act(async () => {
    root ??= createRoot(document.querySelector('#root'));
    root.render(React.createElement(LocaleProvider, null, React.createElement(UserProfile, { username: 'alice' })));
  });
  const stop = async () => { await act(async () => root.unmount()); root = undefined; };
  const click = async node => { assert(node); await act(async () => { node.focus(); node.click(); }); };
  const counter = kind => document.querySelector(`.profile-stats button:nth-child(${kind === 'following' ? 1 : 2})`);
  const panel = () => document.querySelector('#profile-community-panel');
  const assertRevealed = kind => {
    const selected = document.querySelector(`#profile-tab-${kind}`);
    assert.equal(selected.getAttribute('aria-selected'), 'true');
    assert.equal(document.activeElement, selected, 'Counter activation must move keyboard focus to the committed selected tab');
    assert.equal(panel().getAttribute('aria-labelledby'), selected.id);
    assert.equal(scrolls.at(-1)?.element, document.querySelector('.profile-community-content'), 'Counter activation must bring the below-fold list section into view');
    assert.equal(scrolls.at(-1)?.options.block, 'start');
    assert(scrolls.some(call => call.element === selected && call.options.inline === 'nearest'), 'The selected tab must also be revealed in the horizontally scrollable mobile tab bar');
  };

  await render();
  assert.equal(document.querySelector('.profile-workspace-settings').getAttribute('href'), '#/achievements?from=profile', 'The profile achievement entry must retain its source across refresh and new-tab navigation');
  assert.equal(scrolls.length, 0, 'Loading a profile must not unexpectedly jump past progress');
  assert.equal(counter('following').getAttribute('aria-controls'), panel().id);
  holdKind = 'following';
  await click(counter('following'));
  assertRevealed('following');
  assert.equal(panel().getAttribute('aria-busy'), 'true');
  assert(panel().textContent.includes('Loading'));
  await act(async () => held.resolve({ items: [member('following_bob')], total: 21 }));
  holdKind = '';
  assert(panel().querySelector('a[href="#/u/following_bob"]'));
  assert.equal(location.hash, '#/u/alice', 'Opening connections stays on the current profile');

  const previousScrolls = scrolls.length, previousRequests = requests.length;
  await click(counter('following'));
  assert(scrolls.length > previousScrolls, 'Repeated clicks on the current category must reveal it again');
  assertRevealed('following');
  assert.equal(requests.length, previousRequests, 'Revealing an already loaded list need not refetch it');
  await click(document.querySelector('[aria-label="Next page"]'));
  assert(panel().textContent.includes('following_last_page'));
  await click(counter('following'));
  assertRevealed('following');
  assert(panel().textContent.includes('following_bob'));
  assert(!panel().textContent.includes('following_last_page'), 'Sidebar shortcuts reopen the first page');
  assert.equal(new URL(requests.at(-1), 'https://fixture.test/').searchParams.get('offset'), '0');

  holdKind = 'followers';
  await click(counter('followers'));
  assertRevealed('followers');
  assert.equal(panel().getAttribute('aria-busy'), 'true');
  assert(!panel().textContent.includes('following_bob'), 'Switching lists must not show stale following rows as followers');
  await act(async () => held.reject(Error('Connections temporarily unavailable')));
  holdKind = '';
  assert(panel().querySelector('[role="alert"]'));
  await click(panel().querySelector('.feed-error button'));
  assert(panel().querySelector('a[href="#/u/follower_carol"]'), 'A visible failed connection list can be retried');

  // Tab navigation itself keeps the reader's position and keyboard behavior.
  let scrollCount = scrolls.length;
  await click(document.querySelector('#profile-tab-following'));
  assert.equal(scrolls.length, scrollCount);
  await act(async () => document.activeElement.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true })));
  assert.equal(document.activeElement.id, 'profile-tab-followers');
  assert.equal(document.activeElement.getAttribute('aria-selected'), 'true');
  assert.equal(scrolls.length, scrollCount, 'Arrow navigation must not force the whole section back to the top');
  for (const kind of ['posts', 'replies']) {
    await click(document.querySelector(`[aria-label="View ${kind}"]`));
    assertRevealed(kind);
    assert.equal(new URL(requests.at(-1), 'https://fixture.test/').searchParams.get('tab'), kind);
  }

  await stop();
  localStorage.setItem('paperleet.locale', 'zh');
  Object.defineProperty(dom.window, 'innerWidth', { value: 390, configurable: true });
  profile.counts.followers = 0; emptyFollowers = true;
  await render();
  assert(counter('followers').textContent.includes('粉丝'));
  await click(counter('followers'));
  assertRevealed('followers');
  assert(panel().textContent.includes('还没有粉丝'), 'A zero count still opens the corresponding empty list');
  assert.equal(document.querySelectorAll('.profile-tabs [tabindex="0"]').length, 1);
  assert.deepEqual(errors, []);
  console.log('PASS profile connection shortcuts: visible selected lists, repeat activation, first-page reset, loading/error/retry/empty states, mobile tab reveal, focus and unchanged normal tab navigation');
} finally {
  if (root) await act(async () => root.unmount());
  await server.close();
  dom.window.close();
  if (timezone === undefined) delete process.env.TZ; else process.env.TZ = timezone;
  for (const [key, descriptor] of previous) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
}
