import assert from 'node:assert/strict';
import React, { act } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

const dom = new JSDOM('<!doctype html><button id="upload">Upload picture</button><div id="root"></div>', { url: 'https://paperleet.test/' });
const names = ['window', 'document', 'navigator', 'HTMLElement', 'Image', 'createImageBitmap', 'fetch', 'IS_REACT_ACT_ENVIRONMENT'];
const originals = new Map(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
for (const name of ['window', 'document', 'navigator', 'HTMLElement']) Object.defineProperty(globalThis, name, { value: dom.window[name], configurable: true, writable: true });
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
globalThis.fetch = () => { throw Error('Cropping must never upload or fetch'); };
const draws = [], images = [];
let decode;
globalThis.createImageBitmap = async () => {
  const image = { width: 800, height: 400, closed: 0, close() { this.closed++; } };
  images.push(image);
  if (decode) await new Promise(resolve => { decode.resolve = resolve; });
  return image;
};
window.HTMLCanvasElement.prototype.getContext = function () {
  const canvas = this;
  return { fillRect() {}, drawImage(...args) { draws.push({ canvas, args }); } };
};
window.HTMLCanvasElement.prototype.toDataURL = function (type, quality) {
  assert.equal(type, 'image/jpeg'); assert.equal(quality, 0.9);
  assert.equal(this.width, 256); assert.equal(this.height, 256);
  return 'data:image/jpeg;base64,eA==';
};
window.HTMLElement.prototype.setPointerCapture = function (id) { this.capturedPointer = id; };
const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
let root;
try {
  const { default: AvatarCropper } = await server.ssrLoadModule('/src/AvatarCropper.tsx');
  const { avatarCrop, decodeProfileAvatar, prepareProfileAvatar, AVATAR_MAX_FILE_BYTES } = await server.ssrLoadModule('/src/profileAvatar.ts');
  assert.deepEqual(avatarCrop(800, 400), { x: 200, y: 0, side: 400 });
  assert.deepEqual(avatarCrop(400, 800, { zoom: 2, centerX: 0, centerY: 1 }), { x: 0, y: 600, side: 200 });
  assert.deepEqual(avatarCrop(800, 400, { zoom: 100, centerX: -20, centerY: 20 }), { x: 0, y: 300, side: 100 });
  assert.deepEqual(avatarCrop(800, 400, { zoom: NaN, centerX: Infinity, centerY: NaN }), { x: 200, y: 0, side: 400 });
  for (const dimensions of [[0, 1], [1, NaN], [-1, 8], [Infinity, 9]]) assert.throws(() => avatarCrop(...dimensions));
  for (const file of [{ type: 'image/svg+xml', size: 10 }, { type: 'image/jpeg', size: 0 }, { type: 'image/png', size: AVATAR_MAX_FILE_BYTES + 1 }]) await assert.rejects(decodeProfileAvatar(file));
  assert.equal(images.length, 0, 'Invalid files are rejected before any decode');
  await prepareProfileAvatar({ type: 'image/webp', size: 10 }, { zoom: 2, centerX: 0.25, centerY: 0.75 });
  assert.deepEqual(draws.at(-1).args.slice(1), [100, 200, 200, 200, 0, 0, 256, 256]);
  assert.equal(images.at(-1).closed, 1);

  const { createRoot } = await import('react-dom/client');
  root = createRoot(document.querySelector('#root'));
  const file = new window.File(['image'], 'portrait.png', { type: 'image/png' });
  const confirmations = [], errors = [];
  let cancels = 0;
  const close = () => root.render(null);
  const props = { file, onConfirm: value => { confirmations.push(value); close(); }, onCancel: () => { cancels++; close(); }, onError: error => { errors.push(error); close(); } };
  const render = async (overrides = {}) => {
    document.querySelector('#upload').focus();
    await act(async () => root.render(React.createElement(AvatarCropper, { ...props, ...overrides })));
  };
  const key = async (element, value, options = {}) => act(async () => element.dispatchEvent(new window.KeyboardEvent('keydown', { key: value, bubbles: true, cancelable: true, ...options })));
  const pointer = async (element, type, x, y, pointerType = 'mouse', pointerId = 42, isPrimary = true) => act(async () => {
    const event = new window.MouseEvent(type, { bubbles: true, cancelable: true, button: 0, clientX: x, clientY: y });
    Object.defineProperties(event, { pointerId: { value: pointerId }, pointerType: { value: pointerType }, isPrimary: { value: isPrimary } });
    element.dispatchEvent(event);
  });
  const click = async element => act(async () => element.dispatchEvent(new window.MouseEvent('click', { bubbles: true })));
  const cropArgs = () => draws.at(-1).args.slice(1, 4).map(value => Math.round(value * 100) / 100);
  await render();
  const dialog = document.querySelector('[role="dialog"]');
  assert.equal(document.activeElement, dialog, 'The modal receives focus');
  assert.equal(dialog.getAttribute('aria-modal'), 'true');
  assert.equal(document.body.style.overflow, 'hidden');
  const buttons = [...dialog.querySelectorAll('button')];
  await key(dialog, 'Tab');
  assert.equal(document.activeElement, buttons[0]);
  await key(buttons[0], 'Tab', { shiftKey: true });
  assert.equal(document.activeElement, buttons.at(-1), 'Shift-Tab remains inside the dialog');
  const frame = dialog.querySelector('[role="group"]');
  frame.getBoundingClientRect = () => ({ width: 320 });
  assert.deepEqual(cropArgs(), [200, 0, 400]);
  await pointer(frame, 'pointerdown', 100, 100, 'touch');
  await pointer(frame, 'pointerdown', 140, 100, 'touch', 43, false);
  await pointer(frame, 'pointermove', 240, 100, 'touch', 43, false);
  assert.deepEqual(cropArgs(), [200, 0, 400], 'A second finger cannot replace the active drag');
  await pointer(frame, 'pointerup', 240, 100, 'touch', 43, false);
  await pointer(frame, 'pointermove', 180, 100, 'touch');
  assert.deepEqual(cropArgs(), [100, 0, 400]);
  assert.equal(frame.capturedPointer, 42, 'Touch drags capture their pointer');
  await pointer(frame, 'pointercancel', 180, 100, 'touch');
  await pointer(frame, 'pointermove', 260, 100, 'touch');
  assert.deepEqual(cropArgs(), [100, 0, 400], 'Cancelled gestures cannot keep moving the crop');
  await key(frame, 'ArrowRight');
  assert.deepEqual(cropArgs(), [92, 0, 400]);
  const slider = dialog.querySelector('input[type="range"]');
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(slider, '2');
    slider.dispatchEvent(new window.Event('input', { bubbles: true }));
  });
  assert.equal(dialog.querySelector('output').textContent, '200%');
  assert.deepEqual(cropArgs(), [192, 100, 200]);
  await click(buttons.at(-1));
  assert.deepEqual(confirmations, ['data:image/jpeg;base64,eA==']);
  assert.deepEqual(draws.at(-1).args.slice(1), [192, 100, 200, 200, 0, 0, 256, 256], 'Export uses the selected preview pixels');
  assert.equal(images.at(-1).closed, 1);
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.equal(document.activeElement.id, 'upload');
  assert.equal(document.body.style.overflow, '');

  await render();
  await key(document.querySelector('[role="dialog"]'), 'Escape');
  assert.equal(cancels, 1);
  assert.equal(images.at(-1).closed, 1);
  assert.equal(confirmations.length, 1, 'Escape must not stage an avatar');
  assert.equal(document.activeElement.id, 'upload');
  decode = {};
  await render();
  assert(document.querySelector('button.primary').disabled, 'Cannot confirm before decoding finishes');
  await click(document.querySelector('button[aria-label="Cancel crop"]'));
  await act(async () => decode.resolve());
  assert.equal(cancels, 2);
  assert.equal(images.at(-1).closed, 1, 'A decode that resolves after cancel still releases the bitmap');
  assert.equal(confirmations.length, 1);
  assert.equal(errors.length, 0, 'Late completion cannot surface a stale error');
  decode = undefined;
  await render({ file: new window.File(['bad'], 'bad.svg', { type: 'image/svg+xml' }) });
  assert.equal(errors.length, 1);
  assert.equal(errors[0].code, 'type');
  assert.equal(document.querySelector('[role="dialog"]'), null);

  // Safari-style fallback must revoke its object URL immediately on cancellation,
  // even if the browser's image decode promise never settles.
  const originalCreateUrl = URL.createObjectURL, originalRevokeUrl = URL.revokeObjectURL;
  const bitmapDecoder = globalThis.createImageBitmap;
  try {
    globalThis.createImageBitmap = undefined;
    const revoked = [];
    URL.createObjectURL = () => 'blob:avatar-fixture';
    URL.revokeObjectURL = value => revoked.push(value);
    globalThis.Image = class { decode() { return new Promise(() => {}); } };
    const controller = new AbortController();
    const pending = decodeProfileAvatar(file, controller.signal);
    controller.abort();
    await assert.rejects(pending);
    assert.deepEqual(revoked, ['blob:avatar-fixture']);
  } finally {
    URL.createObjectURL = originalCreateUrl; URL.revokeObjectURL = originalRevokeUrl;
    globalThis.createImageBitmap = bitmapDecoder;
  }
  console.log('PASS avatar square crop bounds, exact selected pixels, local-only export, touch/keyboard/zoom controls, focus trap, cancel and pending-decode cleanup');
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); dom.window.close();
  for (const [name, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  }
}
