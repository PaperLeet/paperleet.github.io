import React from 'react';
import { renderToString } from 'react-dom/server';

/** Render a deliberate saved preference instead of relying on the host's language. */
export async function localizedRenderer(server, locale, render = renderToString) {
  if (locale !== 'zh' && locale !== 'en') throw new Error('A test locale is required');
  const { LocaleProvider } = await server.ssrLoadModule('/src/i18n.tsx');
  return (element) => {
    const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    let storage;
    try { storage = globalThis.localStorage; } catch { /* Storage can be unavailable in Node. */ }
    const fixture = new Proxy(storage || {}, {
      get(target, property) {
        if (property === 'getItem') return key => key === 'paperleet.locale' ? locale : storage?.getItem(key) ?? null;
        const value = Reflect.get(target, property, target);
        return typeof value === 'function' ? value.bind(target) : value;
      },
    });
    Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: fixture });
    try {
      return render(React.createElement(LocaleProvider, null, element));
    } finally {
      if (previous) Object.defineProperty(globalThis, 'localStorage', previous);
      else delete globalThis.localStorage;
    }
  };
}
