/** Optional cross-tab hints. A notification failure must never fail a save. */
export function browserNotifications(name: string, owner: string, reload: () => void) {
  let channel: BroadcastChannel | undefined, disposed = false;
  const refresh = () => {if (!disposed) reload();};
  const visible = () => {if (document.visibilityState === 'visible') refresh();};
  function disconnect() {
    const previous = channel; channel = undefined;
    try {previous?.close();} catch { /* Focus and visibility refresh still work. */ }
  }
  try {
    if (typeof BroadcastChannel !== 'undefined') {
      channel = new BroadcastChannel(name);
      channel.onmessage = event => {if (event.data?.owner === owner) refresh();};
    }
  } catch {disconnect();}
  window.addEventListener('focus',refresh);
  document.addEventListener('visibilitychange',visible);
  return {
    notify() {
      if (disposed) return;
      try {channel?.postMessage({owner});} catch {disconnect();}
    },
    dispose() {
      disposed = true; disconnect();
      window.removeEventListener('focus',refresh);
      document.removeEventListener('visibilitychange',visible);
    },
  };
}
