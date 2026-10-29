import { useEffect, useState } from 'react';

const cacheKey = 'paperleet.github-stars.v1';
const maxAge = 60 * 60 * 1000;
let request: Promise<number | undefined> | undefined;

async function loadStars(): Promise<number | undefined> {
  try {
    const cached = JSON.parse(localStorage.getItem(cacheKey) || 'null');
    const age = Date.now() - cached?.at;
    if (Number.isSafeInteger(cached?.count) && cached.count >= 0 && age >= 0 && age < maxAge) return cached.count;
  } catch { /* Storage is optional. */ }
  try {
    const response = await fetch('https://api.github.com/repos/Hongyang-Du/PaperLeet', {
      signal: AbortSignal.timeout(5000), credentials: 'omit', referrerPolicy: 'no-referrer',
    });
    if (!response.ok) return;
    const { stargazers_count: count } = await response.json();
    if (!Number.isSafeInteger(count) || count < 0) return;
    try { localStorage.setItem(cacheKey, JSON.stringify({ count, at: Date.now() })); } catch { /* Storage is optional. */ }
    return count;
  } catch { /* Keep the repository link usable when GitHub is unavailable. */ }
}

export default function RepositoryStarCount() {
  const [count, setCount] = useState<number | null>();
  useEffect(() => {
    let active = true;
    request ||= loadStars();
    void request.then(value => { if (active) setCount(value ?? null); });
    return () => { active = false; };
  }, []);
  // A pending count keeps its reserved slot; a failed count renders nothing.
  return count === undefined ? '\u00a0' : count === null ? null : count.toLocaleString('en-US', count >= 10_000 ? { notation: 'compact', maximumFractionDigits: 1 } : undefined);
}
