export type Locale = "zh" | "en";

/** A saved choice wins; otherwise follow the primary browser language. */
export function preferredLocale(): Locale {
  try {
    const saved = localStorage.getItem("paperleet.locale");
    if (saved === "zh" || saved === "en") return saved;
  } catch { /* System language still works when storage is unavailable. */ }
  const language = typeof navigator === "undefined"
    ? "en" : navigator.languages?.[0] || navigator.language || "en";
  return /^zh(?:-|$)/i.test(language) ? "zh" : "en";
}

/** Non-React messages follow the rendered page, including unsaved session choices. */
export function currentLocale(): Locale {
  const language = typeof document === "undefined" ? "" : document.documentElement.lang;
  if (/^zh(?:-|$)/i.test(language)) return "zh";
  if (/^en(?:-|$)/i.test(language)) return "en";
  return preferredLocale();
}
