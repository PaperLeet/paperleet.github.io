import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { preferredLocale, type Locale } from "./locale";

export type { Locale } from "./locale";
type I18n = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (zh: string, en: string) => string;
};
const Context = createContext<I18n>({
  locale: "en",
  setLocale: () => {},
  t: (_zh, en) => en,
});
export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, updateLocale] = useState<Locale>(preferredLocale);
  function setLocale(next: Locale) {
    updateLocale(next);
    try {
      localStorage.setItem("paperleet.locale", next);
    } catch {
      /* Keep the session usable without storage. */
    }
  }
  useEffect(() => {
    document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
  }, [locale]);
  return (
    <Context.Provider
      value={{ locale, setLocale, t: (zh, en) => (locale === "zh" ? zh : en) }}
    >
      {children}
    </Context.Provider>
  );
}
export const useI18n = () => useContext(Context);
