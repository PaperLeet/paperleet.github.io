import AppLink from './AppLink';
import { useI18n } from "./i18n";

/** A smiling black-cat emoji with closed eyes and warm cheeks. */
export function BrandMark() {
  return <svg className="brand-symbol" viewBox="0 0 64 64" aria-hidden="true" focusable="false">
    <circle cx="32" cy="32" r="31" fill="#f3efe6" />
    <path d="M12 27 11.5 16Q11 11 15.5 13.5L24 19Q32 17 40 19L48.5 13.5Q53 11 52.5 16L52 27Q57 33 55 41Q52 54 32 55Q12 54 9 41Q7 33 12 27Z" fill="#161619" />
    <path d="M18.5 34q4-5 8 0m11 0q4-5 8 0" fill="none" stroke="#f3efe6" strokeWidth="2.75" strokeLinecap="round" />
    <ellipse cx="18" cy="39" rx="3" ry="1.8" fill="#f09996" />
    <ellipse cx="46" cy="39" rx="3" ry="1.8" fill="#f09996" />
    <path d="M27 40.5q5 3 10 0c-.5 5-2.5 7.5-5 7.5s-4.5-2.5-5-7.5Z" fill="#f3efe6" />
    <path d="M29.5 46q2.5-2 5 0-2.5 4-5 0Z" fill="#f09996" />
  </svg>;
}

export default function Brand() {
  const { t } = useI18n();
  return <AppLink href="#" className="brand" aria-label={t("PaperLeet 首页", "PaperLeet home")}>
    <BrandMark />
    <span className="brand-wordmark">Paper<span>Leet</span></span>
  </AppLink>;
}
