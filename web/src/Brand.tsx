import brandMark from './assets/brand-mark.svg?no-inline';
import AppLink from './AppLink';
import { useI18n } from "./i18n";

/** A smiling black-cat emoji with closed eyes and warm cheeks. */
export function BrandMark() {
  return <img className="brand-symbol" src={brandMark} width={34} height={34} alt="" aria-hidden="true"/>;
}

export default function Brand() {
  const { t } = useI18n();
  return <AppLink href="#" className="brand" aria-label={t("PaperLeet 首页", "PaperLeet home")}>
    <BrandMark />
    <span className="brand-wordmark">Paper<span>Leet</span></span>
  </AppLink>;
}
