import AppLink from './AppLink';
import { useI18n } from "./i18n";
import brandMark from './assets/paperleet-naiwa.png';

/** The original round Naiwa face, shared by both interface themes. */
export function BrandMark() {
  return <img className="brand-symbol" src={brandMark} width={35} height={35} alt="" aria-hidden="true" />;
}

export default function Brand() {
  const { t } = useI18n();
  return <AppLink href="#" className="brand" aria-label={t("PaperLeet 首页", "PaperLeet home")}>
    <BrandMark />
    <span className="brand-wordmark">Paper<span>Leet</span></span>
  </AppLink>;
}
