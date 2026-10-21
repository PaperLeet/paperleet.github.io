import AppLink from './AppLink';
import { deferredFeature } from './deferredFeature';
import { useI18n } from "./i18n";

const StarCount = deferredFeature(() => import('./RepositoryStarCount'), ['', ''], {
  loading: () => '\u00a0', failure: () => null,
});

export default function RepositoryLink() {
  const { t } = useI18n();
  const label = t("离线版仓库（GitHub）", "Offline repository (GitHub)");
  return (
    <AppLink className="repository-link" href="https://github.com/Hongyang-Du/PaperLeet"
      target="_blank" rel="noopener noreferrer" title={label} aria-label={label}>
      <span className="repository-icon" aria-hidden="true" />
      <span className="repository-stars"><span className="repository-star" aria-hidden="true">★</span><span className="repository-star-count"><StarCount /></span></span>
    </AppLink>
  );
}
