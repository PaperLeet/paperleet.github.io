import { Github } from "lucide-react";
import { useI18n } from "./i18n";

export default function RepositoryLink() {
  const { t } = useI18n();
  const label = t("网站源码（GitHub）", "Website source (GitHub)");
  return (
    <a className="repository-link" href="https://github.com/PaperLeet/paperleet.github.io"
      target="_blank" rel="noopener noreferrer" title={label} aria-label={label}>
      <Github size={18} aria-hidden="true" />
    </a>
  );
}
