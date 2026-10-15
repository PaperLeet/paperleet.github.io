import { Github } from "lucide-react";
import { useI18n } from "./i18n";

export default function RepositoryLink() {
  const { t } = useI18n();
  const label = t("GitHub 开源仓库", "GitHub repository");
  return (
    <a className="repository-link" href="https://github.com/Hongyang-Du/PaperLeet"
      target="_blank" rel="noopener noreferrer" title={label} aria-label={label}>
      <Github size={18} aria-hidden="true" />
    </a>
  );
}
