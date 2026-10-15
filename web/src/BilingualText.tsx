import { useI18n } from './i18n';
import './bilingual-text.css';

/** Reserve both translations' natural size while exposing only the active one. */
export function BilingualText({ zh, en }: { zh: string; en: string }) {
  const { locale } = useI18n();
  return <span className="bilingual-text">
    <span lang="zh-CN" aria-hidden={locale !== 'zh'}>{zh}</span>
    <span lang="en" aria-hidden={locale !== 'en'}>{en}</span>
  </span>;
}
