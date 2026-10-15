import { useI18n } from "./i18n";
import { withLibraryContext } from "./libraryNavigation";
type Props = Parameters<typeof import("./ModuleNavigator").ModuleNavigator>[0];
/** Native navigation remains available while the enhanced menu loads or retries. */
export function ModuleNavigationFallback({ current, modules, returnTo }: Props) {
  const { t, locale } = useI18n();
  return <div className="problem-navigation module-nav-fallback"><select aria-label={t("选择模块", "Choose module")} value={current.id} onChange={event => {
    const module = modules.find(module => module.id === event.target.value);
    if (module) location.hash = withLibraryContext(module.number,returnTo).slice(1);
  }}>{modules.map((module,index) => <option key={module.id} value={module.id}>{index+1} / {modules.length} · {locale === "en" ? module.translations?.en?.title || module.title : module.title}</option>)}</select></div>;
}
