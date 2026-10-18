import {appLocation} from './appLocation';
import AppLink from './AppLink';
import { useI18n } from "./i18n";
import { readLibraryRoute,withLibraryContext } from "./libraryNavigation";
import {readInterviewRoute} from './interviewNavigation';
type Props = Parameters<typeof import("./ModuleNavigator").ModuleNavigator>[0];
/** Native navigation remains available while the enhanced menu loads or retries. */
export function ModuleNavigationFallback({ current, modules, returnTo,scopeReady=true }: Props) {
  const { t, locale } = useI18n();
  if(readLibraryRoute(returnTo||'')?.filters.company&&(!scopeReady||!modules.some(module=>module.id===current.id)))return <AppLink className="text-button" href={returnTo}>{t('返回公司练习','Back to company practice')}</AppLink>;
  if(returnTo&&readInterviewRoute(returnTo)?.track)return <div className="problem-navigation module-nav-fallback"><AppLink className="text-button" href={returnTo}>{t('返回准备路径','Return to preparation path')}</AppLink></div>;
  return <div className="problem-navigation module-nav-fallback"><select aria-label={t("选择模块", "Choose module")} value={current.id} onChange={event => {
    const module = modules.find(module => module.id === event.target.value);
    if (module) appLocation.hash = withLibraryContext(module.number,returnTo).slice(1);
  }}>{modules.map((module,index) => <option key={module.id} value={module.id}>{index+1} / {modules.length} · {locale === "en" ? module.translations?.en?.title || module.title : module.title}</option>)}</select></div>;
}
