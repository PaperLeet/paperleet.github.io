import {ArrowLeft,BookOpen,Terminal} from 'lucide-react';
import AppLink from './AppLink';
import {useI18n} from './i18n';
/** Keep stale-link recovery out of the first usable catalog list. */
export default function CatalogFeedback(props:{path:string;href:string;label:string}|{error:string;onRetry:()=>void}) {
  const {t}=useI18n();
  if('error' in props)return <main className="load-error"><Terminal size={35}/><h2>{t('题库暂时不可用','Catalog unavailable')}</h2><p>{props.error}</p><button className="button" onClick={props.onRetry}>{t('重新加载','Reload')}</button></main>;
  const {path,href,label}=props;
  return <main className="route-not-found"><BookOpen size={32}/><h1>{t('找不到这篇论文或模块','Paper or module not found')}</h1><p>{t('链接可能已失效，或内容已经调整。','This link may be outdated or the content may have changed.')}</p><code>{path.slice(0,120)}</code><AppLink className="button" href={href}><ArrowLeft size={15}/>{label}</AppLink></main>;
}
