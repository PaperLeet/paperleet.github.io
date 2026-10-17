import type {Problem} from './types';
import {useI18n} from './i18n';

/** Keep the current problem identifiable without blocking its statement or code. */
export function ProblemOverviewFallback({problem,retry}:{problem:Pick<Problem,'number'|'title'>;retry?:()=>void}) {
  const {t}=useI18n();
  return <div className="problem-heading">
    <h1>{problem.number}. {problem.title}</h1>
    {retry ? <button className="text-button" type="button" onClick={retry}>{t('重试题目信息','Retry problem details')}</button> : <p role="status" aria-busy="true">{t('正在加载题目信息…','Loading problem details…')}</p>}
  </div>;
}
