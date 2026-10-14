import { useI18n } from './i18n';
import type { ReportSourceState } from './reportSource';
import { isPassed, type Problem, type Report } from './types';
import { verificationReason } from './browserVerification';

export function BrowserResultActions({ problem, report, sourceState, signedIn, online, ready, conflict, syncError, busy, error, onSignIn, onVerify, onRefresh, onLab }: {
  problem: Problem; report: Report; sourceState: ReportSourceState; signedIn: boolean; online: boolean; ready: boolean;
  conflict: boolean; syncError: boolean; busy: boolean; error?: string;
  onSignIn: () => void; onVerify: () => void; onRefresh: () => void; onLab: () => void;
}) {
  const { t } = useI18n();
  const reason = verificationReason({ passed: isPassed(problem,report), sourceState, signedIn, online, ready, conflict, syncError, busy });
  const explanations = {
    busy: t('正在处理判题任务；服务端返回结果后更新账号成绩。', 'A judging task is in progress. Account results update when the server returns its report.'),
    failed: t('先根据失败原因修改代码，再运行完整测试。', 'Use the failure details to improve your code, then run the complete suite.'),
    signin: t('登录后可确认账号成绩。本题代码会暂存在本标签页，登录后由你决定是否导入。', 'Sign in to record account results. This module’s code stays in this tab until you choose to import it after signing in.'),
    workspace: t('正在加载账号工作区，稍后可确认。', 'Your account workspace is loading. Verification will be available shortly.'),
    conflict: t('请先解决下方的账号草稿版本冲突。', 'Resolve the account draft conflict below before verifying.'),
    sync: t('草稿同步尚未完成，请先重试同步。', 'Draft synchronization needs attention. Retry syncing first.'),
    changed: t('代码已改变；请重新运行完整测试后确认当前代码。', 'Code changed. Run the complete suite again before verifying the current code.'),
    version: t('正在核对代码版本；无法确认时可直接选择服务端运行。', 'The code version is being checked. You can select Server directly if the version cannot be confirmed.'),
    offline: t('服务端暂时不可用，本机测试结果仍可查看。', 'The server is unavailable. Your local test results remain available.'),
    ready: t('服务端会重新执行这份代码；通过后计入账号进度与统计。', 'The server reruns this source. A passing result records account progress and statistics.'),
  };
  const hasArtifacts = report.tests.some(test => Object.keys(test.artifacts || {}).length);
  return <section className="browser-result-actions" aria-label={t('确认账号成绩', 'Record account results')}>
    <strong>{t('浏览器本地结果 · 本次未写入账号', 'Local browser run · account progress unchanged')}</strong>
    <p>{explanations[reason]}</p>
    <div className="browser-result-buttons">
      {!signedIn ? <button className="button" type="button" disabled={busy} onClick={onSignIn}>{t('保留本题代码并登录', 'Keep this code and sign in')}</button>
        : reason !== 'failed' && <button className="button" type="button" disabled={reason !== 'ready'} onClick={onVerify}>{t('服务端确认成绩', 'Verify on server')}</button>}
      {reason === 'offline' && <button className="text-button" type="button" onClick={onRefresh}>{t('重试连接', 'Retry connection')}</button>}
      {hasArtifacts && <button className="text-button" type="button" onClick={onLab}>{t('查看本次实验', 'View this run’s experiments')}</button>}
    </div>
    {error && <p role="alert">{t('无法暂存草稿。请先下载代码备份，再使用页面顶部的登录入口。', 'The draft could not be held for sign-in. Download a backup before using the sign-in link at the top of the page.')}</p>}
  </section>;
}
