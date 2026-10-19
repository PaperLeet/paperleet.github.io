import { useI18n } from './i18n';
import type { ReportSourceState } from './reportSource';
import { isPassed, type Problem, type Report } from './types';
import { verificationReason } from './browserVerification';

export function BrowserResultActions({ problem, report, sourceState, signedIn, online, ready, conflict, syncError, busy, onSignIn, onVerify, onRefresh, onLab }: {
  problem: Problem; report: Report; sourceState: ReportSourceState; signedIn: boolean; online: boolean; ready: boolean;
  conflict: boolean; syncError: boolean; busy: boolean;
  onSignIn: () => void; onVerify: () => void; onRefresh: () => void; onLab: () => void;
}) {
  const { t } = useI18n();
  const passed = isPassed(problem,report);
  const reason = verificationReason({ passed, sourceState, signedIn, online, ready, conflict, syncError, busy });
  const explanations = {
    busy: t('正在处理判题任务；服务端返回结果后更新账号成绩。', 'A judging task is in progress. Account results update when the server returns its report.'),
    failed: t('先根据失败原因修改代码，再运行完整测试。', 'Use the failure details to improve your code, then run the complete suite.'),
    signin: t('登录后提交，将解答和成绩保存到账号，换设备也能回看。游客草稿自动保存在当前浏览器，不会自动并入账号。', 'Sign in and submit to save solutions and results to your account, ready to revisit on another device. Guest drafts save automatically in this browser and stay separate.'),
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
    <strong>{!signedIn && passed && sourceState === 'current' ? t('测试通过了，把这次解答保存下来', 'Tests passed. Save your solution.') : t('练习结果 · 尚未提交到账号', 'Practice result · not yet submitted to your account')}</strong>
    <p>{explanations[reason]}</p>
    <div className="browser-result-buttons">
      {!signedIn ? <button className="button" type="button" disabled={busy} onClick={onSignIn}>{t('注册 / 登录后提交', 'Join / Sign in to submit')}</button>
        : reason !== 'failed' && <button className="button" type="button" disabled={reason !== 'ready'} onClick={onVerify}>{t('提交并保存解答', 'Submit and save solution')}</button>}
      {reason === 'offline' && <button className="text-button" type="button" onClick={onRefresh}>{t('重试连接', 'Retry connection')}</button>}
      {hasArtifacts && <button className="text-button" type="button" onClick={onLab}>{t('查看本次实验', 'View this run’s experiments')}</button>}
    </div>
  </section>;
}
