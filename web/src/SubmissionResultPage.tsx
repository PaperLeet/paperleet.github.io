import AppLink from './AppLink';
import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, Check, CheckCircle2, ChevronDown, Clock3, Code2, FlaskConical, FolderCheck, LoaderCircle, RefreshCw, TriangleAlert, X } from 'lucide-react';
import { apiFetch } from './apiClient';
import { signInHref } from './accountNavigation';
import { deferredFeature } from './deferredFeature';
import { useI18n } from './i18n';
import { withLibraryContext } from './libraryNavigation';
import { useAccount } from './social';
import { loadSubmissionResult, submissionFailurePreview, type SubmissionResultState } from './submissionResultData';
import type { Problem } from './types';
import { runtimeMs, verdictKind } from './feedbackVerdict';
import {SubmittedSourceDisclosure} from './SubmittedSource';
import './submission-result.css';

const ResultReport = deferredFeature(() => import('./ReportView'), ['提交测试详情', 'submission test details']);

type Props = {
  runId: string;
  problems: Problem[];
  libraryHref?: string;
  nextProblemFor?: (problem: Problem) => Problem | undefined;
};

export default function SubmissionResultPage({ runId, problems, libraryHref = '#', nextProblemFor }: Props) {
  const { t, locale } = useI18n();
  const { user, loading: accountLoading, refresh: refreshAccount } = useAccount();
  const [retry, setRetry] = useState(0);
  const scope = JSON.stringify([user?.id, accountLoading, runId, retry]);
  const [view, setView] = useState<{ scope: string; state: SubmissionResultState }>();
  const [expanded, setExpanded] = useState<string>();
  const heading = useRef<HTMLHeadingElement>(null);
  // Effects run after render: never leave a previous account's result visible for that frame.
  const state: SubmissionResultState = view?.scope === scope ? view.state : { status: 'loading' };

  useEffect(() => {
    if (accountLoading || !user) return;
    const reader = loadSubmissionResult(user.id, runId,
      (path, signal) => apiFetch(path, { signal, cache: 'no-store' }),
      next => setView({ scope, state: next }));
    return () => reader.dispose();
  }, [scope, user?.id, accountLoading, runId]);

  useEffect(() => {
    if (state.status === 'ready') heading.current?.focus({ preventScroll: true });
  }, [scope, state.status]);

  const returnLink = <AppLink className="submission-result-back" href={libraryHref}><ArrowLeft size={16} />{t('返回题库', 'Back to papers')}</AppLink>;
  const loading = accountLoading || (!!user && state.status === 'loading');
  const needsAuth = !accountLoading && (!user || state.status === 'auth');
  if (loading || needsAuth || state.status !== 'ready') {
    const missing = state.status === 'missing';
    const identity = state.status === 'identity';
    return <main className="submission-result-page">
      {returnLink}
      <section className="submission-result-unavailable" aria-busy={loading}>
        <div className="submission-result-empty-icon" aria-hidden="true">{loading ? <LoaderCircle size={32} className="submission-result-spinner" /> : <FlaskConical size={32} />}</div>
        <h1>{loading ? t('正在读取提交结果', 'Loading your result') : needsAuth ? t('登录后查看提交结果', 'Sign in to view your result') : missing ? t('找不到这次提交', 'Submission not found') : identity ? t('账号已切换', 'Your account has changed') : t('结果暂时无法加载', 'Your result could not load')}</h1>
        <p role={loading ? 'status' : undefined}>{loading ? t('正在从你的账号读取这次提交记录…', 'Retrieving this submission from your account…') : needsAuth ? t('提交的代码和判题结果仅你本人可见。', 'Your submitted code and judge results are private to your account.') : missing ? t('这条记录可能不存在，或属于另一个账号。', 'This record may not exist or may belong to another account.') : identity ? t('请刷新账号信息，再查看这次提交。', 'Refresh your account before viewing this submission.') : t('这不会重新提交代码。请重试读取已有记录。', 'Retry to read the saved record. Your code will not be submitted again.')}</p>
        {needsAuth ? <AppLink className="button" href={signInHref(`#/submission/${runId}`)}>{t('登录', 'Sign in')}<ArrowRight size={16} /></AppLink> : !loading && !missing && <button className="button" type="button" onClick={() => {
          if (identity) void refreshAccount();
          setRetry(value => value + 1);
        }}><RefreshCw size={16} />{t('重试', 'Retry')}</button>}
      </section>
    </main>;
  }

  const { run } = state;
  const problem = problems.find(candidate => candidate.id === run.problem);
  const title = problem ? (locale === 'en' ? problem.translations?.en?.title || problem.title : problem.title) : run.problem;
  const next = problem ? nextProblemFor ? nextProblemFor(problem) : [...problems].sort((a, b) => a.number - b.number).find(candidate => candidate.number > problem.number && !candidate.module.optional) : undefined;
  const problemHref = problem ? withLibraryContext(problem.number, libraryHref) : `#/${run.problem}`;
  const passed = run.report.tests.filter(test => test.passed).length;
  const total = run.report.tests.length;
  const runtime = runtimeMs(run.report);
  const firstFailure = run.report.tests.find(test => !test.passed);
  const incomplete = !!run.report.error;
  const partial = !run.accepted && !incomplete && total > 0 && passed === total;
  const failure = run.report.error || firstFailure?.message;
  const statusName = { accepted: t('通过', 'Accepted'), incomplete: t('判题未完成', 'Judge incomplete'), 'not-accepted': t('未通过', 'Not accepted'), runtime: t('执行出错', 'Runtime Error'), wrong: t('解答错误', 'Wrong Answer') }[verdictKind(run.accepted, run.report)];
  const detailOpen = expanded === scope;
  const submittedAt = new Date(run.created_at);
  const dateLocale = locale === 'zh' ? 'zh-CN' : 'en-US';

  return <main className={`submission-result-page ${run.accepted ? 'is-accepted' : 'is-failed'}`}>
    <div className="submission-result-navigation">
      {returnLink}
      <span>{t('提交结果', 'Submission result')}</span>
    </div>
    <section className="submission-result-card" aria-labelledby="submission-result-title">
      <div className="submission-result-hero">
        <div className="submission-result-mark" aria-hidden="true">
          <span className="submission-result-halo" />
          {run.accepted ? <Check size={46} strokeWidth={2.8} /> : incomplete ? <TriangleAlert size={40} strokeWidth={1.8} /> : <X size={43} strokeWidth={2.4} />}
          {run.accepted && <><i /><i /><i /><i /></>}
        </div>
        <div className="submission-result-heading">
          <span className="submission-result-eyebrow">{t('提交完成', 'Submission complete')}</span>
          <h1 id="submission-result-title" ref={heading} tabIndex={-1}>{statusName}</h1>
          <p>{run.accepted ? t(`全部 ${total} 项测试通过。`, `All ${total} tests passed.`) : incomplete ? t('判题没有完整结束，查看下方原因后可以重新提交。', 'The judge could not finish. Review the details before trying again.') : partial ? t('本次运行的测试已通过，但未覆盖全部必需测试。请返回代码提交完整测试。', 'These tests passed, but this attempt did not cover every required test. Return to your code and submit the full suite.') : t(`通过 ${passed} / ${total} 项测试。修正后再试一次。`, `${passed} of ${total} tests passed. Review the failures and try again.`)}</p>
        </div>
        <div className="submission-result-saved"><FolderCheck size={15} />{t('已保存到账号', 'Saved to your account')}</div>
      </div>

      <AppLink className="submission-result-problem" href={problemHref}>
        <span className="submission-result-problem-icon"><Code2 size={21} /></span>
        <span><small>{t('本次题目', 'This problem')}</small><strong>{problem ? `${problem.number}. ` : ''}{title}</strong></span>
        <ArrowRight size={18} />
      </AppLink>

      <dl className="submission-result-metrics">
        <div><dt><CheckCircle2 size={15} />{t('测试通过', 'Tests passed')}</dt><dd>{passed}<span> / {total}</span></dd></div>
        {runtime !== undefined && <div><dt><Clock3 size={15} />{t('运行时间', 'Runtime')}</dt><dd>{runtime < 1 ? '< 1' : runtime < 1000 ? Math.round(runtime) : (runtime / 1000).toFixed(2)}<span>{runtime < 1000 ? ' ms' : ' s'}</span></dd></div>}
        <div className="submission-result-date"><dt>{t('提交时间', 'Submitted')}</dt><dd><time dateTime={run.created_at}>{submittedAt.toLocaleTimeString(dateLocale, { hour: '2-digit', minute: '2-digit', second: '2-digit' })}<span>{submittedAt.toLocaleDateString(dateLocale, { month: 'short', day: 'numeric', year: 'numeric' })}</span></time></dd></div>
      </dl>

      {!run.accepted && (firstFailure || failure) && <div className="submission-result-failure">
        <div><TriangleAlert size={17} /><strong>{incomplete ? t('判题信息', 'Judge details') : t('先看这项测试', 'Start with this test')}</strong></div>
        {firstFailure && <code>{firstFailure.id}</code>}
        {firstFailure?.why && <p>{firstFailure.why}</p>}
        {failure && <pre>{submissionFailurePreview(failure)}</pre>}
      </div>}

      <div className="submission-result-actions">
        {run.accepted && next ? <AppLink className="button submission-result-primary" href={withLibraryContext(next.number, libraryHref)}>{t('下一道题', 'Next problem')}<ArrowRight size={17} /></AppLink> : <AppLink className="button submission-result-primary" href={problemHref}><Code2 size={17} />{t('返回代码', 'Back to code')}</AppLink>}
        {run.accepted && next ? <AppLink className="button subtle" href={problemHref}><Code2 size={16} />{t('返回代码', 'Back to code')}</AppLink> : <AppLink className="button subtle" href={libraryHref}>{t('浏览题库', 'Browse papers')}<ArrowRight size={16} /></AppLink>}
      </div>
    </section>

    <SubmittedSourceDisclosure key={scope} scope={scope} code={run.code}/>

    <section className="submission-result-details">
      <button type="button" className="submission-result-details-toggle" aria-expanded={detailOpen} aria-controls="submission-result-tests" onClick={() => setExpanded(detailOpen ? undefined : scope)}>
        <span><FlaskConical size={18} />{t('完整测试结果', 'Full test results')}</span>
        <span>{passed} / {total}<ChevronDown size={18} /></span>
      </button>
      {detailOpen && <div id="submission-result-tests" className="submission-result-tests">
        {problem ? <ResultReport report={run.report} problem={problem} sourceState="current" source={run.code} accepted={run.accepted} /> : <ul>{run.report.tests.map((test, index) => <li key={test.nodeid || `${test.id}:${index}`}><span>{test.passed ? '✓' : '×'}</span><code>{test.id}</code>{test.message && <pre>{test.message}</pre>}</li>)}</ul>}
      </div>}
    </section>
    <p className="submission-result-footnote">{t('这是本次提交的记录。继续编辑不会改变这次结果。', 'This is a saved snapshot. Further edits will not change this result.')}</p>
  </main>;
}
