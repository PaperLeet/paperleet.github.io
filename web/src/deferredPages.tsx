import { deferredFeature, FeatureFailure } from './deferredFeature';
import { EditorFailure, EditorLoading, type EditorRecovery } from './EditorFallback';
import { PaperDetailFailure } from './PaperDetailFallback';
import { WorkspaceBackupFailure } from './WorkspaceBackupFallback';
import { ModuleNavigationFallback } from "./ModuleNavigationFallback";
import { QueueAdmissionFallback } from './QueueAdmissionFallback';
import { useI18n } from './i18n';
import { Globe2 } from 'lucide-react';
import NavMetric from './NavMetric';

function PresencePlaceholder({retry}: {retry?: () => void}) {
  const {t} = useI18n();
  if (retry) return <button className="nav-metric online-presence unavailable" type="button" onClick={retry}
    aria-label={t('重试在线人数', 'Retry online count')} title={t('重试在线人数', 'Retry online count')}>
    <Globe2 size={14} strokeWidth={1.8} aria-hidden="true" /><b className="nav-metric-value" aria-hidden="true">—</b>
  </button>;
  return <NavMetric className="online-presence unavailable" value="—" label={t('在线人数暂不可用', 'Online count unavailable')}
    icon={<Globe2 size={14} strokeWidth={1.8} aria-hidden="true" />} />;
}
export const OnlinePresence = deferredFeature(() => import('./OnlinePresence'), ['在线人数', 'online count'], {
  loading: () => <PresencePlaceholder />,
  failure: (_props,retry) => <PresencePlaceholder retry={retry} />,
});

export const InboxBell = deferredFeature(() => import('./InboxBell'), ['通知', 'notifications'], {loading: () => null});
export const Messages = deferredFeature(() => import('./Messages'), ['私信', 'messages']);
export const ReadingTabControls = deferredFeature(() => import('./ReadingTabControls'), ['题目选项导航','problem tab navigation'], {loading:()=>null,failure:()=>null});

export const InterviewPracticeNextStep = deferredFeature(() => import('./InterviewPracticeNextStep'), ['下一道准备练习','next preparation module']);
export const PaperDetail = deferredFeature<Parameters<typeof import('./PaperDetail').default>[0]&{recovery:EditorRecovery}>(() => import('./PaperDetail'), ['论文详情','paper details'], {
  failure:props=><PaperDetailFailure returnTo={props.returnTo} recovery={props.recovery}/>,
});

export const Interviews = deferredFeature(() => import('./Interviews'), ['面经与面试准备','interview preparation']);
export const ProblemCompanyTags = deferredFeature(() => import('./ProblemCompanyTags'), ['公司面经关联','company mentions'], {loading:()=>null,failure:()=>null});
export const PreparationGuide = deferredFeature(() => import('./PreparationGuide'), ['面试自测提纲','study prompts'], {loading:()=>null,failure:(_props,retry)=><RuntimeDetailsFailure retry={retry} label={['重试自测提纲','Retry study prompts']}/>});
export const Community = deferredFeature(() => import('./Community'), ['社区', 'community']);
export const DraftCenter = deferredFeature(() => import('./DraftCenter'), ['创作草稿','writing drafts']);
export const ProblemSolutions = deferredFeature<Parameters<typeof import('./Community').ProblemSolutions>[0]>(() => import('./Community').then(module=>({default:module.ProblemSolutions})), ['社区题解','community solutions']);
export const ProblemSolutionPanel = deferredFeature(() => import('./ProblemSolutionPanel'), ['本题题解','problem solutions']);
export const UserProfile = deferredFeature<Parameters<typeof import('./Community').UserProfile>[0]>(() => import('./Community').then(module => ({ default: module.UserProfile })), ['个人主页', 'profile']);
export const Account = deferredFeature(() => import('./Account'), ['账号页面', 'account page']);
export const SolutionHistory = deferredFeature(() => import('./SolutionHistory'), ['提交记录', 'submission history']);
export const SubmissionResultPage = deferredFeature(() => import('./SubmissionResultPage'), ['提交结果', 'submission result']);
export const ResizableSplit = deferredFeature(() => import('./ResizableSplit'), ['做题布局', 'workspace layout']);
export const SettingsPage = deferredFeature(() => import('./Settings'), ['设置', 'settings']);
export const Achievements = deferredFeature(() => import('./Achievements'), ['学习勋章', 'achievements']);
export const AchievementObserver = deferredFeature(() => import('./AchievementObserver'), ['勋章提示', 'achievement notifications'], {loading: () => null, failure: () => null});
export const WorkspaceLayoutMenu = deferredFeature(() => import('./WorkspaceLayoutMenu'), ['布局选项', 'layout options'], {loading:()=>null,failure:(_props,retry)=><RuntimeDetailsFailure retry={retry} label={['重试布局选项','Retry layout options']}/>});
export const JudgeRuntimeDetails = deferredFeature(() => import('./JudgeRuntimeDetails'), ['运行说明','run details'], {loading:()=>null,failure:(_props,retry)=><RuntimeDetailsFailure retry={retry}/>});
function RuntimeDetailsFailure({retry,label=['重试加载运行说明','Retry run details']}:{retry:()=>void;label?:[string,string]}) {
  const {t}=useI18n();
  return <button type="button" className="text-button" onClick={retry}>{t(...label)}</button>;
}
export const BrowserResultActions = deferredFeature<Parameters<typeof import('./BrowserResultActions').BrowserResultActions>[0]>(() => import('./BrowserResultActions').then(module => ({default:module.BrowserResultActions})), ['账号成绩确认', 'account verification'], {failure: (_props,retry) => <VerificationFailure retry={retry} />});
function VerificationFailure({retry}: {retry:()=>void}) {
  const {t} = useI18n();
  return <div className="feature-failure"><p role="alert">{t('账号成绩确认面板暂时无法加载，本机测试结果仍然保留。','Account verification could not load. Your local test results are preserved.')}</p><button type="button" className="button subtle" onClick={retry}>{t('重试加载','Retry loading')}</button></div>;
}
export const WorkspaceBackup = deferredFeature(() => import('./WorkspaceBackup'), ['代码备份', 'code backup'], {
  failure: props => <WorkspaceBackupFailure onClose={props.onClose} recovery={props.recovery} />,
});
export const Submissions = deferredFeature(() => import('./Submissions'), ['论文提交页面', 'paper submission page']);
export const Visuals = deferredFeature(() => import('./Lab'), ['实验室', 'lab']);
export const ArtifactCharts = deferredFeature<Parameters<typeof import('./Visuals').ArtifactCharts>[0]>(() => import('./Visuals').then(module => ({ default: module.ArtifactCharts })), ['实验图表', 'experiment charts']);
export const Prose = deferredFeature<Parameters<typeof import('./RichText').Prose>[0]>(() => import('./RichText').then(module => ({ default: module.Prose })), ['题面与公式', 'statements and formulas'], {
  failure: (props, retry) => <div><FeatureProseFailure retry={retry} /><pre className="feature-raw-prose">{props.content}</pre></div>,
});
export const PythonEditor = deferredFeature(() => import('./PythonEditor'), ['代码编辑器', 'code editor'], {
  loading: props => <EditorLoading {...props} />,
  failure: (props, retry) => <EditorFailure {...props} retry={retry} />,
});

function FeatureProseFailure({ retry }: { retry: () => void }) {
  return <FeatureFailure label={['题面与公式', 'statements and formulas']} retry={retry} />;
}

export const QueueAdmissionNotice = deferredFeature(() => import('./QueueAdmissionNotice'), ['等待状态', 'queue waiting status'], {loading: props => <QueueAdmissionFallback {...props} />, failure: props => <QueueAdmissionFallback {...props} />});

export const WorkspaceTestResults = deferredFeature(() => import('./WorkspaceTestResults'), ['测试反馈','test feedback'], {failure:(_props,retry)=><RuntimeDetailsFailure retry={retry} label={['重试加载测试结果','Retry test results']}/>});

export const ReportView = deferredFeature(() => import("./ReportView"), ["测试报告", "test report"]);
export const ReportWorkflow = deferredFeature(() => import('./ReportWorkflow'), ['修正与验证','fix and verify'], {loading:()=>null,failure:(_props,retry)=><RuntimeDetailsFailure retry={retry} label={['重试加载结果操作','Retry result actions']}/>});

export const ModuleNavigator = deferredFeature<Parameters<typeof import('./ModuleNavigator').ModuleNavigator>[0]>(() => import('./ModuleNavigator').then(module => ({ default: module.ModuleNavigator })), ['模块导航', 'module navigation'], {
  loading: props => <ModuleNavigationFallback {...props} />,
  failure: props => <ModuleNavigationFallback {...props} />,
});

function ResetCodeFallback({ retry }: { retry?: () => void }) {
  const { t } = useI18n();
  return <button className="icon-button" type="button" disabled={!retry} onClick={retry}
    aria-label={retry ? t('重试加载代码恢复', 'Retry loading code restore') : t('正在加载代码恢复', 'Loading code restore')}
    title={retry ? t('重试加载代码恢复', 'Retry loading code restore') : t('正在加载代码恢复', 'Loading code restore')}><span aria-hidden="true">↺</span></button>;
}
export const ResetCodeButton = deferredFeature<Parameters<typeof import('./ResetCodeButton').ResetCodeButton>[0]>(() => import('./ResetCodeButton').then(module => ({default: module.ResetCodeButton})), ['代码恢复', 'code restore'], {
  loading: () => <ResetCodeFallback />,
  failure: (_props, retry) => <ResetCodeFallback retry={retry} />,
});
export const BrowserDraftBackups = deferredFeature<Parameters<typeof import('./BrowserDraftBackups').BrowserDraftBackups>[0]>(() => import('./BrowserDraftBackups').then(module => ({default: module.BrowserDraftBackups})), ['草稿版本', 'draft versions'], {
  loading: props => props.conflict || props.backups.some(backup => backup.code !== props.source) ? <div className="feature-loading" role="status"><span className="spinner" /></div> : null,
});

export const ProblemFavoriteButton = deferredFeature<Parameters<typeof import('./ProblemFavorites').ProblemFavoriteButton>[0]>(()=>import('./ProblemFavorites').then(module=>({default:module.ProblemFavoriteButton})),['题目收藏','problem bookmarks'],{loading:()=>null,failure:(_props,retry)=><RuntimeDetailsFailure retry={retry} label={['重试收藏入口','Retry bookmark']}/>});
export const ProblemFavoritesPage = deferredFeature<Parameters<typeof import('./ProblemFavorites').ProblemFavoritesPage>[0]>(()=>import('./ProblemFavorites').then(module=>({default:module.ProblemFavoritesPage})),['收藏题目','saved problems']);
