import {SubmissionCompletionFallback} from './SubmissionCompletionFallback';
import {ProblemOverviewFallback} from './ProblemOverviewFallback';
import { deferredFeature, FeatureFailure, FeatureRetry as RuntimeDetailsFailure } from './deferredFeature';
import { EditorFailure, EditorLoading, type EditorRecovery } from './EditorFallback';
import { PaperDetailFailure } from './PaperDetailFallback';
import { WorkspaceBackupFailure } from './WorkspaceBackupFallback';
import { ModuleNavigationFallback } from "./ModuleNavigationFallback";
import { QueueAdmissionFallback } from './QueueAdmissionFallback';
import { useI18n } from './i18n';

// Both custom renderers supply their own visible labels.
const customRecoveryLabel:[string,string]=['',''];
const quietEnhancement={loading:()=>null,failure:()=>null};

// Like LeetCode, the bar never shows a placeholder counter (an unavailable metric is display:none),
// so the presence count simply joins the bar once its chunk and data arrive.
export const OnlinePresence = deferredFeature(() => import('./OnlinePresence'), customRecoveryLabel, quietEnhancement);

export const InboxBell = deferredFeature(() => import('./InboxBell'), ['通知', 'notifications'], {loading: () => null});
export const Messages = deferredFeature(() => import('./Messages'), ['私信', 'messages']);
// These optional controls render nothing while loading or after failure.
export const ReadingTabControls = deferredFeature(() => import('./ReadingTabControls'), customRecoveryLabel, quietEnhancement);

export const InterviewPracticeNextStep = deferredFeature(() => import('./InterviewPracticeNextStep'), ['下一道准备练习','next preparation module']);
export const PaperDetail = deferredFeature<Parameters<typeof import('./PaperDetail').default>[0]&{recovery:EditorRecovery}>(() => import('./PaperDetail'), ['论文详情','paper details'], {
  failure:props=><PaperDetailFailure returnTo={props.returnTo} recovery={props.recovery}/>,
});

export const Interviews = deferredFeature(() => import('./Interviews'), ['面经与面试准备','interview preparation']);
export const ProblemOverview = deferredFeature(() => import('./ProblemOverview'), customRecoveryLabel, {loading:props=><ProblemOverviewFallback problem={props.problem}/>,failure:(props,retry)=><ProblemOverviewFallback problem={props.problem} retry={retry}/>});
export const ProblemCompanyTags = deferredFeature(() => import('./ProblemCompanyTags'), customRecoveryLabel, quietEnhancement);
export const PreparationGuide = deferredFeature(() => import('./PreparationGuide'), customRecoveryLabel, {loading:()=>null,failure:(_props,retry)=><RuntimeDetailsFailure retry={retry} label={['重试自测提纲','Retry study prompts']}/>});
export const Community = deferredFeature(() => import('./Community'), ['社区', 'community']);
export const DraftCenter = deferredFeature(() => import('./DraftCenter'), ['创作草稿','writing drafts']);
export const ProblemSolutions = deferredFeature<Parameters<typeof import('./Community').ProblemSolutions>[0]>(() => import('./Community').then(module=>({default:module.ProblemSolutions})), ['社区题解','community solutions']);
export const ProblemDiscussions = deferredFeature<Parameters<typeof import('./Community').ProblemDiscussions>[0]>(() => import('./Community').then(module=>({default:module.ProblemDiscussions})), ['本题讨论','problem discussions']);
export const ProblemSolutionPanel = deferredFeature(() => import('./ProblemSolutionPanel'), ['本题题解','problem solutions']);
export const UserProfile = deferredFeature<Parameters<typeof import('./Community').UserProfile>[0]>(() => import('./Community').then(module => ({ default: module.UserProfile })), ['个人主页', 'profile']);
export const Account = deferredFeature(() => import('./Account'), ['账号页面', 'account page']);
export const SolutionHistory = deferredFeature(() => import('./SolutionHistory'), ['提交记录', 'submission history']);
export const SubmissionResultPage = deferredFeature(() => import('./SubmissionResultPage'), ['提交结果', 'submission result']);
export const ResizableSplit = deferredFeature(() => import('./ResizableSplit'), ['做题布局', 'workspace layout']);
export const SettingsPage = deferredFeature(() => import('./Settings'), ['设置', 'settings']);
export const Achievements = deferredFeature(() => import('./Achievements'), ['学习勋章', 'achievements']);
export const AchievementObserver = deferredFeature(() => import('./AchievementObserver'), customRecoveryLabel, quietEnhancement);
export const WorkspaceLayoutMenu = deferredFeature(() => import('./WorkspaceLayoutMenu'), customRecoveryLabel, {loading:()=>null,failure:(_props,retry)=><RuntimeDetailsFailure retry={retry} label={['重试布局选项','Retry layout options']}/>});
export const JudgeRuntimeDetails = deferredFeature(() => import('./JudgeRuntimeDetails'), customRecoveryLabel, {loading:()=>null,failure:(_props,retry)=><RuntimeDetailsFailure retry={retry} label={['重试加载运行说明','Retry run details']}/>});
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
export const PythonEditor = deferredFeature(() => import('./PythonEditor'), customRecoveryLabel, {
  loading: props => <EditorLoading {...props} />,
  failure: (props, retry) => <EditorFailure {...props} retry={retry} />,
});

function FeatureProseFailure({ retry }: { retry: () => void }) {
  return <FeatureFailure label={['题面与公式', 'statements and formulas']} retry={retry} />;
}

export const QueueAdmissionNotice = deferredFeature(() => import('./QueueAdmissionNotice'), customRecoveryLabel, {loading: props => <QueueAdmissionFallback {...props} />, failure: props => <QueueAdmissionFallback {...props} />});

export const WorkspaceTestResults = deferredFeature(() => import('./WorkspaceTestResults'), ['测试反馈','test feedback'], {failure:(_props,retry)=><RuntimeDetailsFailure retry={retry} label={['重试加载测试结果','Retry test results']}/>});

export const ReportView = deferredFeature(() => import("./ReportView"), ["测试报告", "test report"]);
export const ReportWorkflow = deferredFeature(() => import('./ReportWorkflow'), customRecoveryLabel, {loading:()=>null,failure:(_props,retry)=><RuntimeDetailsFailure retry={retry} label={['重试加载结果操作','Retry result actions']}/>});

export const ModuleNavigator = deferredFeature<Parameters<typeof import('./ModuleNavigator').ModuleNavigator>[0]>(() => import('./ModuleNavigator').then(module => ({ default: module.ModuleNavigator })), customRecoveryLabel, {
  loading: props => <ModuleNavigationFallback {...props} />,
  failure: props => <ModuleNavigationFallback {...props} />,
});

export const BrowserDraftBackups = deferredFeature<Parameters<typeof import('./BrowserDraftBackups').BrowserDraftBackups>[0]>(() => import('./BrowserDraftBackups').then(module => ({default: module.BrowserDraftBackups})), ['草稿版本', 'draft versions'], {
  loading: props => props.conflict || props.backups.some(backup => backup.code !== props.source) ? <div className="feature-loading" role="status"><span className="spinner" /></div> : null,
});

export const ProblemFavoriteButton = deferredFeature<Parameters<typeof import('./ProblemFavorites').ProblemFavoriteButton>[0]>(()=>import('./ProblemFavorites').then(module=>({default:module.ProblemFavoriteButton})),customRecoveryLabel,{loading:()=>null,failure:(_props,retry)=><RuntimeDetailsFailure retry={retry} label={['重试收藏入口','Retry bookmark']}/>});
export const ProblemFavoritesPage = deferredFeature<Parameters<typeof import('./ProblemFavorites').ProblemFavoritesPage>[0]>(()=>import('./ProblemFavorites').then(module=>({default:module.ProblemFavoritesPage})),['收藏题目','saved problems']);
export const StudyListPracticeNavigation = deferredFeature<Parameters<typeof import('./StudyLists').StudyListPracticeNavigation>[0]>(()=>import('./StudyLists').then(module=>({default:module.StudyListPracticeNavigation})),['题单导航','study-list navigation']);

export const SubmissionCompletionNotice=deferredFeature(()=>import('./SubmissionCompletionNotice'),customRecoveryLabel,{loading:()=>null,failure:props=><SubmissionCompletionFallback {...props}/>});

export const PaperRegistry = deferredFeature(() => import('./PaperRegistry'), ['论文库', 'paper registry']);
