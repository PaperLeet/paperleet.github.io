import { deferredFeature, FeatureFailure } from './deferredFeature';
import { EditorFailure, EditorLoading } from './EditorFallback';
import { WorkspaceBackupFailure } from './WorkspaceBackupFallback';

export const Community = deferredFeature(() => import('./Community'), ['社区', 'community']);
export const UserProfile = deferredFeature<Parameters<typeof import('./Community').UserProfile>[0]>(() => import('./Community').then(module => ({ default: module.UserProfile })), ['个人主页', 'profile']);
export const Account = deferredFeature(() => import('./Account'), ['账号页面', 'account page']);
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
