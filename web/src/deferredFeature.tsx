import { Component, Suspense, lazy, useMemo, useState } from 'react';
import type { ComponentType, ReactNode } from 'react';
import { useI18n } from './i18n';

export type FeatureLabel = [string, string];
/** Small optional actions retry in place without reloading the workspace. */
export function FeatureRetry({retry,label=['重试加载','Retry loading']}:{retry:()=>void;label?:FeatureLabel}){
 const {t}=useI18n();return <button type="button" className="text-button" onClick={retry}>{t(...label)}</button>;
}
export function FeatureLoading({ label }: { label: FeatureLabel }) {
  const { t } = useI18n();
  return <div className="feature-loading" role="status" aria-busy="true"><span className="spinner" />{t(`正在加载${label[0]}…`, `Loading ${label[1]}…`)}</div>;
}
export function FeatureFailure({ label, retry, children }: { label: FeatureLabel; retry: () => void; children?: ReactNode }) {
  const { t } = useI18n();
  return <div className="feature-failure">
    <p role="alert">{t(`${label[0]}暂时无法加载。`, `${label[1]} could not load.`)}</p>
    <div className="feature-recovery-actions">
      <button className="button subtle" type="button" onClick={retry}>{t('重试加载', 'Retry loading')}</button>
      <button className="text-button" type="button" onClick={() => location.reload()}>{t('刷新页面', 'Reload page')}</button>
    </div>
    {children}
  </div>;
}
export class FeatureBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? this.props.fallback : this.props.children; }
}

/** A failed optional bundle must not replace the app shell or discard parent-owned data. */
export function deferredFeature<P extends object>(load: () => Promise<{ default: ComponentType<P> }>, label: FeatureLabel, options: {
  loading?: (props: P) => ReactNode;
  failure?: (props: P, retry: () => void) => ReactNode;
} = {}): ComponentType<P> {
  return function DeferredFeature(props: P) {
    const [attempt, setAttempt] = useState(0);
    // Recreate only on an explicit retry; ordinary prop changes keep forms mounted.
    const Loaded = useMemo(() => lazy(load), [attempt]);
    const retry = () => setAttempt(value => value + 1);
    return <FeatureBoundary key={attempt} fallback={options.failure ? options.failure(props, retry) : <FeatureFailure label={label} retry={retry} />}>
      <Suspense fallback={options.loading ? options.loading(props) : <FeatureLoading label={label} />}>
        <Loaded {...props} />
      </Suspense>
    </FeatureBoundary>;
  };
}
