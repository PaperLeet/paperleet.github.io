import { Component, type ReactNode } from 'react';
import { reloadApp } from './appRecovery';

/** The recovery screen remains independent of account, locale, and app state. */
export class AppErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main className="startup-screen" aria-labelledby="app-recovery-title">
        <section className="startup-card" role="alert">
          <p className="startup-brand">PaperLeet</p>
          <h1 id="app-recovery-title">
            <span lang="zh-CN">页面暂时无法显示</span>
            <span className="startup-english" lang="en" style={{ display: 'block', fontSize: '.7em', fontWeight: 400 }}>This page could not be displayed</span>
          </h1>
          <p>
            <span lang="zh-CN">请重新加载页面后再试。</span>
            <span className="startup-english" lang="en" style={{ display: 'block' }}>Reload the page to try again.</span>
          </p>
          <button className="startup-reload" type="button" onClick={() => reloadApp()}>
            <span lang="zh-CN">重新加载</span> / <span lang="en">Reload</span>
          </button>
        </section>
      </main>
    );
  }
}
