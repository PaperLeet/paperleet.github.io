import { useI18n } from './i18n';
import type { Problem } from './types';

export function paperJudgeCapability(modules: Problem[]) {
  const browser = modules.filter(module => module.browser_judge).length;
  return { browser, server: modules.length - browser, total: modules.length };
}

export function ModuleJudgeBadge({ problem }: { problem: Problem }) {
  const { t } = useI18n();
  return <span className={`judge-capability-badge ${problem.browser_judge ? 'browser' : 'server'}`} title={problem.browser_judge
    ? t('本机 CPU 可运行完整测试；不兼容的实现可转服务端。', 'Run the complete suite on your CPU; unsupported implementations can use the server.')
    : t('在网页编辑代码，由服务端 CPU 执行测试。', 'Edit on the web; the server CPU executes this suite.')}>
    {problem.browser_judge ? t('浏览器判题 · 无需 GPU', 'Browser tests · No GPU') : t('服务端判题', 'Server tests')}
  </span>;
}

export function PaperJudgeBadge({ modules }: { modules: Problem[] }) {
  const { t } = useI18n();
  const { browser, server, total } = paperJudgeCapability(modules);
  if (!total) return null;
  const kind = !browser ? 'server' : !server ? 'browser' : 'mixed';
  return <span className={`judge-capability-badge ${kind}`} title={t(`${browser} 道题可在本机 CPU 运行，${server} 道题使用服务端。`, `${browser} modules run on your CPU; ${server} use the server.`)}>
    {kind === 'browser' ? t('浏览器', 'Browser') : kind === 'server' ? t('服务端', 'Server') : t(`部分浏览器 ${browser}/${total}`, `Browser ${browser}/${total}`)}
  </span>;
}
