import { useI18n } from './i18n';
import type { Problem } from './types';

export function paperJudgeCapability(modules: Problem[]) {
  const browser = modules.filter(module => module.browser_judge).length;
  return { browser, server: modules.length - browser, total: modules.length };
}

export function ModuleJudgeBadge({ problem }: { problem: Problem }) {
  const { t } = useI18n();
  if (problem.browser_judge) return null;
  return <span className="judge-capability-badge server" title={t('在网页编辑代码，由服务端 CPU 执行测试。', 'Edit on the web; the server CPU executes this suite.')}>
    {t('服务端判题', 'Server tests')}
  </span>;
}

export function PaperJudgeBadge({ modules }: { modules: Problem[] }) {
  const { t } = useI18n();
  const { browser, server, total } = paperJudgeCapability(modules);
  if (!total || !server) return null;
  const kind = !browser ? 'server' : 'mixed';
  return <span className={`judge-capability-badge ${kind}`} title={t(`${browser} 道题可在本机 CPU 运行，${server} 道题使用服务端。`, `${browser} modules run on your CPU; ${server} use the server.`)}>
    {kind === 'server' ? t('服务端', 'Server') : t(`部分模块需服务端 ${server}/${total}`, `Server required ${server}/${total}`)}
  </span>;
}
