import {useI18n} from './i18n';
import {browserProgressText,type BrowserProgress} from './browserProgress';
import './browser-progress-details.css';

export default function BrowserRunProgressDetails({progress}:{progress:BrowserProgress}){
  const {t,locale}=useI18n();
  const phases=['loading','calibrating','running'];
  const titles=[t('准备环境','Prepare runtime'),t('校验判题','Validate runtime'),t('执行测试','Run tests')];
  const active=phases.indexOf(progress.stage);
  const hasCount=progress.completed!==undefined&&progress.total!==undefined;
  return <>
    <ol aria-label={t('判题步骤', 'Judging steps')}>
      {phases.map((stage, index) => <li key={stage} data-state={index < active ? 'done' : index === active ? 'current' : 'next'} aria-current={index === active ? 'step' : undefined}>
        <span aria-hidden="true">{index < active ? '✓' : index + 1}</span>{titles[index]}
      </li>)}
    </ol>
    <div role="status" aria-live="polite" aria-atomic="true">
      <p>{browserProgressText(progress, locale === 'en')}</p>
      <progress aria-label={t('当前阶段进度', 'Current stage progress')} value={hasCount ? progress.completed : undefined} max={hasCount ? progress.total : undefined} />
      {progress.activity && <div className="browser-run-activity">
        <p>{t('当前测试', 'Current test')} · {progress.activity.layer} · <code>{progress.activity.test}</code></p>
        {progress.activity.mutant && <p>{t('错误实现', 'Mutation case')} · <code>{progress.activity.mutant}</code></p>}
        <p>{progress.activity.why}</p>
      </div>}
    </div>
    <p className="browser-run-progress-hint">{progress.stage === 'loading'
      ? t('首次运行需要下载环境，之后可复用缓存。你可以继续编辑或停止运行。', 'The first run downloads the runtime; later runs can reuse cached assets. You can keep editing or stop the run.')
      : progress.stage === 'calibrating'
        ? t('先确认参考解通过、错误实现会失败，再执行你的代码。', 'The reference must pass and mutation cases must fail before your code runs.')
        : t('这里统计已完成的测试，通过情况将在运行结束后显示。', 'This counts finished tests. Pass/fail results appear when the run ends.')}</p>
  </>;
}
