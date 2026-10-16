import { useI18n } from './i18n';

export type JudgeRuntimeDetailsProps = {
  server:boolean; browserResult:boolean; diagnostics?:string;
  calibration?:{reused?:boolean;reference_passed:number;mutants_caught:Record<string,unknown>|number};
  measurement?:{loading_ms:number;calibration_ms:number;learner_ms:number};
};
export default function JudgeRuntimeDetails({server,browserResult,calibration,measurement,diagnostics}:JudgeRuntimeDetailsProps) {
  const {t} = useI18n();
  return <details className="judge-runtime-details"><summary>{t('运行说明与校验记录','Run details and calibration')}</summary>
    <small>{server ? t('将提交当前代码，由服务端 CPU 执行完整测试。','Submit the current source for the complete suite on the server CPU.') : t('完整测试在本机 CPU / WASM 中运行，无需安装或 GPU。','The complete suite runs on your CPU / WASM. No installation or GPU needed.')}</small>
    {browserResult && calibration && <small>{calibration.reused ? t('该结果复用了兼容性校验','This result reused calibration') : t('兼容性校验记录','Calibration for this result')}: {calibration.reference_passed} {t('项参考测试通过','reference tests passed')} · {typeof calibration.mutants_caught === 'number' ? calibration.mutants_caught : Object.keys(calibration.mutants_caught).length} {t('个错误实现被抓住','mutants caught')}</small>}
    {browserResult && measurement && <small>{t('该结果耗时','Timing for this result')}: {t('准备','Preparation')} {(measurement.loading_ms/1000).toFixed(2)}s · {t('校验','Calibration')} {(measurement.calibration_ms/1000).toFixed(2)}s · {t('完整测试','Complete tests')} {(measurement.learner_ms/1000).toFixed(2)}s</small>}
    {diagnostics && <details><summary>{t('运行时详情','Runtime details')}</summary><code>{diagnostics}</code></details>}
    {browserResult && <p>{t('本次结果来自浏览器 · NumPy / WASM','Results from your browser · NumPy / WASM')}</p>}
  </details>;
}
