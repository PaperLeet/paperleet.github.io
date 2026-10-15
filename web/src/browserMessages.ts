export function browserReason(reason: string, en = false) {
  const messages: Record<string, [string, string]> = {
    python_api_unsupported: ['当前实现使用了浏览器兼容层未支持的 Python / PyTorch API', 'This implementation uses Python / PyTorch APIs outside the browser adapter'],
    calibration_failed: ['浏览器运行时未通过参考实现及错误实现校验', 'The browser runtime failed reference and mutation calibration'],
    collection_failed: ['浏览器未能收集完整的原始测试', 'The browser could not collect the complete maintained suite'],
    runtime_failed: ['浏览器 Python 运行时加载或执行失败', 'The browser Python runtime could not load or execute'],
    timeout: ['浏览器运行超时', 'The browser run timed out'],
    unavailable: ['此环境不支持隔离的浏览器判题', 'Isolated browser judging is unavailable in this environment'],
  };
  return (messages[reason] || messages.unavailable)[en ? 1 : 0];
}
