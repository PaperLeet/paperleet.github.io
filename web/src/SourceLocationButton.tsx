import { useI18n } from './i18n';
import type { SourceLocation } from './sourceLocation';

export function SourceLocationButton({ location, onLocate }: { location?: SourceLocation; onLocate?: (location: SourceLocation) => void }) {
  const { t } = useI18n();
  if (!location || !onLocate) return null;
  return <button className="button subtle source-location-button" type="button" onClick={() => onLocate(location)}>
    <span aria-hidden="true">↗</span>
    {t(`定位代码第 ${location.line} 行`, `Go to code line ${location.line}`)}
  </button>;
}
