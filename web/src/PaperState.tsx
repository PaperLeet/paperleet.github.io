import { CheckCircle2, Circle } from "lucide-react";
import { useI18n } from "./i18n";

export function StateIcon({ state }: { state: string }) {
  const { t } = useI18n();
  const label = state === "done" ? t("已完成", "Solved") : state === "started" ? t("进行中", "In progress") : t("未开始", "Not started");
  return <span className={`row-state ${state}`} role="img" aria-label={label} title={label}>
    {state === "done" ? <CheckCircle2 size={18} /> : state === "started" ? <span className="in-progress-dot" /> : <Circle size={17} />}
  </span>;
}

export function ProgressBar({ completed, total }: { completed: number; total: number }) {
  return <div className="paper-progress-bar" aria-hidden="true"><span style={{ width: `${total ? completed / total * 100 : 0}%` }} /></div>;
}

