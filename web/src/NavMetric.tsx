import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import "./nav-metric.css";

export default function NavMetric({ className, icon, value, label }: { className: string; icon: ReactNode; value: string; label: string }) {
  const [dismissed, setDismissed] = useState(false);
  const target = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const dismiss = (event: KeyboardEvent) => {
      if (event.key === "Escape" && target.current?.matches(":hover, :focus-within")) setDismissed(true);
    };
    document.addEventListener("keydown", dismiss);
    return () => document.removeEventListener("keydown", dismiss);
  }, []);
  return <span ref={target} className={`nav-metric ${className}`} role="img" aria-label={label} tabIndex={0}
    data-tooltip-dismissed={dismissed || undefined}
    onFocus={() => setDismissed(false)} onMouseEnter={() => setDismissed(false)}
    onBlur={() => setDismissed(false)} onMouseLeave={() => setDismissed(false)}>
    {icon}
    <b className="nav-metric-value" aria-hidden="true">{value}</b>
    <span className="nav-metric-tooltip" aria-hidden="true">{label}</span>
  </span>;
}
