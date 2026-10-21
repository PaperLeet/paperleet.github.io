import AppLink from './AppLink';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent } from "react";
import { ArrowRight, Check, CheckCheck, Flame, LockKeyhole, RefreshCw } from "lucide-react";
import { useI18n } from "./i18n";
import { DateLabel, SocialNotice, errorMessage, socialApi, useAccount } from "./social";
import { activityLevel, browserTimezone, calendarCells, currentYear, emptyPracticeDay } from "./profile-progress";
import type { PracticeDay, ProfilePractice, ProfilePracticeResponse } from "./profile-progress";
import { usePracticeInvalidation } from './usePracticeInvalidation';
import "./profile-progress.css";
import "./lc-profile.css";

function difficultyLabel(difficulty: string, t: (zh: string, en: string) => string) {
  return ({ E: t("简单", "Easy"), M: t("中等", "Medium"), H: t("困难", "Hard") })[difficulty] || difficulty;
}

function PracticeCalendar({ data, year, setYear }: {
  data: ProfilePractice;
  year: number;
  setYear: (year: number) => void;
}) {
  const { locale, t } = useI18n();
  const calendar = data.calendar;
  // Keep the full year in view so a year change does not move every month.
  const cells = useMemo(() => calendarCells(`${year}-01-01`, `${year}-12-31`, calendar.days), [year, calendar.days]);
  const [selectedDate, setSelectedDate] = useState(calendar.end);
  const [hoverDate, setHoverDate] = useState<string>();
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const scroller = useRef<HTMLDivElement>(null);
  // Narrow screens open on the latest activity instead of January.
  useLayoutEffect(() => {
    const element = scroller.current, cell = buttons.current.get(calendar.end);
    if (element && cell && element.scrollWidth > element.clientWidth) element.scrollLeft = Math.max(0, cell.offsetLeft + cell.offsetWidth - element.clientWidth + 24);
  }, [calendar.end]);
  const selected = cells.find((cell) => cell.date === (hoverDate || selectedDate))?.day || emptyPracticeDay(calendar.end);
  const columns = cells.length / 7;
  const dateFormatter = useMemo(() => new Intl.DateTimeFormat(locale === "zh" ? "zh-CN" : "en", {
    month: "short", day: "numeric", year: "numeric", timeZone: "UTC",
  }), [locale]);
  const monthFormatter = useMemo(() => new Intl.DateTimeFormat(locale === "zh" ? "zh-CN" : "en", {
    month: "short", timeZone: "UTC",
  }), [locale]);
  const dateLabel = (date: string) => dateFormatter.format(new Date(`${date}T12:00:00Z`));
  const dayLabel = (day: PracticeDay) => t(
    `${dateLabel(day.date)}：提交 ${day.submissions} 次，尝试 ${day.attempted} 题，通过 ${day.solved} 题，新完成 ${day.newly_solved} 题`,
    `${dateLabel(day.date)}: ${day.submissions} submissions, ${day.attempted} attempted, ${day.solved} solved, ${day.newly_solved} newly solved`,
  );
  function navigate(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const offset = ({ ArrowUp: -1, ArrowDown: 1, ArrowLeft: -7, ArrowRight: 7 })[event.key];
    let next = offset === undefined ? index : index + offset;
    if (event.key === "Home") next = cells.findIndex((cell) => cell.date === calendar.start);
    else if (event.key === "End") next = cells.findIndex((cell) => cell.date === calendar.end);
    else if (offset === undefined) return;
    event.preventDefault();
    const target = cells[next];
    if (target && target.date >= calendar.start && target.date <= calendar.end) {
      setSelectedDate(target.date);
      buttons.current.get(target.date)?.focus();
    }
  }
  return (
    <section className="social-panel practice-calendar" aria-labelledby="practice-calendar-title">
      <div className="practice-calendar-heading">
        <h2 id="practice-calendar-title"><strong>{calendar.submissions.toLocaleString()}</strong> {t("次提交", "submissions")} <span>{t(`在 ${year} 年`, `in ${year}`)}</span></h2>
        <label className="practice-year"><span className="profile-sr-only">{t("选择年份", "Select year")}</span>
          <select value={year} onChange={(event) => setYear(Number(event.target.value))}>
            {data.years.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
      </div>
      <div className="practice-calendar-summary">
        <span>{t("活跃天数", "Active days")} <b>{calendar.active_days}</b></span>
        <span>{t("最长连续", "Longest streak")} <b>{calendar.max_streak}</b> {t("天", "days")}</span>
        {year === currentYear(data.timezone) && <span className="practice-current-streak"><Flame size={14} /> {t("当前连续", "Current streak")} <b>{calendar.current_streak}</b> {t("天", "days")}</span>}
      </div>
      <div ref={scroller} className="practice-calendar-scroll" role="group" aria-label={t("每日提交记录，可使用方向键选择日期", "Daily submissions; use arrow keys to select a date")}>
        <div className="practice-calendar-chart" style={{ "--calendar-columns": columns } as CSSProperties}>
          <div className="practice-weekdays" aria-hidden="true"><span>{t("一", "Mon")}</span><span>{t("三", "Wed")}</span><span>{t("五", "Fri")}</span></div>
          <div className="practice-calendar-months" aria-hidden="true">
            {cells.filter((cell) => cell.date.endsWith("-01") && cell.inRange).map((cell) => {
              const index = cells.indexOf(cell);
              return <span key={cell.date} style={{ gridColumn: Math.floor(index / 7) + 1 }}>{monthFormatter.format(new Date(`${cell.date}T12:00:00Z`))}</span>;
            })}
          </div>
          <div className="practice-calendar-grid" onMouseLeave={() => setHoverDate(undefined)}>
            {cells.map((cell, index) => cell.inRange && cell.date <= calendar.end ? (
              <button key={cell.date} type="button"
                ref={(node) => { if (node) buttons.current.set(cell.date, node); else buttons.current.delete(cell.date); }}
                className={`practice-day level-${activityLevel(cell.day.submissions)}${selectedDate === cell.date ? " selected" : ""}`}
                tabIndex={selectedDate === cell.date ? 0 : -1}
                aria-label={dayLabel(cell.day)} aria-pressed={selectedDate === cell.date} title={dayLabel(cell.day)}
                onClick={() => setSelectedDate(cell.date)} onFocus={() => { setHoverDate(undefined); setSelectedDate(cell.date); }}
                onMouseEnter={() => setHoverDate(cell.date)} onKeyDown={(event) => navigate(event, index)}
              />
            ) : <span key={cell.date} className={`practice-day ${cell.inRange ? "future" : "outside"}`} aria-hidden="true" />)}
          </div>
        </div>
      </div>
      <div className="practice-calendar-footer">
        <span className="practice-timezone">{t("时区", "Timezone")}: {data.timezone.replaceAll("_", " ")}</span>
        <div className="practice-legend" aria-label={t("颜色越深，提交越多", "Darker colors indicate more submissions")}><span>{t("少", "Less")}</span>{[0, 1, 2, 3, 4].map((level) => <i key={level} className={`practice-day level-${level}`} />)}<span>{t("多", "More")}</span></div>
      </div>
      <div className="practice-day-detail" aria-live="polite" aria-atomic="true">
        <strong>{dateLabel(selected.date)}</strong>
        <span>{t("提交", "Submissions")} <b>{selected.submissions}</b></span>
        <span>{t("尝试题数", "Attempted")} <b>{selected.attempted}</b></span>
        <span>{t("通过题数", "Solved")} <b>{selected.solved}</b></span>
        <span>{t("首次通过", "Newly solved")} <b>{selected.newly_solved}</b></span>
      </div>
    </section>
  );
}

function ProgressSummary({ data }: { data: ProfilePractice }) {
  const { t } = useI18n();
  // LeetCode-style gauge: an arc open at the bottom, split by each difficulty's share of the
  // catalogue; each segment shows its own track and the solved part in the difficulty colour.
  const circumference = 2 * Math.PI * 73, gap = 4;
  const groups = data.by_difficulty.filter((entry) => entry.total > 0);
  const total = groups.reduce((sum, entry) => sum + entry.total, 0);
  const usable = circumference * 300 / 360 - gap * Math.max(0, groups.length - 1);
  let cursor = 0;
  const arcs = groups.map((entry) => {
    const length = total ? usable * entry.total / total : 0, start = cursor;
    cursor += length + gap;
    return { key: entry.difficulty, start, length, solved: length * Math.min(1, entry.solved / entry.total) };
  });
  const arc = (className: string, start: number, length: number, key: string) => length > 0 && <circle key={key} className={className} cx="85" cy="85" r="73" strokeDasharray={`${length} ${circumference}`} strokeDashoffset={-start} />;
  return (
    <div className="practice-overview">
      <section className="social-panel practice-solved" aria-label={t("解题进度", "Problem progress")}>
        <div className="practice-solved-ring">
          <svg viewBox="0 0 170 170" aria-hidden="true">{!arcs.length && arc("practice-ring-track", 0, usable, "empty")}{arcs.map((item) => <g key={item.key} className={`difficulty-${item.key}`}>{arc("practice-ring-track", item.start, item.length, "track")}{arc("practice-ring-progress", item.start, item.solved, "solved")}</g>)}</svg>
          <div><p><strong>{data.solved}</strong><span>/{data.total_problems}</span></p><span className="practice-solved-label"><Check size={17} />{t("已完成", "Solved")}</span><small>{data.attempting} {t("题尝试中", "attempting")}</small></div>
        </div>
        <div className="practice-difficulties">
          {data.by_difficulty.map((entry) => <div key={entry.difficulty} className={`practice-difficulty difficulty-${entry.difficulty}`}><span>{difficultyLabel(entry.difficulty, t)}</span><p><strong>{entry.solved}</strong><span> / {entry.total}</span></p></div>)}
        </div>
      </section>
      <section className="social-panel practice-totals" aria-labelledby="practice-totals-title">
        <h2 id="practice-totals-title">{t("累计练习", "Practice totals")}</h2>
        <div className="practice-total-main"><strong>{data.attempted}</strong><span>{t("题已尝试", "problems attempted")}</span></div>
        <dl>
          <div><dt>{t("总提交", "Submissions")}</dt><dd>{data.total_submissions.toLocaleString()}</dd></div>
          <div><dt><i className="accepted" />{t("成功", "Accepted")}</dt><dd>{data.accepted_submissions.toLocaleString()}</dd></div>
          <div><dt><i className="failed" />{t("失败", "Not accepted")}</dt><dd>{data.failed_submissions.toLocaleString()}</dd></div>
        </dl>
      </section>
    </div>
  );
}

export default function ProfileProgress({ username, isSelf }: { username: string; isSelf: boolean }) {
  const { locale, t } = useI18n();
  const { health, user } = useAccount();
  const [timezone] = useState(browserTimezone);
  const [year, setYear] = useState(() => currentYear(timezone));
  const [revision, setRevision] = useState(0);
  const [result, setResult] = useState<{ scope: string; data: ProfilePracticeResponse }>();
  const [failure, setFailure] = useState<{ request: string; message: string }>();
  usePracticeInvalidation(() => setRevision(value => value + 1), isSelf && user ? `user:${user.id}` : undefined);
  const scope = `${user?.id || 'anonymous'}:${username}:${year}:${timezone}`;
  const request = `${scope}:${revision}`;
  const data = result?.scope === scope ? result.data : undefined;
  const error = failure?.request === request ? failure.message : undefined;
  useEffect(() => {
    let live = true;
    const params = new URLSearchParams({ year: String(year), timezone });
    socialApi<ProfilePracticeResponse>(`community/profiles/${encodeURIComponent(username)}/practice?${params}`)
      .then((value) => { if (live) setResult({ scope, data: value }); })
      .catch((value) => { if (live) setFailure({ request, message: errorMessage(value) }); });
    return () => { live = false; };
  }, [request, scope, username, year, timezone]);
  if (error) return <section className="social-panel practice-load-state"><SocialNotice error={error} /><button className="button subtle" onClick={() => setRevision((value) => value + 1)}><RefreshCw size={14} />{t("重试练习记录", "Retry practice history")}</button></section>;
  // Placeholders keep the final card sizes so the activity below does not jump when data arrives.
  if (!data) return <div className="profile-progress practice-skeletons" aria-busy="true">
    <span className="profile-sr-only" role="status">{t("正在加载练习记录…", "Loading practice history…")}</span>
    <div className="practice-overview" aria-hidden="true"><div className="social-panel practice-skeleton" /><div className="social-panel practice-skeleton" /></div>
    <div className="social-panel practice-skeleton practice-skeleton-calendar" aria-hidden="true" />
    <div className="social-panel practice-skeleton practice-skeleton-recent" aria-hidden="true" />
  </div>;
  if (!data.visible) return <section className="social-panel practice-private"><LockKeyhole size={19} /><span>{t("该用户的练习记录仅自己可见", "This member’s practice history is private")}</span></section>;
  return (
    <div className="profile-progress">
      {health?.mode === "local" && <p className="practice-storage-note">{t("这里统计服务器保存的判题记录。本地开发模式中的练习记录不会计入。", "These totals use server-saved judge results. Practice in local development mode is not included.")}</p>}
      <ProgressSummary data={data} />
      <PracticeCalendar key={`${data.year}:${data.timezone}`} data={data} year={year} setYear={setYear} />
      <section className="social-panel practice-recent" aria-labelledby="practice-recent-title">
        <div className="practice-section-heading"><h2 id="practice-recent-title"><CheckCheck size={19} />{t("最近通过", "Recent accepted")}</h2>{isSelf && <AppLink href="#/">{t("继续做题", "Explore problems")}<ArrowRight size={14} /></AppLink>}</div>
        {data.recent_accepted.length ? <ol>{data.recent_accepted.map((problem) => <li key={problem.problem_id}><AppLink href={`#/${problem.number}`}><span className="practice-ac-check"><Check size={15} /></span><span className="practice-recent-title"><span className="practice-problem-number">{problem.number}.</span> {locale === "en" ? problem.title_en || problem.title : problem.title}</span><span className={`practice-recent-difficulty difficulty-${problem.difficulty}`}>{difficultyLabel(problem.difficulty, t)}</span><DateLabel value={problem.accepted_at} /><ArrowRight className="practice-recent-arrow" size={14} /></AppLink></li>)}</ol> : <div className="practice-recent-empty"><CheckCheck size={25} /><p>{t("还没有通过的题目", "No accepted problems yet")}</p>{isSelf && <AppLink href="#/">{t("去做第一道题", "Solve your first problem")}<ArrowRight size={14} /></AppLink>}</div>}
      </section>
    </div>
  );
}
