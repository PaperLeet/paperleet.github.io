import {useEffect, useId, useState} from 'react';
import type {PointerEvent, KeyboardEvent} from 'react';
import {apiFetch} from './apiClient';
import {useI18n} from './i18n';
import {readRegistrationGrowth, type RegistrationGrowth} from './registrationGrowth';
import './user-growth.css';

export default function UserGrowthChart() {
  const {t, locale} = useI18n();
  const id = useId();
  const [data, setData] = useState<RegistrationGrowth>();
  const [failed, setFailed] = useState(false);
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  const [selected, setSelected] = useState<number>();
  useEffect(() => {
    let disposed = false, lastRequest = -Infinity;
    let request: AbortController | undefined;
    async function refresh() {
      if (disposed || request) return;
      lastRequest = Date.now();
      request = new AbortController();
      const controller = request;
      const timeout = window.setTimeout(() => controller.abort(), 12_000);
      setLoading(true);
      try {
        const response = await apiFetch('/api/community/registration-growth', {credentials: 'omit', cache: 'no-store', signal: controller.signal});
        if (!response.ok) throw Error('Registration history unavailable');
        const next = readRegistrationGrowth(await response.json());
        if (!disposed) {setData(next); setFailed(false); setSelected(undefined);}
      } catch {
        if (!disposed) setFailed(true);
      } finally {
        window.clearTimeout(timeout);
        request = undefined;
        if (!disposed) setLoading(false);
      }
    }
    function resume() {
      if (document.visibilityState === 'visible' && Date.now() - lastRequest >= 60_000) void refresh();
    }
    void refresh();
    const timer = window.setInterval(resume, 5 * 60_000);
    window.addEventListener('focus', resume);
    window.addEventListener('online', resume);
    document.addEventListener('visibilitychange', resume);
    return () => {
      disposed = true; request?.abort(); window.clearInterval(timer);
      window.removeEventListener('focus', resume);
      window.removeEventListener('online', resume);
      document.removeEventListener('visibilitychange', resume);
    };
  }, [retry]);

  const number = (n: number) => n.toLocaleString(locale === 'zh' ? 'zh-CN' : 'en-US');
  const days = data?.days || [];
  const index = Math.min(selected ?? days.length - 1, days.length - 1);
  const current = days[index];
  const maximum = Math.max(10, data?.total_users || 0);
  const magnitude = 10 ** Math.floor(Math.log10(maximum));
  const ceiling = Math.ceil(maximum / (magnitude / 2)) * (magnitude / 2);
  const x = (i: number) => days.length === 1 ? 140 : 34 + 212 * i / (days.length - 1);
  const y = (total: number) => 142 - 120 * total / ceiling;
  const points = days.map((day, i) => `${x(i)},${y(day.total_users)}`).join(' ');
  const shortDate = (date: string) => new Date(`${date}T00:00:00Z`).toLocaleDateString(locale === 'zh' ? 'zh-CN' : 'en-US', {month: 'short', day: 'numeric', timeZone: 'UTC'});
  const detail = current ? t(`${current.date}：累计 ${number(current.total_users)} 人，当天新增 ${number(current.new_users)} 人`, `${current.date}: ${number(current.total_users)} registered, ${number(current.new_users)} joined that day`) : '';
  function point(event: PointerEvent<HTMLDivElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    if (box.width) setSelected(Math.max(0, Math.min(days.length - 1, Math.round(((event.clientX - box.left) / box.width * 260 - 34) / 212 * (days.length - 1)))));
  }
  function keyboard(event: KeyboardEvent<HTMLDivElement>) {
    const next = ({ArrowLeft: index - 1, ArrowDown: index - 1, ArrowRight: index + 1, ArrowUp: index + 1, Home: 0, End: days.length - 1} as Record<string, number>)[event.key];
    if (next !== undefined) {event.preventDefault(); setSelected(Math.max(0, Math.min(days.length - 1, next)));}
  }
  return <section className="user-growth-card" aria-labelledby={`${id}-title`} aria-busy={loading}>
    <header><h2 id={`${id}-title`}>{t('一起成长', 'User growth')}</h2><svg className="growth-spark" viewBox="0 0 28 28" aria-hidden="true"><path d="m14 2 2.7 8.8 8.3 2.9-8.8 2.6L13 25l-2.4-8.7L2 13l8.7-2.2Z"/></svg></header>
    <div className="growth-total"><strong>{data ? number(data.total_users) : '—'}</strong><span>{t('位注册用户', 'registered users')}</span></div>
    {data && current ? <>
      <div className="growth-plot" role="slider" tabIndex={0} title={detail} aria-label={t('查看每日注册人数', 'Explore daily registrations')} aria-valuemin={0} aria-valuemax={days.length - 1} aria-valuenow={index} aria-valuetext={detail} aria-describedby={`${id}-hint`} onKeyDown={keyboard} onPointerDown={point} onPointerMove={event => {if (event.pointerType === 'mouse') point(event);}}>
        <svg viewBox="0 0 260 174" aria-hidden="true">
          <path className="growth-grid" d="M34 22H246 M34 82H246"/>
          <path className="growth-axis" d="M33 21Q35 58 33 94L34 142Q93 141 142 143T247 142 M31 23L29 145L246 145"/>
          {[0, ceiling / 2, ceiling].map(value => <text key={value} className="growth-tick" x="26" y={y(value) + 4} textAnchor="end">{number(value)}</text>)}
          <polyline className="growth-line" points={points}/>
          <path className="growth-guide" d={`M${x(index)} ${y(current.total_users)}V142`}/>
          <circle className="growth-dot-halo" cx={x(index)} cy={y(current.total_users)} r="7"/>
          <circle className="growth-dot" cx={x(index)} cy={y(current.total_users)} r="3.5"/>
          <text className="growth-tick" x={days.length === 1 ? 140 : 34} y="166" textAnchor={days.length === 1 ? 'middle' : 'start'}>{shortDate(days[0].date)}</text>
          {days.length > 1 && <text className="growth-tick" x="246" y="166" textAnchor="end">{shortDate(days.at(-1)!.date)}</text>}
        </svg>
      </div>
      <div className="growth-day" aria-live="polite"><time dateTime={current.date}>{shortDate(current.date)}</time><span>+{number(current.new_users)} {t('人加入', 'joined')}</span></div>
      <p className="growth-caption" id={`${id}-hint`} title={t('按注册时间统计，排除测试账号。移动指针、点选或用方向键查看每天的数据。', 'By registration date, excluding test accounts. Hover, tap or use arrow keys to explore each day.')}>{t('累计注册 · 每天记录 · UTC', 'Cumulative · Daily · UTC')}</p>
    </> : <div className="growth-empty" role="status">{loading ? t('正在加载增长曲线…', 'Loading our growth…') : t('暂时无法加载增长曲线', 'Growth chart is unavailable')}</div>}
    {failed && <div className="growth-recovery"><span role="status">{data ? t('暂未更新', 'Update unavailable') : ''}</span><button type="button" disabled={loading} onClick={() => setRetry(value => value + 1)}>{loading ? t('刷新中…', 'Refreshing…') : t('重试', 'Retry')}</button></div>}
  </section>;
}
