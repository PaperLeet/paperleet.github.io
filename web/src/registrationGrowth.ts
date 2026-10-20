export type RegistrationDay = {date: string; new_users: number; total_users: number};
export type RegistrationGrowth = {
  total_users: number; new_users_today: number; timezone: 'UTC';
  start_date: string; end_date: string; updated_at: string; days: RegistrationDay[];
};

export function readRegistrationGrowth(value: unknown): RegistrationGrowth {
  const data = value as RegistrationGrowth | undefined;
  const count = (n: unknown) => typeof n === 'number' && Number.isSafeInteger(n) && n >= 0;
  const day = (s: unknown) => typeof s === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(s) &&
    Number.isFinite(Date.parse(s)) && new Date(s).toISOString().slice(0, 10) === s;
  if (!data || !count(data.total_users) || !count(data.new_users_today) || data.timezone !== 'UTC' ||
    !day(data.start_date) || !day(data.end_date) || !Number.isFinite(Date.parse(data.updated_at)) ||
    !Array.isArray(data.days) || !data.days.length || data.days.length > 366) throw Error('Invalid registration history');
  for (const [index, item] of data.days.entries()) {
    const previous = data.days[index - 1];
    if (!item || !day(item.date) || !count(item.new_users) || !count(item.total_users) || item.new_users > item.total_users ||
      (previous && (Date.parse(item.date) - Date.parse(previous.date) !== 86_400_000 || item.total_users !== previous.total_users + item.new_users))) throw Error('Invalid registration history');
  }
  const first = data.days[0], last = data.days.at(-1)!;
  if (first.date !== data.start_date || last.date !== data.end_date || last.total_users !== data.total_users || last.new_users !== data.new_users_today) throw Error('Invalid registration history');
  return data;
}
