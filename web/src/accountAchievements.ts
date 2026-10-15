import { achievementDefinitions, achievementProgress, type Achievement } from './achievementProgress';
import type { Paper, Problem } from './types';

export const MAX_PINNED_ACHIEVEMENTS = 3;
export type AccountAchievement = {id:string; current:number; target:number; earned_at:string|null};
export type AccountAchievements = {
  user_id:string; visible:true; badges:AccountAchievement[]; pinned_ids:string[];
  revision:number; max_pins:3; solved:number; completed_papers:number;
};
export type PublicAchievements = {visible:false} | {
  visible:true; user_id:string; earned:{id:string;earned_at:string}[]; pinned_ids:string[];
};
const known = new Set(achievementDefinitions.map(badge => badge.id));
const integer = (value:unknown): value is number => Number.isSafeInteger(value) && Number(value) >= 0;
const date = (value:unknown): value is string => typeof value === 'string' && Number.isFinite(Date.parse(value));
const object = (value:unknown): value is Record<string,unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
function pins(value:unknown, earned:Set<string>): value is string[] {
  return Array.isArray(value) && value.length <= MAX_PINNED_ACHIEVEMENTS && new Set(value).size === value.length && value.every(id => typeof id === 'string' && known.has(id) && earned.has(id));
}
export function readAccountAchievements(value:unknown, userId:string): AccountAchievements {
  if (!object(value) || value.visible !== true || value.user_id !== userId || !Array.isArray(value.badges) ||
      !integer(value.revision) || value.max_pins !== MAX_PINNED_ACHIEVEMENTS || !integer(value.solved) || !integer(value.completed_papers)) throw new Error('Invalid account achievements response');
  const rows = value.badges;
  if (rows.length !== known.size || new Set(rows.map(row => row?.id)).size !== known.size || rows.some(row => !object(row) || !known.has(String(row.id)) || !integer(row.current) || !integer(row.target) || row.target < 1 || !(row.earned_at === null || date(row.earned_at)))) throw new Error('Invalid account achievement evidence');
  if (!pins(value.pinned_ids,new Set(rows.filter(row => row.earned_at).map(row => row.id)))) throw new Error('Invalid pinned achievements');
  return value as AccountAchievements;
}
export function readPublicAchievements(value:unknown, userId:string): PublicAchievements {
  if (object(value) && value.visible === false) return {visible:false};
  if (!object(value) || value.visible !== true || value.user_id !== userId || !Array.isArray(value.earned) || value.earned.length > known.size ||
      new Set(value.earned.map(row => row?.id)).size !== value.earned.length || value.earned.some(row => !object(row) || !known.has(String(row.id)) || !date(row.earned_at)) ||
      !pins(value.pinned_ids,new Set(value.earned.map(row => row.id)))) throw new Error('Invalid profile achievements response');
  return value as PublicAchievements;
}
/** Only server-awarded timestamps unlock account medals. Local reports supply no evidence here. */
export function accountAchievementProgress(snapshot:AccountAchievements, problems:Problem[], papers:Paper[]) {
  const presentation = achievementProgress(problems,papers,{});
  const byId = new Map(snapshot.badges.map(row => [row.id,row]));
  const badges:Achievement[] = presentation.badges.map(badge => {
    const row = byId.get(badge.id)!;
    return {...badge,current:row.current,target:row.target,unlocked:!!row.earned_at,earnedAt:row.earned_at || undefined};
  });
  const earned = badges.filter(badge => badge.unlocked);
  const next = badges.filter(badge => !badge.unlocked).sort((a,b) => b.current / b.target - a.current / a.target)[0];
  return {badges,earned,next,solved:snapshot.solved,completedPapers:snapshot.completed_papers};
}
export function movePinned(ids:string[], id:string, direction:-1|1) {
  const index = ids.indexOf(id), next = index + direction;
  if (index < 0 || next < 0 || next >= ids.length) return ids;
  const result = [...ids]; [result[index],result[next]] = [result[next],result[index]];
  return result;
}
