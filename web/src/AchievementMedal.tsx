import { BookOpen, Code2, Compass, Cpu, Crown, Gem, Layers3, Library, Mountain, Orbit, Scan, Sparkles, Stars, Target, WandSparkles } from 'lucide-react';
import type { Achievement } from './achievementProgress';

const icons = {spark:Sparkles,layers:Layers3,code:Code2,gem:Gem,crown:Crown,book:BookOpen,library:Library,compass:Compass,mountain:Mountain,cpu:Cpu,target:Target,orbit:Orbit,scan:Scan,wand:WandSparkles,stars:Stars};
export function AchievementMedal({badge,large = false}: {badge:Pick<Achievement,'icon'|'tier'|'unlocked'>;large?:boolean}) {
  const Icon = icons[badge.icon];
  return <span className={`achievement-medal${large ? ' large' : ''}`} data-tier={badge.tier} data-earned={badge.unlocked} aria-hidden="true">
    <svg className="medal-outline" viewBox="0 0 100 112" fill="none"><path className="medal-body" d="M50 4 90 27v46L50 99 10 73V27Z"/><path className="medal-inset" d="m50 12 33 19v38L50 90 17 69V31Z"/><path className="medal-ribbon" d="m29 87-4 20 25-10 25 10-4-20"/><path className="medal-facet" d="m17 31 33 19 33-19M50 50v40"/></svg>
    <span className="medal-symbol"><Icon size={large ? 34 : 28} strokeWidth={1.6} /></span>
    <span className="medal-glint" />
  </span>;
}
