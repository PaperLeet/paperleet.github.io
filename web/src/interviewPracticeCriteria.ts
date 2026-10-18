import type {InterviewFilters} from './interviewNavigation';
import type {Problem} from './types';
type Criteria=Pick<InterviewFilters,'difficulty'|'max_minutes'>;
/** Unknown metadata never qualifies for a constrained list; public source order is unchanged. */
export function interviewPracticeMatch(problem:Pick<Problem,'difficulty'|'estimated_minutes'>|undefined,criteria:Criteria):'match'|'exclude'|'unknown'{
 const difficulty=problem?.difficulty,minutes=problem?.estimated_minutes;
 const knownDifficulty=difficulty!==undefined&&['E','M','H'].includes(difficulty);
 const knownMinutes=typeof minutes==='number'&&Number.isFinite(minutes)&&minutes>0;
 if(criteria.difficulty&&knownDifficulty&&difficulty!==criteria.difficulty)return 'exclude';
 if(criteria.max_minutes&&knownMinutes&&minutes>Number(criteria.max_minutes))return 'exclude';
 if((criteria.difficulty&&!knownDifficulty)||(criteria.max_minutes&&!knownMinutes))return 'unknown';
 return 'match';
}
