import CustomCases from './CustomCases';
import DailyChallenges from './DailyChallenges';

type Props=Parameters<typeof CustomCases>[0]|Parameters<typeof DailyChallenges>[0];
/** Load challenge controls only after the shell is usable. */
export default function ChallengePanels(props:Props) {
  return 'problem' in props?<CustomCases {...props}/>:<DailyChallenges {...props}/>;
}
