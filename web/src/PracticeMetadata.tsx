import {Clock3} from 'lucide-react';
import {useI18n} from './i18n';
import './practice-metadata.css';

/** Describe published exercise metadata, independently of grades or interview frequency. */
export default function PracticeMetadata({difficulty,estimatedMinutes,className}:{difficulty?:string;estimatedMinutes?:number;className?:string}){
 const {t}=useI18n();
 const level=difficulty==='E'?t('简单','Easy'):difficulty==='M'?t('中等','Medium'):difficulty==='H'?t('困难','Hard'):undefined;
 const minutes=typeof estimatedMinutes==='number'&&Number.isFinite(estimatedMinutes)&&estimatedMinutes>0?estimatedMinutes:undefined;
 if(!level&&minutes===undefined)return null;
 return <span className={['practice-metadata',className].filter(Boolean).join(' ')}>
  {level&&<span className="practice-metadata-difficulty" data-difficulty={difficulty}><span className="sr-only">{t('难度：','Difficulty: ')}</span>{level}</span>}
  {minutes!==undefined&&<span className="practice-metadata-time" title={t('题目给出的预计练习用时，实际用时因人而异。','Estimated practice time from the exercise; actual time varies.')}><Clock3 size={14} aria-hidden="true"/><span>{t(`预计 ${minutes} 分钟`,`Est. ${minutes} min`)}</span></span>}
 </span>;
}
