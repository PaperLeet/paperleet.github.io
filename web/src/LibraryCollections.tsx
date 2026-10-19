import {DailyProblemRow} from './DailyChallenges';
import {useRef} from 'react';
import {ArrowRight} from 'lucide-react';
import AppLink from './AppLink';
import {useI18n} from './i18n';
import {status,type Paper,type Problem,type Saved} from './types';
import {paperCollections,papersInCollection} from './paperCollections';
import CollectionArtwork,{CollectionControls} from './CollectionArtwork';
export type Props={papers:Paper[];problems:Problem[];saved:Record<string,Saved>;progressReady:boolean};
/** Optional collection discovery leaves the searchable list usable while loading. */
export default function LibraryCollections({papers,problems,saved,progressReady}:Props){
 const {t}=useI18n(),collectionRail=useRef<HTMLElement>(null);
 return (
    <section className="featured-collections" aria-label={t("精选合集", "Featured collections")}>
      <div className="collections-heading">
        <h2>{t("精选合集", "Featured collections")}</h2>
        <CollectionControls rail={collectionRail}/>
      </div>
      <nav ref={collectionRail} className="collection-rail" aria-label={t("精选合集导航", "Featured collection navigation")}>
        {paperCollections.map((collection) => {
          const members = papersInCollection(papers, collection);
          const memberIds = new Set(members.flatMap((paper) => paper.module_ids));
          const count = memberIds.size;
          const completed = problems.filter(problem => memberIds.has(problem.id) && status(problem,saved[problem.id]) === "done").length;
          const label = t(collection.name, collection.name_en);
          return <AppLink key={collection.id} className={`collection-card collection-${collection.id}`} href={`#/collections/${collection.id}`} aria-description={progressReady ? `${completed} / ${count} ${t("模块已通过", "modules solved")}` : undefined} aria-label={`${label} · ${collection.kind === "course" ? t("课程", "Course") : `${members.length} ${t("篇论文", "papers")}`} · ${count} ${t("道题", "problems")}`}>
            <div className="collection-copy">
              <h3>{label}</h3>
            </div>
            <div className="collection-learning-progress" aria-hidden="true"><span>{progressReady ? `${completed}/${count} ${t("已通过", "solved")}` : t("正在加载进度…", "Loading progress…")}</span></div>
            <CollectionArtwork kind={collection.id} />
            <span className="collection-open" aria-hidden="true"><ArrowRight size={20} /></span>
          </AppLink>;
        })}
      </nav>
      <DailyProblemRow problems={problems}/>
    </section>
 );
}
