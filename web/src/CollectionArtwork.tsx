import {useEffect,useState,type RefObject} from 'react';
import {ChevronLeft,ChevronRight} from 'lucide-react';
import {useI18n} from './i18n';
export function CollectionControls({rail}:{rail:RefObject<HTMLElement|null>}){
 const {t}=useI18n();
  const [railEdges, setRailEdges] = useState({ previous: false, next: false });
  useEffect(() => {
    const element = rail.current;
    if (!element) return;
    const reveal = () => {
      const card = element.querySelector('[aria-current="page"]');
      if (!card) return;
      const bounds = element.getBoundingClientRect(), selected = card.getBoundingClientRect();
      if (selected.left < bounds.left) element.scrollLeft -= bounds.left - selected.left;
      else if (selected.right > bounds.right) element.scrollLeft += selected.right - bounds.right;
    };
    reveal();
    const measure = () => setRailEdges({ previous: element.scrollLeft > 2, next: element.scrollLeft + element.clientWidth < element.scrollWidth - 2 });
    measure();
    element.addEventListener("scroll", measure, { passive: true });
    const observer = new ResizeObserver(() => {
      reveal();
      measure();
    });
    observer.observe(element);
    return () => { element.removeEventListener("scroll", measure); observer.disconnect(); };
  }, [rail]);
  function scrollCollections(direction: number) {
    const element = rail.current;
    if (element) element.scrollBy({ left: direction * element.clientWidth });
  }
 return (        <div className="collection-controls">
          <button aria-label={t("上一组合集", "Previous collections")} disabled={!railEdges.previous} onClick={() => scrollCollections(-1)}><ChevronLeft size={18} /></button>
          <button aria-label={t("下一组合集", "Next collections")} disabled={!railEdges.next} onClick={() => scrollCollections(1)}><ChevronRight size={18} /></button>
        </div>);
}
/** Decorative, text-free diagrams for the featured paper collections. */
export default function CollectionArtwork({ kind }: { kind: string }) {
  let artwork;

  switch (kind) {
    case "llm-basics":
    case "llm-frontier":
      artwork = (
        <>
          <path d="M18 130 89 165 161 126 91 92Z" fill="white" fillOpacity=".07" />
          <g transform="translate(43 27) rotate(12 53 61)">
            <rect x="-8" y="8" width="114" height="122" rx="16" fill="white" fillOpacity=".06" stroke="white" strokeOpacity=".2" />
            <rect width="114" height="122" rx="16" fill="white" fillOpacity=".12" stroke="white" strokeOpacity=".46" />
            {[0, 1, 2, 3].flatMap((row) =>
              [0, 1, 2, 3].map((column) => (
                <rect key={`${row}-${column}`} x={13 + column * 24} y={17 + row * 24} width="16" height="16" rx="4"
                  fill="white" fillOpacity={row === column ? .88 : column <= row ? .35 : .12} />
              )),
            )}
          </g>
          <path d="M22 114C45 114 35 68 57 68M22 114C53 114 41 143 79 143" fill="none" stroke="white" strokeOpacity=".65" strokeWidth="2" />
          <circle cx="22" cy="114" r="7" fill="white" fillOpacity=".85" />
          <circle cx="57" cy="68" r="3" fill="white" />
          <circle cx="79" cy="143" r="3" fill="white" />
        </>
      );
      break;
    case "retrieval":
      artwork = (
        <g fill="none" stroke="white" strokeWidth="2" strokeLinecap="round">
          <path d="M28 90h32m0 0 35-43m-35 43 35 43m0-86 46-18m-46 18 46 18m-46 68 46-18m-46 18 46 18" strokeOpacity=".55" />
          {[[28,90],[60,90],[95,47],[95,133],[141,29],[141,65],[141,115],[141,151]].map(([cx,cy],i) => (
            <circle key={i} cx={cx} cy={cy} r={i < 4 ? 9 : 6} fill="white" fillOpacity={i === 5 ? .9 : .2} strokeOpacity=".7" />
          ))}
          <path d="m135 64 4 4 8-9" stroke="#15464e" />
        </g>
      );
      break;
    case "rl":
      artwork = (
        <>
          <circle cx="122" cy="88" r="58" fill="white" fillOpacity=".04" stroke="white" strokeOpacity=".13" />
          <circle cx="122" cy="88" r="43" fill="none" stroke="white" strokeOpacity=".12" strokeDasharray="3 6" />
          <g fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round">
            <path d="M29 96C53 96 43 52 78 52M29 96C54 96 50 132 84 132" strokeOpacity=".65" />
            <path d="M78 52C112 52 105 30 142 30M78 52C110 52 110 80 142 80" strokeOpacity=".65" />
            <path d="M84 132C117 132 111 119 148 119M84 132C116 132 115 157 148 157" strokeOpacity=".3" />
            <path d="m131 26 11 4-9 7" strokeOpacity=".8" />
          </g>
          <circle cx="29" cy="96" r="12" fill="white" fillOpacity=".18" stroke="white" strokeOpacity=".6" />
          <circle cx="29" cy="96" r="4" fill="white" />
          <circle cx="78" cy="52" r="10" fill="white" fillOpacity=".85" />
          <circle cx="84" cy="132" r="9" fill="white" fillOpacity=".3" stroke="white" strokeOpacity=".6" />
          <circle cx="148" cy="30" r="7" fill="white" fillOpacity=".95" />
          <circle cx="148" cy="80" r="7" fill="white" fillOpacity=".6" />
          <circle cx="148" cy="119" r="7" fill="white" fillOpacity=".25" />
          <circle cx="148" cy="157" r="7" fill="white" fillOpacity=".25" />
        </>
      );
      break;
    case "diffusion":
      artwork = (
        <>
          <g fill="none" stroke="white" strokeLinecap="round">
            <path d="M27 145C110 162 56 50 154 34" strokeOpacity=".08" strokeWidth="27" />
            <path d="M28 145C110 162 56 50 154 34" strokeOpacity=".7" strokeWidth="2" />
            <path d="M23 126C87 148 76 63 157 55" strokeOpacity=".37" strokeWidth="1.5" />
            <path d="M42 163C112 168 74 70 163 77" strokeOpacity=".28" strokeWidth="1.5" />
          </g>
          {[[27, 51, 3], [45, 32, 2], [58, 63, 4], [23, 83, 2], [47, 96, 3], [73, 37, 2], [79, 82, 2], [20, 111, 3], [50, 122, 2], [81, 119, 3]].map(([cx, cy, r], index) => (
            <circle key={index} cx={cx} cy={cy} r={r} fill="white" fillOpacity={.2 + (index % 3) * .15} />
          ))}
          <circle cx="141" cy="61" r="28" fill="white" fillOpacity=".09" stroke="white" strokeOpacity=".25" />
          <circle cx="141" cy="61" r="17" fill="white" fillOpacity=".23" />
          <circle cx="141" cy="61" r="7" fill="white" fillOpacity=".9" />
        </>
      );
      break;
    case "post-training":
      artwork = (
        <>
          <g stroke="white" strokeLinejoin="round">
            <path d="m28 111 61-35 65 36-61 36Z" fill="white" fillOpacity=".07" strokeOpacity=".2" />
            <path d="m28 95 61-35 65 36-61 36Z" fill="white" fillOpacity=".12" strokeOpacity=".36" />
            <path d="m28 79 61-35 65 36-61 36Z" fill="white" fillOpacity=".2" strokeOpacity=".7" />
            <path d="m43 71 64 36M59 62l64 36M74 53l64 36M43 87l61-35M59 96l61-35M75 105l61-35" fill="none" strokeOpacity=".28" />
            <path d="m39 40 19-11 65 36-19 11Z" fill="white" fillOpacity=".6" strokeOpacity=".8" />
            <path d="m103 39 13-8 28 16-13 8Z" fill="white" fillOpacity=".85" strokeOpacity=".9" />
            <path d="M154 80v48l-61 36-65-36v-17" fill="none" strokeOpacity=".18" />
          </g>
          <path d="M22 56v14M15 63h14M150 38v10M145 43h10" stroke="white" strokeOpacity=".7" strokeWidth="2" strokeLinecap="round" />
        </>
      );
      break;
    case "jepa-world-models":
      artwork = (
        <>
          <rect x="18" y="49" width="57" height="83" rx="13" fill="white" fillOpacity=".09" stroke="white" strokeOpacity=".3" transform="rotate(-9 46 90)" />
          <rect x="107" y="32" width="57" height="83" rx="13" fill="white" fillOpacity=".13" stroke="white" strokeOpacity=".45" transform="rotate(8 135 73)" />
          <g fill="none" stroke="white" strokeOpacity=".4" strokeWidth="1.5">
            <path d="m30 79 25-10 6 35-29 8 23-43M30 79l31 25M123 58l24 11-10 27-22-14 32-13M123 58l14 38" />
            <path d="M70 88h37m-7-6 7 6-7 6" strokeOpacity=".85" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M46 140c24 29 66 22 83-12" strokeDasharray="3 5" strokeOpacity=".5" />
          </g>
          {[[30, 79], [55, 69], [61, 104], [32, 112], [123, 58], [147, 69], [137, 96], [115, 82]].map(([cx, cy], index) => (
            <circle key={index} cx={cx} cy={cy} r={index % 3 === 0 ? 6 : 4} fill="white" fillOpacity={index % 3 === 0 ? .9 : .45} />
          ))}
          <circle cx="89" cy="88" r="13" fill="white" fillOpacity=".08" />
        </>
      );
      break;
    case "visual-generation":
    default:
      artwork = (
        <>
          <g transform="rotate(10 101 96)" stroke="white" strokeLinejoin="round">
            <rect x="62" y="63" width="94" height="94" rx="12" fill="white" fillOpacity=".13" strokeOpacity=".65" />
            <path d="M85 63v94M109 63v94M132 63v94M62 87h94M62 110h94M62 134h94" fill="none" strokeOpacity=".23" />
            <path d="m73 139 22-30 16 16 15-26 20 40Z" fill="white" fillOpacity=".2" strokeOpacity=".35" />
            <circle cx="131" cy="83" r="6" fill="white" fillOpacity=".7" stroke="none" />
            <rect x="39" y="42" width="62" height="62" rx="9" fill="white" fillOpacity=".24" strokeOpacity=".75" />
            <path d="M70 42v62M39 73h62" fill="none" strokeOpacity=".5" />
            <rect x="20" y="24" width="35" height="35" rx="7" fill="white" fillOpacity=".68" strokeOpacity=".9" />
          </g>
          <path d="M29 120c-1 16 7 29 22 35m-5-10 5 10-12-1" fill="none" stroke="white" strokeOpacity=".55" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </>
      );
  }

  return <svg className="collection-art" viewBox="0 0 180 180" aria-hidden="true" focusable="false">{artwork}</svg>;
}
