/** Decorative LeetCode-style frequency meter; the adjacent counts stay the accessible source. */
export default function InterviewFrequency({value,max}:{value:number;max:number}){
 return <span className="interview-frequency" aria-hidden="true"><i style={{width:`${Math.max(6,Math.min(100,Math.round(value/Math.max(1,max)*100)))}%`}}/></span>;
}
