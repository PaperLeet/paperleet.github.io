import type { TestResult } from "./types";
export type TestOutcome = "all" | "failed" | "passed";
export function testKey(test: TestResult, index: number) { return `${test.nodeid || test.id}-${index}`; }
export function visibleReportTests(tests: TestResult[], outcome: TestOutcome, layer: string, query = "") {
  const normalize=(text:string)=>text.normalize("NFKC").toLowerCase();
  const terms=normalize(query).trim().split(/\s+/).filter(Boolean);
  return tests.map((test,index)=>({test,key:testKey(test,index)})).filter(({test})=>{
    if((outcome!=="all"&&test.passed!==(outcome==="passed"))||(layer!=="all"&&test.layer!==layer))return false;
    if(!terms.length)return true;
    const text=normalize([test.id,test.nodeid,test.layer,test.message,test.why].filter(Boolean).join("\n"));
    return terms.every(term=>text.includes(term));
  });
}
export function formatTestDuration(seconds: number) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  return safe > 0 && safe < .01 ? "<0.01s" : `${safe.toFixed(2)}s`;
}
