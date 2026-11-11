type MathJaxRuntime = {
  startup: {
    promise: Promise<void>;
    defaultReady: () => void;
    document: { safe: { filterAttributes: Map<string, string>; filterMethods: Record<string, (safe: unknown, family: string) => string> } };
  };
  typesetPromise: (elements: HTMLElement[]) => Promise<void>;
  typesetClear: (elements: HTMLElement[]) => void;
};
declare global { interface Window { MathJax?: MathJaxRuntime | Record<string, unknown> } }

let loading: Promise<MathJaxRuntime> | undefined;
let queue = Promise.resolve();
const assetRoot = "/vendor/mathjax-3.2.2/es5";

export function loadOpenReviewMathJax(): Promise<MathJaxRuntime> {
  if (loading) return loading;
  // openreview-web/lib/mathjax-config.js, including its safe font filter.
  window.MathJax = {
    loader: { paths: { mathjax: assetRoot }, load: ["ui/safe"] },
    options: { ignoreHtmlClass: "disable-tex-rendering" },
    tex: { inlineMath: [["$", "$"], ["\\(", "\\)"]] },
    chtml: {
      scale: 1, minScale: 0.5, matchFontHeight: true, mtextInheritFont: false,
      merrorInheritFont: true, mathmlSpacing: false, skipAttributes: {}, exFactor: 0.5,
      displayAlign: "left", displayIndent: "0",
    },
    startup: {
      typeset: false,
      ready() {
        const mathjax = window.MathJax as MathJaxRuntime;
        mathjax.startup.defaultReady();
        const { safe } = mathjax.startup.document;
        safe.filterAttributes.set("fontfamily", "filterFamily");
        safe.filterMethods.filterFamily = (_safe, family) => family.split(/;/)[0];
      },
    },
  };
  loading = new Promise<MathJaxRuntime>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `${assetRoot}/tex-chtml-full.js`;
    script.async = true;
    script.dataset.rebuttalMathjax = "true";
    const fail = () => { script.remove(); loading = undefined; reject(new Error("MathJax could not load")); };
    script.onerror = fail;
    script.onload = () => {
      const mathjax = window.MathJax as MathJaxRuntime;
      if (!mathjax.startup?.promise) { fail(); return; }
      mathjax.startup.promise.then(() => resolve(mathjax), fail);
    };
    document.head.append(script);
  });
  return loading;
}

/** Serialize global MathJax work and clear removed subtrees after in-flight work. */
export function typesetOpenReview(element: HTMLElement, active: () => boolean): Promise<void> {
  const work = queue.then(async () => {
    if (!active()) return;
    const mathjax = await loadOpenReviewMathJax();
    if (!active()) return;
    try { await mathjax.typesetPromise([element]); }
    finally { if (!active()) mathjax.typesetClear([element]); }
  });
  queue = work.catch(() => {});
  return work;
}

export function clearOpenReviewMath(element: HTMLElement): void {
  queue = queue.then(() => {
    const mathjax = window.MathJax as MathJaxRuntime | undefined;
    mathjax?.typesetClear?.([element]);
  }).catch(() => {});
}
