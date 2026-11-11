import { useEffect, useMemo, useRef, useState } from "react";
import { renderOpenReviewMarkdown } from "./openReviewMarkdown";
import { clearOpenReviewMath, typesetOpenReview } from "./openReviewMathJax";
import "./rebuttalEditor.css";

function PreviewContent({ text }: { text: string }) {
  const element = useRef<HTMLDivElement>(null);
  const html = useMemo(() => ({ __html: renderOpenReviewMarkdown(text) }), [text]);
  const [status, setStatus] = useState("loading");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const target = element.current;
    if (!target) return;
    let active = true;
    setStatus("loading");
    typesetOpenReview(target, () => active).then(
      () => { if (active) setStatus("ready"); },
      () => { if (active) setStatus("error"); },
    );
    return () => { active = false; clearOpenReviewMath(target); };
  }, [retry]);
  return <>
    <div ref={element} className="rst-markdown rbe-preview" data-math-status={status} dangerouslySetInnerHTML={html} />
    {status === "error" && <p className="rbe-message" role="status">Math preview could not load. <button type="button" onClick={() => setRetry(value => value + 1)}>Retry</button></p>}
  </>;
}

/** A new DOM subtree isolates each edit from any older asynchronous typesetting. */
export function RebuttalPreview({ text }: { text: string }) {
  return <PreviewContent key={text} text={text} />;
}

export default RebuttalPreview;
