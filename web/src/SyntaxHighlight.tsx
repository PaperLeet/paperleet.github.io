import { useEffect, useState } from "react";

export function Highlight({
  code,
  lang = "python",
}: {
  code: string;
  lang?: string;
}) {
  const [html, setHtml] = useState("");
  useEffect(() => {
    let active = true;
    import("./highlight")
      .then(({ highlight }) => highlight(code, lang))
      .then((h) => {
        if (active) setHtml(h);
      })
      .catch(() => {
        if (active) setHtml("");
      });
    return () => {
      active = false;
    };
  }, [code, lang]);
  return html ? (
    <div className="highlight" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <div className="highlight">
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  );
}
