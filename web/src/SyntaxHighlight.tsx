import { useEffect, useState } from "react";

export function Highlight({
  code,
  lang = "python",
}: {
  code: string;
  lang?: string;
}) {
  const [rendered, setRendered] = useState<{ code: string; lang: string; html: string }>();
  const html = rendered?.code === code && rendered.lang === lang ? rendered.html : '';
  useEffect(() => {
    let active = true;
    import("./highlight")
      .then(({ highlight }) => highlight(code, lang))
      .then((h) => {
        if (active) setRendered({ code, lang, html: h });
      })
      .catch(() => {
        if (active) setRendered(undefined);
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
