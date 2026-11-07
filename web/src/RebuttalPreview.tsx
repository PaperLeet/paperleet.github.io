import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { remarkRebuttalMath } from "./rebuttalFormatting";
import "katex/dist/katex.min.css";
import "./rebuttalEditor.css";

type RenderNode = { tagName?: string; properties?: Record<string, unknown>; children?: RenderNode[] };
function openReviewMathSizes() {
  return (tree: RenderNode) => {
    const visit = (node: RenderNode, insideMath = false) => {
      const classes = node.properties?.className;
      const names = Array.isArray(classes) ? classes as string[] : [];
      const math = insideMath || names.includes("katex");
      if (math && node.properties) {
        // MathJax 3's \small is .85em; KaTeX's is .9em. Match both visual and accessible output.
        if (node.tagName === "mstyle" && node.properties.mathsize === "0.9em") node.properties.mathsize = "0.85em";
        const reset = names.find(name => /^reset-size\d+$/.test(name));
        const size = names.find(name => /^size\d+$/.test(name));
        if (reset && size) {
          const from = Number(reset.slice(10)), to = Number(size.slice(4));
          const sizes = [0, .5, .6, .7, .8, .85, 1, 1.2, 1.44, 1.728, 2.074, 2.488];
          if ((from === 5 || to === 5) && sizes[from] && sizes[to]) node.properties.style = `${node.properties.style || ""};font-size:${sizes[to] / sizes[from]}em`;
        }
      }
      node.children?.forEach(child => visit(child, math));
    };
    visit(tree);
  };
}

/** OpenReview-compatible source preview. Raw HTML stays literal, and math never trusts HTML commands. */
export function RebuttalPreview({ text }: { text: string }) {
  return <div className="rst-markdown rbe-preview"><Markdown
    remarkPlugins={[remarkGfm, remarkMath, remarkRebuttalMath]}
    rehypePlugins={[[rehypeKatex, { trust: false, strict: "ignore", maxExpand: 1000, maxSize: 3, macros: Object.create(null) }], openReviewMathSizes]}
    components={{
      a: ({ children, href }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
      img: ({ src, alt }) => <span>{`![${alt || ""}](${typeof src === "string" ? src : ""})`}</span>,
    }}
  >{text}</Markdown></div>;
}

export default RebuttalPreview;
