import type { Element as HastElement, ElementContent, Root, RootContent, Text } from 'hast';

const text = (value: string): Text => ({type:'text', value});
const element = (tagName: string, properties: HastElement['properties'], children: ElementContent[]): HastElement => ({type:'element', tagName, properties, children});
function headingText(node: RootContent): string {
  if (node.type === 'text') return node.value;
  if (node.type !== 'element') return '';
  return node.children.map(headingText).join('');
}

/** Use parsed headings, so code fences and quoted examples never become sections. */
export function readingOutline({ title, sectionLabel }: { title: string; sectionLabel: string }) {
  return (tree: Root) => {
    const headings = tree.children.flatMap(node => node.type === 'element' && node.tagName === 'h2'
      && headingText(node).trim() ? [{node, title:headingText(node).replace(/\s+/g, ' ').trim()}] : []);
    if (headings.length < 3) return;
    for (const [index, heading] of headings.entries()) {
      heading.node.properties['data-reading-anchor'] = String(index);
      heading.node.properties.tabIndex = -1;
    }
    tree.children.unshift(element('details', {className:['reading-outline']}, [
      element('summary', {}, [
        element('span', {className:['reading-outline-title']}, [text(title)]),
        element('span', {className:['reading-outline-count']}, [text(`${headings.length} ${sectionLabel}`)]),
      ]),
      element('nav', {className:['reading-outline-menu'], ariaLabel:title}, headings.map((heading, index) =>
        element('button', {type:'button', 'data-reading-section':String(index)}, [
          element('span', {ariaHidden:'true'}, [text(String(index + 1).padStart(2, '0'))]), text(heading.title),
        ]))),
    ]));
  };
}
