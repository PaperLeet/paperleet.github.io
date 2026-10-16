import { pythonLanguage } from '@codemirror/lang-python';
import { editorSource } from './sourceLocation';

export type PythonDefinition = { name: string; qualifiedName: string; kind: 'function' | 'class'; from: number; to: number; line: number };

/** Navigation only: read Python syntax, never execute source or infer a verdict. */
export function pythonDefinitions(value: string): PythonDefinition[] {
  const source = editorSource(value);
  const starts = [0];
  for (let index = source.indexOf('\n'); index >= 0; index = source.indexOf('\n', index + 1)) starts.push(index + 1);
  const lineAt = (position: number) => {
    let low = 0, high = starts.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (starts[middle] <= position) low = middle + 1;
      else high = middle;
    }
    return low;
  };
  const definitions: PythonDefinition[] = [];
  const scopes: { from: number; to: number; name: string }[] = [];
  pythonLanguage.parser.parse(source).iterate({
    enter(node) {
      if (node.name !== 'FunctionDefinition' && node.name !== 'ClassDefinition') return;
      const identifier = node.node.getChild('VariableName');
      if (!identifier || identifier.from >= identifier.to) return;
      const name = source.slice(identifier.from, identifier.to);
      definitions.push({ name, qualifiedName: [...scopes.map(scope => scope.name), name].join('.'),
        kind: node.name === 'ClassDefinition' ? 'class' : 'function',
        from: identifier.from, to: identifier.to, line: lineAt(identifier.from) });
      scopes.push({ from: node.from, to: node.to, name });
    },
    leave(node) {
      const last = scopes.at(-1);
      if (last?.from === node.from && last.to === node.to) scopes.pop();
    },
  });
  return definitions;
}
