# BUG: 把边当作有向边，只沿 u 到 v 记录邻居。
def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    names = list(nodes)
    if len(set(names)) != len(names):
        raise ValueError("duplicate node name")
    adjacency = {name: set() for name in names}
    for u, v in edges:
        if u not in adjacency or v not in adjacency:
            raise ValueError(f"edge {(u, v)!r} uses an unknown node")
        if u != v:
            adjacency[u].add(v)
    seen, components = set(), []
    for start in sorted(names):
        if start in seen:
            continue
        seen.add(start)
        stack, members = [start], []
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(members))
    return sorted(components, key=lambda component: (-len(component), component[0]))


def largest_component(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    components = connected_components(nodes, edges)
    return components[0] if components else []


def largest_component_fraction(nodes: list[str], edges: list[tuple[str, str]]) -> float:
    components = connected_components(nodes, edges)
    return len(components[0]) / len(list(nodes)) if components else 0.0
