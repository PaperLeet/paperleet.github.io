# BUG: 自环被当作普通调控边保留，使三个节点不再互异。
def find_feed_forward_loops(
    nodes: list[str], edges: list[tuple[str, str]], induced: bool = False
) -> list[tuple[str, str, str]]:
    names = list(nodes)
    if len(set(names)) != len(names):
        raise ValueError("duplicate node name")
    targets = {name: set() for name in names}
    for u, v in edges:
        if u not in targets or v not in targets:
            raise ValueError(f"edge {(u, v)!r} uses an unknown node")
        if True:
            targets[u].add(v)
    loops = []
    for x in sorted(names):
        for y in sorted(targets[x]):
            for z in sorted(targets[y] & targets[x]):
                if induced and (x in targets[y] or x in targets[z] or y in targets[z]):
                    continue
                loops.append((x, y, z))
    return loops
