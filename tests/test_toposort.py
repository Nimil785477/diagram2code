from diagram2code.export_program import _toposort


def test_toposort_branching_dag():
    # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]

    order = _toposort(nodes, edges)

    pos = {n: i for i, n in enumerate(order)}
    for a, b in edges:
        assert pos[a] < pos[b]
