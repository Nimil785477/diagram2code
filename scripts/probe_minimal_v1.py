from diagram2code.datasets import DatasetRegistry, load_dataset

root = DatasetRegistry().resolve_root("example:minimal_v1")
ds = load_dataset(root)

print("dataset_root =", root)
print("splits =", ds.splits())

split = ds.splits()[0]
print("using split =", split)

if hasattr(ds, "iter_samples"):
    it = ds.iter_samples(split=split)
elif hasattr(ds, "get_split"):
    it = ds.get_split(split)
else:
    raise RuntimeError("Unknown Dataset iteration API")

s = next(iter(it))
print("sample_type =", type(s))

metadata = getattr(s, "metadata", None)
print("metadata_type =", type(metadata))
print("metadata_keys =", list(metadata.keys()) if isinstance(metadata, dict) else metadata)

g = None
if isinstance(metadata, dict):
    for k in ("gt_graph", "graph", "ground_truth"):
        if k in metadata:
            g = metadata[k]
            print("using metadata key =", k)
            break
    if g is None:
        g = metadata
else:
    g = metadata

print("gt_graph_type =", type(g))

nodes = getattr(g, "nodes", None)
print("nodes_type =", type(nodes))
print("num_nodes =", len(nodes) if nodes else None)

if nodes:
    n = nodes[0]
    print("first_node_type =", type(n))
    print("has_id =", hasattr(n, "id"))
    print("has_bbox =", hasattr(n, "bbox"))
    print("node_repr =", n)
