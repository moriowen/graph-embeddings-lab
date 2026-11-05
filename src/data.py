"""Fetch SNAP data and build the graphs. python -m src.data

Rules (PLAN section 3): symmetrise, drop self-links, URL-decode titles, keep the largest component, number
nodes from 1 (vendored DeepWalk's `if start:` skips node 0). Output: data/<graph>/{nodes.tsv,edges.tsv,labels.json}
and data/provenance.json (URLs, sha256, counts).
"""
import collections
import gzip
import hashlib
import json
import random
import tarfile
import urllib.parse as U
import urllib.request

import networkx as nx

from config import DATA, GRAPHS, QUERIES, QUERY_RULE_SEED, RAW, SOURCES

PHYS = "subject.Science.Physics"


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in SOURCES.items():
        p = RAW / url.rsplit("/", 1)[1]
        if not p.exists():
            urllib.request.urlretrieve(url, p)
        out[name] = {"url": url, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        if p.name.endswith(".tar.gz") and not (RAW / p.name.replace(".tar.gz", "")).exists():
            with tarfile.open(p) as t:
                t.extractall(RAW, filter="data")
    return out


def rows(path):
    with open(path, encoding="utf8") as f:
        return [l.rstrip("\n").split("\t") for l in f if l.strip() and not l.startswith("#")]


def wikispeedia():
    d = RAW / "wikispeedia_paths-and-graph"
    arts = [U.unquote(r[0]) for r in rows(d / "articles.tsv")]
    cats = collections.defaultdict(set)
    for a, c in rows(d / "categories.tsv"):
        cats[U.unquote(a)].add(c)
    G = nx.Graph()
    G.add_nodes_from(arts)
    n_raw = n_self = 0
    for a, b in rows(d / "links.tsv"):
        a, b = U.unquote(a), U.unquote(b)
        n_raw += 1
        if a == b:
            n_self += 1
        else:
            G.add_edge(a, b)
    info = dict(articles=len(arts), raw_links=n_raw, self_links=n_self, undirected_edges=G.number_of_edges(),
                nodes_with_links=sum(d > 0 for _, d in G.degree()))
    return G, cats, info


def largest(G):
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def save(name, G, labels, info):
    lo, hi = GRAPHS[name]
    assert lo <= G.number_of_nodes() <= hi, (name, G.number_of_nodes())
    assert nx.is_connected(G)
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    titles = sorted(G.nodes, key=str)
    ids = {t: i + 1 for i, t in enumerate(titles)}
    (out / "nodes.tsv").write_text("".join(f"{ids[t]}\t{t}\n" for t in titles), encoding="utf8")
    edges = sorted(tuple(sorted((ids[a], ids[b]))) for a, b in G.edges)
    (out / "edges.tsv").write_text("".join(f"{a}\t{b}\n" for a, b in edges))
    (out / "labels.json").write_text(json.dumps({str(ids[t]): labels[t] for t in titles if t in labels},
                                                 indent=0, sort_keys=True))
    return dict(info, nodes=G.number_of_nodes(), edges=G.number_of_edges(), labelled=sum(t in labels for t in titles),
                max_degree=max(d for _, d in G.degree()), sum_deg2=sum(d * d for _, d in G.degree()))


def query_rule(S):
    o = sorted(S.nodes, key=lambda a: (S.degree(a), a))
    n, rng = len(o), random.Random(QUERY_RULE_SEED)
    return [rng.choice(o[i * n // 5:(i + 1) * n // 5]) for i in range(5)]


def build_wiki():
    G, cats, info = wikispeedia()
    def top(a): return sorted({c.split(".")[1] for c in cats[a]})
    def sub(a): return sorted({c.split(".")[3] for c in cats[a] if c.startswith(PHYS + ".")})
    phys = [a for a in G if any(c.startswith(PHYS) for c in cats[a])]
    S = G.subgraph(phys).copy()
    assert sorted(query_rule(S)) == sorted(QUERIES), query_rule(S)
    L = largest(G)
    # "labels" is what the graph is scored on; "phys" lets S5 re-score physics nodes on the large graph
    # with the small graph's subcategories, since the two label sets are not comparable.
    small = {a: {"labels": sub(a), "phys": sub(a)} for a in S}
    big = {a: {"labels": top(a), "phys": sub(a)} for a in L if cats[a]}
    dropped = sorted(set(G) - set(L))
    a = save("small", S, small, {"source": "wikispeedia", "rule": "induced on " + PHYS, "label_scheme": "physics subcategory"})
    b = save("large", L, big, {**info, "source": "wikispeedia", "rule": "largest component",
                               "label_scheme": "top-level subject", "dropped_nodes": dropped,
                               "no_category_kept": sorted(t for t in L if not cats[t])})
    return a, b, {t: dict(small_degree=S.degree(t), large_degree=L.degree(t)) for t in QUERIES}


def build_facebook():
    d = RAW / "facebook"
    S = nx.Graph()
    S.add_nodes_from(x.split()[0] for x in (d / "0.feat").read_text().splitlines())
    for l in (d / "0.edges").read_text().splitlines():
        a, b = l.split()
        S.add_edge(a, b)
    n_all = S.number_of_nodes()
    S = largest(S)
    circ = collections.defaultdict(list)
    for r in (d / "0.circles").read_text().splitlines():
        p = r.split("\t")
        for m in p[1:]:
            circ[m].append(p[0])
    a = save("fb_small", S, {t: {"labels": sorted(circ[t])} for t in S if t in circ},
             {"source": "ego-Facebook 0", "ego_node": "omitted, as in SNAP's 0.edges", "nodes_before_lcc": n_all,
              "rule": "largest component", "label_scheme": "circle"})
    L = nx.Graph()
    with gzip.open(RAW / "facebook_combined.txt.gz", "rt") as f:
        for l in f:
            u, v = l.split()
            L.add_edge(u, v)
    circ = collections.defaultdict(set)
    for p in d.glob("*.circles"):
        ego = p.stem
        for r in p.read_text().splitlines():
            q = r.split("\t")
            for m in q[1:]:
                circ[m].add(f"{ego}/{q[0]}")
    b = save("fb_large", L, {t: {"labels": sorted(circ[t])} for t in L if t in circ},
             {"source": "ego-Facebook combined", "rule": "as distributed", "label_scheme": "circle (ego/circle)"})
    return a, b


def main():
    src = fetch()
    s, l, q = build_wiki()
    fs, fl = build_facebook()
    prov = dict(sources=src, small=s, large=l, fb_small=fs, fb_large=fl, query_degrees=q)
    (DATA / "provenance.json").write_text(json.dumps(prov, indent=1))
    for k in ("small", "large", "fb_small", "fb_large"):
        print(k, {x: prov[k][x] for x in ("nodes", "edges", "labelled", "max_degree", "sum_deg2")})
    print(q)


if __name__ == "__main__":
    main()
