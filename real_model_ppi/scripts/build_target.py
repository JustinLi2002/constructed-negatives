"""Label-free construction of the declared target, the two regions, the samples
and gates G2 and G3 of the real-model PPI test (DESIGN.md sections 3-5 and 8).

No model is run and no study score is read.  The only prediction scores touched
are SNOOPPI's released TUnA-R scores for the 300 G1 reference pairs, which are
copied unread into the G1 input file.

    python -I build_target.py OUT_DIR
"""
import gzip
import hashlib
import json
import os
import site
import sys
import time

sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RM = os.path.dirname(HERE)                       # real_model_ppi/
IDENT = os.path.dirname(RM)                      # identification/
sys.path.insert(0, IDENT)
from cl3_exclusion import config_threshold, load_edges  # noqa: E402

SEED = 20261118
N_SAMPLE = 10_000
N_G1_TUNA = 300
N_CLI_CHECK = 50
SOURCE_RATIO = 0.064
AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
LMIN, LMAX = 50, 800
DL = os.path.join(RM, "downloads")
EDGES = os.path.join(IDENT, "data", "string_phys_700.tsv")
LINKS = os.path.join(IDENT, "data", "string_physical_9606.txt.gz")


def read_fasta(path):
    opener = gzip.open if path.endswith(".gz") else open
    seqs, name, buf = {}, None, []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf).upper()
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf).upper()
    return seqs


def shash(seq):
    return int.from_bytes(hashlib.sha1(seq.encode()).digest()[:8], "big", signed=True)


def read_pairs(path):
    out = []
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 2:
                out.append((f[0], f[1]))
    return out


def scorable(seq):
    return seq is not None and LMIN <= len(seq) <= LMAX and set(seq) <= AA20


def main():
    out_dir = sys.argv[1]
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit(f"refusing to overwrite non-empty {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    rep = {"seed": SEED, "source_ratio": SOURCE_RATIO, "length_range": [LMIN, LMAX]}

    # ---- network, rule ----------------------------------------------------
    A, names = load_edges(EDGES)
    n = len(names)
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    tau = int(config_threshold(deg, SOURCE_RATIO))
    iu, iv = A.nonzero()
    keep = iu < iv
    edge_codes = np.unique(iu[keep].astype(np.int64) * n + iv[keep])
    m = len(edge_codes)
    rep.update(proteins=n, edges=int(m), tau=tau)
    assert (n, m, tau) == (10746, 86519, 2), "network or rule differs from the recorded counts"

    # ---- sequences, scorability ------------------------------------------
    sseq = read_fasta(os.path.join(DL, "string_seqs", "9606.protein.sequences.v12.0.fa.gz"))
    seq = [sseq.get(x) for x in names]
    S = np.array([scorable(s) for s in seq])
    h = np.array([shash(s) if s is not None else 0 for s in seq], dtype=np.int64)
    rep["sequences_missing"] = int(sum(s is None for s in seq))
    rep["scorable_proteins"] = int(S.sum())
    rep["not_scorable_by_length"] = int(sum(s is not None and not (LMIN <= len(s) <= LMAX) for s in seq))
    rep["not_scorable_by_alphabet"] = int(sum(s is not None and LMIN <= len(s) <= LMAX and not set(s) <= AA20
                                              for s in seq))
    byhash = {}
    for i, s in enumerate(seq):
        if s is not None:
            byhash.setdefault(int(h[i]), []).append(i)
    rep["string_proteins_sharing_a_sequence"] = int(sum(len(v) for v in byhash.values() if len(v) > 1))

    # ---- released datasets (exact sequence identity) ---------------------
    ds_seq = read_fasta(os.path.join(DL, "dscript_data", "human.fasta"))
    bt_seq = read_fasta(os.path.join(DL, "bernett_tuna", "human_swissprot_oneliner.fasta"))
    tg_seq = read_fasta(os.path.join(DL, "tunar_gold", "human_swissprot_oneliner.fasta"))
    released = [
        ("dscript/human_train.tsv", os.path.join(DL, "dscript_data", "human_train.tsv"), ds_seq),
        ("dscript/human_test.tsv", os.path.join(DL, "dscript_data", "human_test.tsv"), ds_seq),
    ] + [
        (f"bernett/{p}_{k}_rr.txt", os.path.join(DL, "bernett_tuna", f"{p}_{k}_rr.txt"), bt_seq)
        for p in ("Intra0", "Intra1", "Intra2") for k in ("pos", "neg")
    ] + [
        (f"tunar_gold/{p}_interaction_1500_or_less.tsv",
         os.path.join(DL, "tunar_gold", f"{p}_interaction_1500_or_less.tsv"), tg_seq)
        for p in ("Intra0", "Intra1", "Intra2")
    ]
    rel_keys, rel_prot = set(), set()
    g2 = {}
    id_pairs = {}
    for label, path, sq in released:
        pairs = read_pairs(path)
        ok = 0
        for a, b in pairs:
            sa, sb = sq.get(a), sq.get(b)
            if sa is None or sb is None:
                continue
            ok += 1
            ha, hb = shash(sa), shash(sb)
            rel_keys.add((min(ha, hb), max(ha, hb)))
            rel_prot.update((ha, hb))
        g2[label] = {"pairs": len(pairs), "resolved": ok, "share": ok / len(pairs)}
        id_pairs[label] = {tuple(sorted(p)) for p in pairs}
    bern_all = set().union(*(id_pairs[k] for k in id_pairs if k.startswith("bernett/")))
    gold_all = set().union(*(id_pairs[k] for k in id_pairs if k.startswith("tunar_gold/")))
    rep["tunar_gold_pairs_within_bernett_raw"] = f"{len(gold_all & bern_all)} of {len(gold_all)}"
    rep["released_sequence_pairs"] = len(rel_keys)
    rep["released_sequences"] = len(rel_prot)

    # released sequence pairs mapped onto STRING protein pairs
    rel_codes = set()
    for ha, hb in rel_keys:
        for u in byhash.get(ha, ()):
            for v in byhash.get(hb, ()):
                if u != v:
                    rel_codes.add(min(u, v) * n + max(u, v))
    rel_codes = np.array(sorted(rel_codes), dtype=np.int64)
    is_edge_rel = np.isin(rel_codes, edge_codes)
    ru, rv = rel_codes // n, rel_codes % n
    in_S_rel = S[ru] & S[rv]
    relNE_S = rel_codes[(~is_edge_rel) & in_S_rel]
    rep["released_pairs_on_string"] = int(len(rel_codes))
    rep["released_pairs_on_string_that_are_edges"] = int(is_edge_rel.sum())
    rep["released_nonedge_pairs_in_scorable_set"] = int(len(relNE_S))

    # ---- unrestricted W (check against the recorded 0.9248) --------------
    L = np.flatnonzero(deg <= tau)
    o_all = []
    for u in L:
        v = L[L > u]
        v = v[deg[u] * deg[v] <= tau]
        o_all.append(u * n + v)
    o_all = np.unique(np.concatenate(o_all))
    o_all = o_all[~np.isin(o_all, edge_codes)]
    nonedges = n * (n - 1) // 2 - m
    rep["unrestricted"] = {"O": int(len(o_all)), "U": int(nonedges - len(o_all)),
                           "W": (nonedges - len(o_all)) / nonedges}
    assert len(o_all) == 4_334_373

    # ---- declared target T, regions O_T and U_T --------------------------
    ou, ov = o_all // n, o_all % n
    O_T = o_all[S[ou] & S[ov]]
    O_T = O_T[~np.isin(O_T, relNE_S)]
    eu, ev = edge_codes // n, edge_codes % n
    edges_S = int((S[eu] & S[ev]).sum())
    nS = int(S.sum())
    T = nS * (nS - 1) // 2 - edges_S - len(relNE_S)
    U_T = T - len(O_T)
    W_T = U_T / T
    rep["target"] = {"T": int(T), "O_T": int(len(O_T)), "U_T": int(U_T), "W_T": W_T,
                     "eligible_pairs_lost_to_scorability": int((~(S[ou] & S[ov])).sum()),
                     "eligible_scorable_pairs_lost_to_release": int(
                         (S[ou] & S[ov]).sum() - len(O_T))}

    # per-protein partner counts for G3
    oTu, oTv = O_T // n, O_T % n
    O_deg = np.bincount(np.concatenate([oTu, oTv]), minlength=n)
    E_S_deg = np.bincount(np.concatenate([eu[S[eu] & S[ev]], ev[S[eu] & S[ev]]]), minlength=n)
    R_deg = np.bincount(np.concatenate([relNE_S // n, relNE_S % n]), minlength=n)
    U_deg = np.where(S, (nS - 1) - E_S_deg - R_deg - O_deg, 0)
    assert U_deg.min() >= 0 and int(U_deg.sum()) == 2 * U_T

    # ---- positives --------------------------------------------------------
    pos_pool = edge_codes[S[eu] & S[ev]]
    pos_pool = pos_pool[~np.isin(pos_pool, rel_codes)]
    rep["positives"] = {"edges_scorable": edges_S, "edges_scorable_not_released": int(len(pos_pool)),
                        "edges_released": int(edges_S - len(pos_pool))}

    # ---- samples (three independent streams from one seed) ---------------
    ss_pos, ss_o, ss_u = np.random.SeedSequence(SEED).spawn(3)
    k_pos = min(N_SAMPLE, len(pos_pool))
    pos_s = np.sort(pos_pool[np.random.default_rng(ss_pos).choice(len(pos_pool), k_pos, replace=False)])
    o_s = np.sort(O_T[np.random.default_rng(ss_o).choice(len(O_T), N_SAMPLE, replace=False)])

    rng_u = np.random.default_rng(ss_u)
    S_idx = np.flatnonzero(S)
    edge_set, rel_set = set(edge_codes.tolist()), set(rel_codes.tolist())
    acc, u_list, draws = set(), [], 0
    while len(u_list) < N_SAMPLE:
        a = S_idx[rng_u.integers(0, len(S_idx), 4096)]
        b = S_idx[rng_u.integers(0, len(S_idx), 4096)]
        for x, y in zip(a.tolist(), b.tolist()):
            draws += 1
            if x == y:
                continue
            u, v = (x, y) if x < y else (y, x)
            if deg[u] * deg[v] <= tau:
                continue
            c = u * n + v
            if c in edge_set or c in rel_set or c in acc:
                continue
            acc.add(c)
            u_list.append(c)
            if len(u_list) == N_SAMPLE:
                break
    u_s = np.sort(np.array(u_list, dtype=np.int64))
    rep["u_sampler_draws"] = draws

    def prot(codes):
        return np.unique(np.concatenate([codes // n, codes % n]))

    # ---- gates ------------------------------------------------------------
    g3 = {
        "O_T_distinct_proteins": int(len(prot(O_T))),
        "U_T_distinct_proteins": int((U_deg > 0).sum()),
        "positive_pairs": int(len(pos_s)),
        "positive_distinct_proteins": int(len(prot(pos_s))),
        "O_sample_distinct_proteins": int(len(prot(o_s))),
        "U_sample_distinct_proteins": int(len(prot(u_s))),
    }
    g3_pass = (g3["O_T_distinct_proteins"] >= 1000 and g3["U_T_distinct_proteins"] >= 1000
               and g3["positive_pairs"] >= 2000 and g3["positive_distinct_proteins"] >= 1000
               and g3["O_sample_distinct_proteins"] >= 500 and g3["U_sample_distinct_proteins"] >= 500)
    g2_pass = all(v["share"] >= 0.95 for v in g2.values())
    rep["G2"] = {"files": g2, "pass": bool(g2_pass)}
    rep["G3"] = {**g3, "pass": bool(g3_pass)}

    # ---- descriptive ------------------------------------------------------
    Rp = np.isin(h, np.array(sorted(rel_prot), dtype=np.int64)) & S
    nSR = int(Rp.sum())
    edges_SR = int((Rp[eu] & Rp[ev]).sum())
    O_RR = int((Rp[oTu] & Rp[oTv]).sum())
    U_RR = nSR * (nSR - 1) // 2 - edges_SR - len(relNE_S) - O_RR
    rep["descriptive_released_proteins"] = {
        "scorable_proteins_seen_in_released_data": nSR,
        "share_O_T_both_seen": O_RR / len(O_T), "share_U_T_both_seen": U_RR / U_T,
        "share_T_both_seen": (O_RR + U_RR) / T}

    mid = set()
    idx = {x: i for i, x in enumerate(names)}
    with gzip.open(LINKS, "rt") as fh:
        next(fh)
        for line in fh:
            a, b, s = line.split()
            s = int(s)
            if 400 <= s < 700 and a in idx and b in idx:
                u, v = idx[a], idx[b]
                mid.add(min(u, v) * n + max(u, v))
    mid = np.array(sorted(mid), dtype=np.int64)
    mu, mv = mid // n, mid % n
    mid_T = mid[S[mu] & S[mv] & ~np.isin(mid, edge_codes) & ~np.isin(mid, relNE_S)]
    mid_O = int(np.isin(mid_T, O_T).sum())
    rep["descriptive_string_400_700"] = {
        "share_O_T": mid_O / len(O_T), "share_U_T": (len(mid_T) - mid_O) / U_T,
        "share_O_sample": float(np.isin(o_s, mid).mean()), "share_U_sample": float(np.isin(u_s, mid).mean())}

    # ---- write samples ----------------------------------------------------
    for label, codes in (("positives", pos_s), ("O", o_s), ("U", u_s)):
        with open(os.path.join(out_dir, f"pairs_{label}.tsv"), "w", newline="\n") as out:
            out.write("protein_a\tprotein_b\n")
            for c in codes.tolist():
                out.write(f"{names[c // n]}\t{names[c % n]}\n")
    need = sorted(set(prot(np.concatenate([pos_s, o_s, u_s])).tolist()))
    with open(os.path.join(out_dir, "sequences.fasta"), "w", newline="\n") as out:
        for i in need:
            out.write(f">{names[i]}\n{seq[i]}\n")
    rep["sequences_written"] = len(need)

    # ---- G1 inputs -----------------------------------------------------------
    import pyarrow.parquet as pq
    tab = pq.read_table(os.path.join(DL, "snooppi_positive", "snooppi_positive__all_models.parquet"),
                        columns=["pair_sequence_1", "pair_sequence_2", "tuna_score"]).to_pydict()
    rows = np.sort(np.random.default_rng(SEED).choice(len(tab["tuna_score"]), N_G1_TUNA, replace=False))
    with open(os.path.join(out_dir, "g1_tuna_pairs.tsv"), "w", newline="\n") as out:
        out.write("row_index\tseq_a\tseq_b\tref_tuna_score\n")
        for r in rows.tolist():
            out.write(f"{r}\t{tab['pair_sequence_1'][r]}\t{tab['pair_sequence_2'][r]}\t"
                      f"{tab['tuna_score'][r]!r}\n")
    test = [line.rstrip("\n").split("\t") for line in open(os.path.join(DL, "dscript_data", "human_test.tsv"))]
    pick = np.sort(np.random.default_rng(SEED).choice(len(test), N_CLI_CHECK, replace=False))
    with open(os.path.join(out_dir, "g1_dscript_cli_subset.tsv"), "w", newline="\n") as out:
        for r in pick.tolist():
            out.write("\t".join(test[r][:2]) + "\n")
    rep["G1_inputs"] = {"tuna_rows": len(rows), "dscript_cli_subset_pairs": len(pick),
                        "dscript_test_pairs": len(test)}

    rep["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "target_summary.json"), "w", newline="\n") as out:
        json.dump(rep, out, indent=1)
    print(json.dumps({k: rep[k] for k in ("target", "positives", "G2", "G3")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
