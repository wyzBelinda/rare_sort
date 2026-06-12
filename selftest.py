"""End-to-end self-test for the Layer 0/1 core.

Run:  python -m ranker.selftest
Builds synthetic cases with hand-computed expected sub-scores, asserts the
contributors reproduce them, asserts theta==1 == sum of components, then runs
the harness and checks recall@k / MRR / masking / collapse behave correctly.
"""
from __future__ import annotations

import pandas as pd

from . import (Case, default_registry, default_theta, evaluate,
               evaluate_all_conditions, case_cv_folds, score)
from ._util import variant_key
from .contributors import (_clinvar, _consequence, _frequency, _prediction,
                           _domain, _splice_lof)

COLS = ["chrom", "pos", "ref", "alt", "gene_symbol", "consequence", "impact",
        "protein_domains", "revel_score", "cadd_phred", "gnomAD_popmax_AF",
        "gnomAD_eas_AF", "gnomAD_nhomalt", "spliceAI_ds_max", "spliceAI_type",
        "loftee_lof_flag", "loftee_lof_filter", "clinvar_significance",
        "clinvar_review_status", "clinvar_star_rating", "transcript_id"]


def row(**kw) -> dict:
    base = {c: "-" for c in COLS}
    base.update(kw)
    return base


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_component_scores():
    # BRAF V600E-like: Pathogenic, 2-star, multiple submitters no conflicts
    r = row(consequence="missense_variant", impact="MODERATE",
            protein_domains="Pfam:PF07714", revel_score="0.95", cadd_phred="32",
            gnomAD_eas_AF="0.00005", gnomAD_popmax_AF="0.00005",
            clinvar_significance="Pathogenic", clinvar_star_rating="2",
            clinvar_review_status="criteria provided, multiple submitters, no conflicts")
    assert _clinvar(r) == 48, _clinvar(r)            # round(40*1.10*1.10*1.0)
    assert _consequence(r) == 15, _consequence(r)    # 12 + 3
    assert _prediction(r) == 27, _prediction(r)      # 15 + 12
    assert _frequency(r) == 10, _frequency(r)        # EAS <1e-4
    assert _domain(r) == 3
    assert _splice_lof(r) == 0

    # Benign common intron, many homozygotes
    b = row(consequence="intron_variant", impact="MODIFIER",
            gnomAD_eas_AF="0.2", gnomAD_popmax_AF="0.2", gnomAD_nhomalt="50",
            clinvar_significance="Benign", clinvar_star_rating="2",
            clinvar_review_status="criteria provided, multiple submitters, no conflicts")
    assert _clinvar(b) == -40, _clinvar(b)           # round(-35*1.10*1.10*0.95)
    assert _consequence(b) == -3, _consequence(b)    # 0 + (-3)
    assert _frequency(b) == -50, _frequency(b)       # -25 -10 -15
    print("  [ok] component scores match hand-computed values")


def test_theta_one_reproduces_raw():
    reg = default_registry()
    theta = default_theta(reg)
    r = row(consequence="missense_variant", impact="MODERATE",
            protein_domains="Pfam:PF07714", revel_score="0.95", cadd_phred="32",
            gnomAD_eas_AF="0.00005", gnomAD_popmax_AF="0.00005",
            clinvar_significance="Pathogenic", clinvar_star_rating="2",
            clinvar_review_status="criteria provided, multiple submitters, no conflicts")
    expected = 48 + 15 + 0 + 27 + 10 + 3   # = 103, the README raw_pathogenic_score
    got = score(r, theta, reg)
    assert approx(got, expected), (got, expected)
    print(f"  [ok] theta==1 reproduces README raw score ({got:.0f})")


def make_case(case_id, causal_gene="BRAF") -> Case:
    rows = [
        # causal: strong missense
        row(chrom="7", pos="140753336", ref="A", alt="T", gene_symbol=causal_gene,
            transcript_id="NM_1", consequence="missense_variant", impact="MODERATE",
            protein_domains="Pfam:PF07714", revel_score="0.95", cadd_phred="32",
            gnomAD_eas_AF="0.00005", gnomAD_popmax_AF="0.00005",
            clinvar_significance="Pathogenic", clinvar_star_rating="2",
            clinvar_review_status="criteria provided, multiple submitters, no conflicts"),
        # same causal variant, second transcript (collapse should merge these)
        row(chrom="7", pos="140753336", ref="A", alt="T", gene_symbol=causal_gene,
            transcript_id="NM_2", consequence="missense_variant", impact="LOW",
            revel_score="0.40", cadd_phred="12",
            gnomAD_eas_AF="0.00005", gnomAD_popmax_AF="0.00005"),
        # decoy 1: rare VUS missense (mid score)
        row(chrom="1", pos="1000", ref="C", alt="G", gene_symbol="DECOY1",
            transcript_id="NM_3", consequence="missense_variant", impact="MODERATE",
            protein_domains="Pfam:PFX", revel_score="0.60", cadd_phred="22",
            gnomAD_eas_AF="0.0005", gnomAD_popmax_AF="0.0005",
            clinvar_significance="Uncertain_significance", clinvar_star_rating="1",
            clinvar_review_status="criteria provided, single submitter"),
        # decoy 2: common benign intron (very negative)
        row(chrom="2", pos="2000", ref="G", alt="A", gene_symbol="DECOY2",
            transcript_id="NM_4", consequence="intron_variant", impact="MODIFIER",
            gnomAD_eas_AF="0.2", gnomAD_popmax_AF="0.2", gnomAD_nhomalt="50",
            clinvar_significance="Benign", clinvar_star_rating="2",
            clinvar_review_status="criteria provided, multiple submitters, no conflicts"),
    ]
    df = pd.DataFrame(rows)
    truth = variant_key("7", "140753336", "A", "T")
    return Case(case_id, df, causal_variants=[truth])


def test_ranking_and_metrics():
    reg = default_registry()
    theta = default_theta(reg)
    cases = [make_case(f"C{i}", causal_gene=g)
             for i, g in enumerate(["BRAF", "GENE2", "GENE3"])]

    res = evaluate(cases, theta, reg)
    v = res["levels"]["variant"]
    g = res["levels"]["gene"]
    assert v["recall@1"] == 1.0, v
    assert v["MRR"] == 1.0, v
    assert g["recall@1"] == 1.0, g
    # collapse worked: causal variant counted once even though it had 2 rows
    assert v["n_found"] == 3
    print(f"  [ok] variant recall@1={v['recall@1']}, gene recall@1={g['recall@1']}, MRR={v['MRR']}")


def test_masking():
    reg = default_registry()
    theta = default_theta(reg)
    cases = [make_case("C0")]

    full = evaluate(cases, theta, reg, masked_groups=frozenset())
    masked = evaluate(cases, theta, reg, masked_groups=frozenset({"clinvar"}))
    # causal still ranks #1 (strong non-clinvar evidence) under both
    assert full["levels"]["variant"]["recall@1"] == 1.0
    assert masked["levels"]["variant"]["recall@1"] == 1.0

    # but the causal score must drop when ClinVar evidence is removed
    r = cases[0].df.iloc[0].to_dict()
    s_full = score(r, theta, reg)
    s_mask = score(r, theta, reg, masked_groups=frozenset({"clinvar"}))
    assert approx(s_full - s_mask, 48), (s_full, s_mask)  # the pathogenic clinvar_score

    # phenotype mask is a harmless no-op in v1 (no phenotype contributor yet)
    pheno = evaluate(cases, theta, reg, masked_groups=frozenset({"phenotype"}))
    assert pheno["levels"]["variant"]["recall@1"] == 1.0

    conds = evaluate_all_conditions(cases, theta, reg)
    assert set(conds) == {"none", "clinvar_masked", "phenotype_masked", "both_masked"}
    print("  [ok] clinvar mask removes 48 pts; phenotype mask is a clean no-op; 2x2 runs")


def test_cv_by_case():
    cases = [make_case(f"C{i}") for i in range(10)]
    seen_test = set()
    for train, test in case_cv_folds(cases, n_folds=5, seed=1):
        train_ids = {c.case_id for c in train}
        test_ids = {c.case_id for c in test}
        assert train_ids.isdisjoint(test_ids)         # no case leaks across split
        seen_test |= test_ids
    assert seen_test == {f"C{i}" for i in range(10)}   # every case tested once
    print("  [ok] 5-fold CV splits by case, disjoint, full coverage")


# ----------------------------------------------------------------------------
# Layer 3/4 compatibility tests (added in the refactor)
# ----------------------------------------------------------------------------
def test_prepare_fastpath_matches_cases():
    """Prepared (parse-once) path must give identical metrics to the Case path."""
    from . import prepare, apply_theta, extract_features
    reg = default_registry(); theta = default_theta(reg)
    cases = [make_case(f"C{i}") for i in range(4)]
    via_cases = evaluate(cases, theta, reg)
    prepared = prepare(cases, reg)
    via_prepared = evaluate(prepared, theta)            # no registry needed
    assert via_cases["levels"]["variant"] == via_prepared["levels"]["variant"]
    b = extract_features(cases[0].df, reg)
    s_matrix = apply_theta(b, theta)
    s_loop = [score(r, theta, reg) for r in cases[0].df.to_dict("records")]
    assert all(approx(a, c) for a, c in zip(s_matrix, s_loop))
    print("  [ok] prepare() fast path == Case path; apply_theta == per-row score")


def test_theta_vector_roundtrip_and_bounds():
    from . import theta_keys, theta_to_vector, vector_to_theta, default_bounds
    reg = default_registry(); keys = theta_keys(reg)
    theta = {k: i + 0.5 for i, k in enumerate(keys)}
    assert vector_to_theta(theta_to_vector(theta, keys), keys) == theta
    assert all(lo == 0.0 for lo, _ in default_bounds(reg).values())
    print("  [ok] theta<->vector round-trips; bounds are non-negative")


def test_collapse_respects_selected_transcript():
    from . import extract_features, apply_theta, rank_units
    reg = default_registry(); theta = default_theta(reg)
    rows = [
        row(chrom="9", pos="500", ref="A", alt="G", gene_symbol="GX",
            tx_rank_within_variant="1", consequence="missense_variant",
            impact="MODERATE", revel_score="0.30", cadd_phred="12"),
        row(chrom="9", pos="500", ref="A", alt="G", gene_symbol="GX",
            tx_rank_within_variant="2", consequence="stop_gained", impact="HIGH",
            loftee_lof_flag="HC", cadd_phred="35"),
    ]
    df = pd.DataFrame(rows); b = extract_features(df, reg); s = apply_theta(b, theta)
    sel = rank_units(b, s, level="variant", collapse="auto").iloc[0]
    mx = rank_units(b, s, level="variant", collapse="max").iloc[0]
    assert sel["_rep_row"] == 0 and mx["_rep_row"] == 1
    assert sel["score"] < mx["score"]
    print(f"  [ok] tx-aware collapse keeps selected tx (score {sel['score']:.0f}), "
          f"not the shopped LoF transcript ({mx['score']:.0f})")


def test_explain_failures_structure():
    from . import explain_failures
    reg = default_registry(); theta = default_theta(reg)
    cases = [make_case("C0")]
    fails = explain_failures(cases, theta, reg, k=0)    # nothing can rank <=0
    assert fails and fails[0]["causal_rank"] == 1
    f = fails[0]
    assert "causal_contributions" in f and "blockers" in f
    assert set(f["causal_contributions"]) == set(default_theta(reg))
    print(f"  [ok] explain_failures gives per-contributor breakdown "
          f"(causal_rank={f['causal_rank']}, {len(f['blockers'])} blockers)")


def test_bootstrap_and_describe():
    from . import bootstrap_ci, describe_model
    import json as _json
    reg = default_registry(); theta = default_theta(reg)
    cases = [make_case(f"C{i}") for i in range(8)]
    ci = bootstrap_ci(cases, theta, lambda res: res["levels"]["variant"]["MRR"],
                      registry=reg, n_boot=200, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    desc = describe_model(reg, theta, metrics={"variant_MRR": ci["point"]},
                          data_snapshot="P001..P050 @ clinvar_20260523",
                          rationale="README defaults, theta==1 baseline")
    _json.dumps(desc)
    assert len(desc["version_signature"]) == 16
    print(f"  [ok] bootstrap CI MRR={ci['point']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}]; "
          f"signature={desc['version_signature']}")


if __name__ == "__main__":
    print("running ranker Layer 0/1 + L3/4-hook self-test")
    test_component_scores()
    test_theta_one_reproduces_raw()
    test_ranking_and_metrics()
    test_masking()
    test_cv_by_case()
    test_prepare_fastpath_matches_cases()
    test_theta_vector_roundtrip_and_bounds()
    test_collapse_respects_selected_transcript()
    test_explain_failures_structure()
    test_bootstrap_and_describe()
    print("ALL PASSED")
