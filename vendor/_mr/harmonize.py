# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: pymr@maxghenis
# 原路径: _upstream/pymr/src/pymr/harmonize.py
# 许可: Unlicense（公有领域）
# 说明: 等位基因协调：列标准化 / 链翻转 / 回文位点剔除
# 源 sha256: ae2ee1550c5aa908
# VENDORED-FILE-END
"""GWAS data harmonization.

Harmonizes exposure and outcome GWAS summary statistics by:
1. Matching SNPs between datasets
2. Aligning effect alleles
3. Flipping effect sizes when needed
4. Handling palindromic SNPs
"""

import pandas as pd


def harmonize(
    exposure: pd.DataFrame,
    outcome: pd.DataFrame,
    action: int = 2,
    remove_palindromic: bool = True,
    maf_threshold: float = 0.42,
) -> pd.DataFrame:
    """Harmonize exposure and outcome GWAS data.

    Ensures effect alleles are aligned between exposure and outcome so that
    effect sizes are on the same scale.

    Args:
        exposure: Exposure GWAS with columns: SNP, effect_allele, other_allele,
            beta, se, and optionally eaf (effect allele frequency)
        outcome: Outcome GWAS with same columns
        action: How to handle ambiguous SNPs:
            1 = Assume all alleles coded on forward strand
            2 = Try to infer strand from allele frequencies
            3 = Remove all palindromic SNPs
        remove_palindromic: Whether to remove A/T and C/G SNPs with MAF ~0.5
        maf_threshold: MAF above which palindromic SNPs are ambiguous

    Returns:
        Harmonized DataFrame with columns:
            SNP, beta_exp, se_exp, beta_out, se_out, and optionally
            effect_allele, other_allele, eaf_exp, eaf_out
    """
    # Standardize column names
    exp = _standardize_columns(exposure, suffix="_exp")
    out = _standardize_columns(outcome, suffix="_out")

    # Merge on SNP
    merged = exp.merge(out, on="SNP", how="inner")

    if len(merged) == 0:
        msg = "No matching SNPs between exposure and outcome"
        raise ValueError(msg)

    # Identify and handle different allele scenarios
    merged = _align_alleles(merged)

    # Handle palindromic SNPs
    if remove_palindromic:
        merged = _remove_palindromic(merged, maf_threshold)

    # Clean up and return
    result = merged[
        ["SNP", "beta_exp", "se_exp", "beta_out", "se_out"]
        + [c for c in merged.columns if c.startswith("effect_allele")]
    ].copy()

    return result


def _standardize_columns(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Standardize column names to common format."""
    df = df.copy()

    # Column name mappings
    mappings = {
        "beta": f"beta{suffix}",
        "se": f"se{suffix}",
        "eaf": f"eaf{suffix}",
        "effect_allele": f"effect_allele{suffix}",
        "other_allele": f"other_allele{suffix}",
    }

    for old, new in mappings.items():
        if old in df.columns and new != old:
            df = df.rename(columns={old: new})

    return df


def _align_alleles(merged: pd.DataFrame) -> pd.DataFrame:
    """Align alleles between exposure and outcome."""
    merged = merged.copy()

    # Check if alleles match
    exp_ea = merged["effect_allele_exp"].str.upper()
    exp_oa = merged["other_allele_exp"].str.upper()
    out_ea = merged["effect_allele_out"].str.upper()
    out_oa = merged["other_allele_out"].str.upper()

    # Case 1: Effect alleles match (no action needed)
    match = (exp_ea == out_ea) & (exp_oa == out_oa)

    # Case 2: Alleles are flipped (need to flip outcome beta)
    flipped = (exp_ea == out_oa) & (exp_oa == out_ea)
    merged.loc[flipped, "beta_out"] = -merged.loc[flipped, "beta_out"]
    if "eaf_out" in merged.columns:
        merged.loc[flipped, "eaf_out"] = 1 - merged.loc[flipped, "eaf_out"]

    # Case 3: Strand flip (complement alleles)
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

    out_ea_comp = out_ea.map(lambda x: complement.get(x, x))
    out_oa_comp = out_oa.map(lambda x: complement.get(x, x))

    strand_flip = (exp_ea == out_ea_comp) & (exp_oa == out_oa_comp)
    strand_flip_and_flipped = (exp_ea == out_oa_comp) & (exp_oa == out_ea_comp)

    merged.loc[strand_flip_and_flipped, "beta_out"] = -merged.loc[
        strand_flip_and_flipped, "beta_out"
    ]
    if "eaf_out" in merged.columns:
        merged.loc[strand_flip_and_flipped, "eaf_out"] = (
            1 - merged.loc[strand_flip_and_flipped, "eaf_out"]
        )

    # Remove SNPs that couldn't be aligned
    aligned = match | flipped | strand_flip | strand_flip_and_flipped
    merged = merged[aligned].copy()

    return merged


def _remove_palindromic(
    merged: pd.DataFrame,
    maf_threshold: float = 0.42,
) -> pd.DataFrame:
    """Remove palindromic SNPs with ambiguous strand.

    Palindromic SNPs are A/T or C/G. When MAF is close to 0.5,
    we cannot determine strand from frequency.
    """
    exp_ea = merged["effect_allele_exp"].str.upper()
    exp_oa = merged["other_allele_exp"].str.upper()

    # Identify palindromic
    at_palindrome = ((exp_ea == "A") & (exp_oa == "T")) | (
        (exp_ea == "T") & (exp_oa == "A")
    )
    cg_palindrome = ((exp_ea == "C") & (exp_oa == "G")) | (
        (exp_ea == "G") & (exp_oa == "C")
    )
    palindromic = at_palindrome | cg_palindrome

    # Check if MAF is ambiguous
    if "eaf_exp" in merged.columns:
        maf = merged["eaf_exp"].apply(lambda x: min(x, 1 - x) if pd.notna(x) else 0.5)
        ambiguous = maf > maf_threshold
        to_remove = palindromic & ambiguous
    else:
        # If no frequency data, remove all palindromic
        to_remove = palindromic

    return merged[~to_remove].copy()
