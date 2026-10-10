"""分析法 ⑫：肿瘤突变特征分析（SBS 96 通道谱 + NMF 分解 + TMB + 特征匹配）。

流程（与 SigProfiler / deconstructSigs 的口径对齐）：
  1. `sbs96_profile`  — 把突变按「三核苷酸上下文 + 替换类型」分箱成 96 通道谱
  2. `nmf_decompose`  — 用 NMF 把 样本×96 矩阵分解成 特征谱 × 暴露
  3. `match_cosmic`   — 把分解出的特征与参考特征做余弦相似度匹配（需自提供参考文献谱）
  4. `tmb`            — 肿瘤突变负荷（每 Mb 的非同义突变数）

⚠️ 许可说明：COSMIC 的参考特征谱**不是开源数据**（需接受 COSMIC 许可条款），
   未内嵌。本模块不自带参考谱，需要调用方自行提供（可从 COSMIC 下载后传入）。
   没有参考谱 ≠ 不能做分解：NMF 分解本身不需要参考，只是无法**命名**特征。

⚠️ NMF 的两个坑：
  1. **秩 k 必须人工指定并做稳定性检验**（不同 k 给出不同的特征数）
  2. NMF 不唯一（旋转不唯一），多次随机初始化可能给出不同分解——
     必须报告 `n_init` 与重构误差，并按多次运行的一致性判断。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# SBS96 的标准通道顺序（COSMIC v3 定义）：
# 6 种替换 × 4 种 5' 上下文 × 4 种 3' 上下文 = 96
SUBS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
BASES = ["A", "C", "G", "T"]


def sbs96_channels() -> List[str]:
    """返回 96 通道的标准名称（顺序固定，与 COSMIC 一致）。"""
    out = []
    for sub in SUBS:
        for five in BASES:
            for three in BASES:
                out.append(f"{five}[{sub}]{three}")
    return out


def _normalize_sub(ref: str, alt: str) -> Optional[Tuple[str, str, str]]:
    """把突变规范到嘧啶参考（C/T）的形式，返回 (替换类型, ref, alt)。

    若 ref/alt 不是单碱基 SNV（如插入缺失、MNVs）则返回 None。
    """
    ref = str(ref).upper()
    alt = str(alt).upper()
    pair = ref + ">" + alt
    flip = {"A": "T", "T": "A", "C": "G", "G": "C"}
    if pair in ("G>T", "G>C", "G>A", "A>T", "A>G", "A>C"):
        # 反向互补：G>T 等价于 C>A
        ref, alt = flip[ref], flip[alt]
    if ref + ">" + alt not in SUBS:
        return None
    return f"{ref}>{alt}", ref, alt


def sbs96_profile(df: pd.DataFrame, ref_col: str = "ref", alt_col: str = "alt",
                  ctx5_col: str = "ctx5", ctx3_col: str = "ctx3",
                  sample_col: Optional[str] = None,
                  normalize: bool = True) -> pd.DataFrame:
    """构建 SBS96 突变谱。

    :param df: 突变表；每行一个突变，需含 ref/alt 与上下游碱基（各 1 bp）
    :param normalize: True 时每个样本归一化为比例（和为 1）
    :return: 样本×96 的谱矩阵（若没有 sample_col 则只有一行 "all"）
    """
    channels = sbs96_channels()
    idx = {c: i for i, c in enumerate(channels)}
    if sample_col is None:
        samples = ["all"] * len(df)
        labels = ["all"]
    else:
        samples = [str(s) for s in df[sample_col]]
        labels = sorted(set(samples))
    mat = np.zeros((len(labels), 96))
    pos = {s: i for i, s in enumerate(labels)}
    n_skipped = 0
    for i in range(len(df)):
        norm = _normalize_sub(df[ref_col].iloc[i], df[alt_col].iloc[i])
        if norm is None:
            n_skipped += 1
            continue
        sub, _, _ = norm
        five = str(df[ctx5_col].iloc[i]).upper()
        three = str(df[ctx3_col].iloc[i]).upper()
        if five not in BASES or three not in BASES:
            n_skipped += 1
            continue
        # 颠换到嘧啶参考时，上下文也要反向互补
        ref = str(df[ref_col].iloc[i]).upper()
        if ref in ("G", "A"):
            flip = {"A": "T", "T": "A", "C": "G", "G": "C"}
            five, three = flip.get(three, three), flip.get(five, five)
        ch = f"{five}[{sub}]{three}"
        if ch in idx:
            mat[pos[samples[i]], idx[ch]] += 1
    if normalize:
        tot = mat.sum(axis=1, keepdims=True)
        mat = mat / np.where(tot == 0, 1, tot)
    out = pd.DataFrame(mat, index=labels, columns=channels)
    out.attrs["n_mutations"] = int(mat.sum()) if not normalize else int(len(df) - n_skipped)
    out.attrs["n_skipped"] = int(n_skipped)
    return out


def nmf_decompose(V: np.ndarray, k: int, n_init: int = 20, max_iter: int = 500,
                  tol: float = 1e-6, seed: int = 0) -> Dict[str, Any]:
    """NMF 分解：V ≈ W·H（乘法更新，Lee-Seung 2001）。

    :param V: 非负矩阵 样本×特征（如 样本×96 通道，值为比例或计数）
    :param k: 分解秩（特征个数）——**必须做不同 k 的稳定性检验**
    :return: W（样本×k 暴露）、H（k×96 特征谱）、重构误差、各次初始化的误差
    """
    V = np.asarray(V, dtype=float)
    if np.any(V < 0):
        raise ValueError("NMF 要求非负矩阵")
    n, m = V.shape
    k = int(min(k, min(n, m)))
    rng = np.random.default_rng(seed)
    best = None
    errors = []
    for init in range(n_init):
        W = rng.random((n, k)) + 0.1
        H = rng.random((k, m)) + 0.1
        prev = np.inf
        for _ in range(max_iter):
            # 乘法更新（Lee-Seung）
            WH = W @ H
            H *= (W.T @ V) / np.maximum(W.T @ WH, 1e-12)
            WH = W @ H
            W *= (V @ H.T) / np.maximum(W @ H @ H.T, 1e-12)
            err = float(np.linalg.norm(V - W @ H, "fro"))
            if abs(prev - err) < tol:
                break
            prev = err
        err = float(np.linalg.norm(V - W @ H, "fro"))
        errors.append(err)
        if best is None or err < best[2]:
            best = (W.copy(), H.copy(), err)
    W, H, err = best
    # 归一化：每个特征的谱和为 1，暴露相应缩放
    hs = H.sum(axis=1, keepdims=True)
    H = H / np.where(hs == 0, 1, hs)
    W = W * hs.T
    tot = float((V ** 2).sum())
    return {
        "W": W, "H": H, "k": k,
        "reconstruction_error": err,
        "relative_error": err / np.sqrt(tot) if tot > 0 else float("nan"),
        "errors_across_inits": np.array(errors),
        "error_sd_across_inits": float(np.std(errors)),
        "n_init": n_init,
        "caution": "NMF 解不唯一；k 需稳定性检验，多次初始化的误差差异要报告",
    }


def match_cosmic(H: np.ndarray, reference: pd.DataFrame,
                 threshold: float = 0.85) -> pd.DataFrame:
    """把分解出的特征与参考特征谱做余弦相似度匹配。

    :param H: 分解出的 k×96 特征谱
    :param reference: 参考谱，行=特征名（如 SBS1、SBS5），列=96 通道
    :return: 每个分解特征的最佳匹配（名称 + 余弦相似度 + 是否超过阈值）
    """
    H = np.asarray(H, dtype=float)
    R = reference.to_numpy(dtype=float)
    names = list(reference.index)
    rows = []
    for i in range(H.shape[0]):
        sims = []
        for j in range(R.shape[0]):
            a, b = H[i], R[j]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            sims.append(float(a @ b / denom) if denom > 0 else 0.0)
        j_best = int(np.argmax(sims))
        rows.append({"component": i, "best_match": names[j_best],
                     "cosine": float(sims[j_best]),
                     "above_threshold": bool(sims[j_best] >= threshold),
                     "second_match": names[int(np.argsort(sims)[-2])] if len(sims) > 1 else None,
                     "second_cosine": float(np.sort(sims)[-2]) if len(sims) > 1 else float("nan")})
    return pd.DataFrame(rows)


def tmb(n_mutations: int, panel_mb: Optional[float] = None,
        genome_mb: float = 32.0) -> Dict[str, Any]:
    """肿瘤突变负荷 = 突变数 / 覆盖的 Mb。

    ⚠️ **口径必须区分**：panel TMB（用 panel 覆盖的 Mb）与 WES/WGS TMB
    （常用 ~32 Mb 的外显子组近似）**不是同一个量**，绝不能混用或换算。
    只报数字不报用了哪个 Mb 分母 = 口径缺失。
    """
    if panel_mb is not None and panel_mb > 0:
        mb = float(panel_mb)
        caliber = "panel"
    else:
        mb = float(genome_mb)
        caliber = "exome_approx"
    return {
        "n_mutations": int(n_mutations),
        "mb": mb,
        "tmb": float(n_mutations) / mb,
        "caliber": caliber,
        "note": "panel 与 WES 的 TMB 不可互相换算；分母必须随结果报告",
    }


def apobec_enrichment(profile: pd.DataFrame) -> Dict[str, Any]:
    """APOBEC 特征富集（TCW 上下文中的 C>T/C>G）的粗粒度提示。

    这是**提示性指标**，不是正式的特征分析：APOBEC 的真正判据是
    SBS2/SBS13 的暴露量（需 NMF 分解后匹配），这个函数只做快速筛查。
    """
    cols = list(profile.columns)
    tcw = [i for i, c in enumerate(cols)
           if c.startswith("T[") and (c.endswith("A]T") or c.endswith("T]T")) and "C>" in c]
    if not tcw:
        return {"apobec_fraction": float("nan"), "note": "未找到 TCW 通道"}
    sub = profile.to_numpy(dtype=float)[:, tcw]
    frac = sub.sum(axis=1) / np.where(profile.to_numpy(dtype=float).sum(axis=1) == 0, 1,
                                      profile.to_numpy(dtype=float).sum(axis=1))
    return {"apobec_fraction": frac,
            "n_channels_used": len(tcw),
            "note": "提示性筛查；正式判据是 NMF 分解后 SBS2/SBS13 的暴露"}
