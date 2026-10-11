<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/brand-dark.svg">
  <img src="assets/brand.svg" alt="biostat-copilot" width="760">
</picture>

**バイオインフォマティクスと医学研究のための「エビデンス優先」ワークフロー・スキル —— AI エージェントがそのまま実行できる 43 の解析手法。**

[![Version](https://img.shields.io/badge/version-1.8.0-0d9488)](VERSION)
[![License: MIT](https://img.shields.io/badge/License-MIT-8b5cf6.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-2563eb)](#動作要件)
[![Self-check](https://img.shields.io/badge/self--check-216%20%2F%20219-16a34a)](tests/run_all.py)
[![Checks](https://github.com/cndoin/med-biostat-skill/actions/workflows/checks.yml/badge.svg)](https://github.com/cndoin/med-biostat-skill/actions/workflows/checks.yml)
[![Agents](https://img.shields.io/badge/agents-Claude%20Code%20%7C%20Codex%20%7C%20WorkBuddy%20%7C%20Cursor-18181b)](#ai-エージェントへのインストール)

[English](README.md) · [简体中文](README.zh-CN.md) · **日本語**

[クイックスタート](#クイックスタート) · [解析手法](#解析手法) · [インストール](#ai-エージェントへのインストール) · [ライセンス](#ライセンス)

</div>

![biostat-copilot の全体像](assets/banner.png)

## これは何か

医学・バイオインフォマティクス研究のための**クロスエージェント研究スキル**です。Claude Code、Codex、WorkBuddy などに導入すると、*「この GWAS 要約統計量を解析して」*、*「この遺伝子は白内障と因果関係があるか」*、*「この結果が正しいか検算して」*といった依頼に対して**自動的に読み込まれ**、固定された方法論フレームワークの中で解析を実行します —— モデルにコードを即興で書かせるのではありません。

解決する課題は「AI がコードを書けるかどうか」ではなく、**算出定義（口径）**です。

- λGC には正当な定義が 2 通りあり、混用すると同じファイルから 2 つの数値が出ます。
- 欠測値の扱い（補完する／しない／どう補完する）は下流の結論すべてを変えます。
- 「公式リリースと一致」は、ハッシュで証明せずに「値が近い」と主張しても検証にはなりません。

そこで本スキルは**定義を第一級の対象**として扱います。定義が書かれていない結果は無効とし、計算できるものはスクリプトで計算し、検証できるものはハッシュで検証します。

## 3 つの中核原則

| # | 原則 | 意味 |
|---|---|---|
| 1 | **数値より定義が先** | 同じ名前でも定義が違えば別の量です。定義が欠けていればエラーとし、既定値で埋めません。 |
| 2 | **「正しく見える」より一貫性** | 「公式と一致」はハッシュとバイト数で証明するもので、「だいたい合っている」から推論するものではありません。 |
| 3 | **交絡が主戦場** | 失敗の多くはアルゴリズムではなく交絡にあります。バイアス監査は独立した層として 12 項目を順に確認します。 |

## 解析手法

| グループ | 数 | 内容 |
|---|---|---|
| 因果と遺伝 | 10 | メンデルランダム化（7 推定量 + 7 感度分析）、創薬ターゲット cis-MR チェーン、共局在、ファインマッピング（SuSiE）、PRS、GWAS 品質管理、独立座位、観察データの因果推論（PSM / IPTW / AIPW）、媒介分析、DiD |
| オミクスとパスウェイ | 10 | 発現差解析、WGCNA 共発現ネットワーク、ORA、GSEA、GSVA / ssGSEA、単一細胞パイプライン、免疫浸潤、擬似時間、空間トランスクリプトーム、変異シグネチャ |
| 臨床統計とモデリング | 11 | 生存解析、競合リスク（CIF）、予測モデル（キャリブレーション + DCA）、C-index / NRI / IDI、正則化回帰（LASSO / Cox-LASSO）、メタアナリシス、出版バイアス、MICE 多重補完、多重検定補正、混合効果、検出力とサンプルサイズ |
| データエンジニアリングと眼科 | 4 | データクリーニングと変数選択、IOL 屈折計算、視力・屈折アウトカム、水晶体遺伝子セット解析 |
| 前処理と細胞間相互作用 | 2 | ComBat バッチ効果補正、細胞間コミュニケーション（リガンド–受容体） |
| 拡張手法 | 4 | 遺伝子–環境交互作用（G×E）、EWAS メチル化、双方向・媒介 MR、細胞型アノテーション |

各手法のエントリポイント、必要な定義、よくある落とし穴は [`SKILL.md`](SKILL.md) にまとめています。

## クイックスタート

### 動作要件

| 項目 | 要件 |
|---|---|
| Python | **3.9+**（Tier 0 は標準ライブラリのみ） |
| OS | Windows / macOS / Linux |
| 任意の科学スタック | `numpy` + `pandas` + `scipy`（Tier 1 のみ） |

```bash
git clone https://github.com/cndoin/med-biostat-skill.git
cd med-biostat-skill
```

### 1. セルフチェックを実行（サードパーティ依存なし）

Tier 0 は純粋な標準ライブラリなので、クローン直後にそのまま動きます。

```bash
python tests/run_all.py            # 全体セルフチェック
python scripts/capabilities.py     # この環境で使える層を判定
python scripts/doctor.py           # 環境と依存のレポート
```

### 2. Tier 1 をインストール（任意・完全な機能に必要）

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r vendor/requirements.txt
```

インストール後、`scripts/capabilities.py` が Tier 1 を「準備完了」と報告します。**依存が欠けても偽の失敗は起きません** —— 実行できないケースは FAIL ではなく SKIP として扱われます。

### 3. 自動トリガーを確認

```bash
python scripts/trigger_check.py
```

29 件のポジティブ例と 4 件のネガティブ例で、どの言い回しがスキルを読み込ませ、どれが読み込ませてはいけないかを検証します。

## AI エージェントへのインストール

本スキルは **唯一の `SKILL.md` を正とし、各エージェントのエントリを生成器で作る**方針です。手書きの重複は作りません（エントリが乖離する原因になります）。すべて `scripts/install_adapters.py` が生成します。

```bash
# 個人レベルのスキルディレクトリ（各エージェントが自動で読み込む）
python scripts/install_adapters.py --target claude       # → ~/.claude/skills/biostat-copilot/
python scripts/install_adapters.py --target codex        # → ~/.codex/skills/biostat-copilot/
python scripts/install_adapters.py --target workbuddy    # → ~/.workbuddy/skills/biostat-copilot/

# プロジェクト単位のルールファイル
python scripts/install_adapters.py --target cursor  --project /path/to/proj   # .cursor/rules/
python scripts/install_adapters.py --target cline   --project /path/to/proj   # .clinerules/
python scripts/install_adapters.py --target gemini  --project /path/to/proj   # GEMINI.md
python scripts/install_adapters.py --target copilot --project /path/to/proj   # .github/copilot-instructions.md
python scripts/install_adapters.py --target agents  --project /path/to/proj   # AGENTS.md

# まとめて導入
python scripts/install_adapters.py --target all --project /path/to/proj
```

| エージェント / ツール | エントリ | ターゲット |
|---|---|---|
| Claude Code | `SKILL.md` | `claude` |
| Codex | `SKILL.md` | `codex` |
| WorkBuddy | `SKILL.md` | `workbuddy` |
| Cursor | `.cursor/rules/<name>.mdc` | `cursor` |
| Cline | `.clinerules/<name>.md` | `cline` |
| Gemini CLI | `GEMINI.md` | `gemini` |
| GitHub Copilot | `.github/copilot-instructions.md` | `copilot` |
| 汎用（AGENTS.md 系） | `AGENTS.md` | `agents` |

**読み込まれたかの確認：** エージェントに *「生存解析と予測モデルを実行して」* と依頼してください。まず定義を確定し、スクリプトを実行し、定義を添えた結果を返し始めれば有効です。ローカルでは `python scripts/trigger_check.py` でトリガー範囲を確認できます。

> インストーラは**その場で上書き同期**します（冪等・再実行安全）。対象ディレクトリの他の内容は破壊しません。

## 3 層アーキテクチャ

| 層 | 場所 | 依存 | できること |
|---|---|---|---|
| **Tier 0** | `scripts/` | Python 標準ライブラリ | 取得 / 検証 / QC / 座位カウント / クレディブルセット絞り込み / 領域注釈 / 交絡監査 / 変異単位の相互検証 / レポートゲート |
| **Tier 1** | `vendor/` + `lib/` | numpy + pandas + scipy | 43 手法の大半、純 SVG グラフ、関数レジストリ、報告規範チェックリスト |
| **ネイティブ層** | `vendor/native/` | なし（単一実行ファイル） | 数百万行の GWAS 要約統計のストリーミング QC（300 万行 551 ms、Python の約 11 倍） |

Tier 1 の 22 モジュールは** 1 つずつインポート可否を判定**します。1 つの依存欠落が他を「利用不可」と誤判定することはありません。

## アルゴリズムの出典

**基本方針：アルゴリズムを自作しない。** 統計・光学のルーチンはすべてオープンソースプロジェクトから取得し、核心部分を抽出して依存を差し替えて同梱し、ファイル先頭に由来・ライセンス・既知の差分を明記しています。

| 上流プロジェクト | ライセンス | 提供する機能 | 同梱先 |
|---|---|---|---|
| [pymr](https://github.com/maxghenis/pymr) | Unlicense | メンデルランダム化一式 | `vendor/_mr/` |
| [GSEApy](https://github.com/zqfang/GSEApy) | BSD-3-Clause | エンリッチメント統計、GSEA コア、ssGSEA | `vendor/_enrich/` |
| [scanpy](https://github.com/scverse/scanpy) | BSD-3-Clause | 単一細胞パイプライン、ComBat | `vendor/_sc/` `vendor/_batch/` |
| [PyWGCNA](https://github.com/mortazavilab/PyWGCNA) | MIT | WGCNA 共発現ネットワーク | `vendor/_wgcna/` |
| [CellPhoneDB](https://github.com/ventolab/CellphoneDB) | MIT | 細胞間コミュニケーション | `vendor/_cellcomm/` |
| [pearldgs_toolbox](https://github.com/gdebel/pearldgs_toolbox) | MIT | IOL 光学 | `vendor/_ophtha/` |
| rsomics-vcf | MIT OR Apache-2.0 | GWAS ネイティブ高速化（ソースレベル fork） | `vendor/native/` |

**GPL 系プロジェクトは同梱しません**（伝播するため）。必要時に取得して外部ツールとして呼び出します。

```bash
python tools/fetch_tools.py --fetch lifelines
```

由来・ハッシュ・重複排除の判断は [`vendor/ATTRIBUTION.md`](vendor/ATTRIBUTION.md) に、サードパーティのライセンス全文は各 `vendor/_*/LICENSE` に同梱しています。

## これは何ではないか

- **テンプレート生成器ではありません。** 「七点セット」の GEO チップ解析（DEG + WGCNA + LASSO + 免疫浸潤）や、フェロトーシス / 銅死 / m6A のテンプレートは入口で拒否します。
- **数値生成器ではありません。** すべての数値は再実行可能なスクリプト、または引用可能な公式リリース値（定義を明記）に由来します。
- **方向性の営業役ではありません。** すでにカバー済みと判定された方向に対しては、言い換えて包み直すのではなく**止める**のが務めです。

## リポジトリ構成

```
med-biostat-skill/
├── SKILL.md              エントリ（ルーティング · レッドライン · ゲート · 手法一覧）
├── VERSION
├── README.md             English
├── README.zh-CN.md       简体中文
├── README.ja.md          日本語（本ファイル）
├── references/           知識層（必要時に読み込み、常駐コストなし）
├── scripts/              Tier 0（標準ライブラリ）+ クロスエージェントインストーラ
├── lib/                  Tier 1 拡張（22 モジュール）
├── vendor/               Tier 1 同梱層（由来は ATTRIBUTION.md）
├── pipeline/             決定論的パイプライン（AI は数値に触れない）
├── tools/                vendor_sync · fetch_tools · wrapper_core · native_accel
├── assets/               ブランド · バナー · テンプレート
├── adapters/             エージェント別エントリテンプレート
└── tests/                セルフチェック層
```

## セルフチェックと品質ゲート

```bash
python tests/run_all.py            # 216 / 219 通過（3 件は lifelines 未導入のためスキップ）
python tests/run_all.py --json     # 機械可読
python scripts/capabilities.py     # 2 層の能力レポート
python scripts/trigger_check.py    # 自動トリガー範囲（29 ポジティブ / 4 ネガティブ）
```

**ゲートは失敗することを示せて初めて有効です。** `tests/test_guards_negative.py` は退化した検出器（定数出力・意味の反転）と偽データを意図的に作り、ゲートが赤くなることを確認します。ゲートが飾りであることによる「失敗 0 件」は、ゲートが無いより危険です。

## ライセンス

**MIT** —— [`LICENSE`](LICENSE) を参照してください。

同梱のサードパーティコンポーネントは元のライセンス（Unlicense / BSD-3-Clause / MIT）を保持します。全文は各 `vendor/_*/LICENSE` に、一覧は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) に、由来とハッシュの完全な記録は [`vendor/ATTRIBUTION.md`](vendor/ATTRIBUTION.md) にあります。
