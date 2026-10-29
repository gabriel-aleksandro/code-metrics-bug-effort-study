"""
================================================================================
 Análise Comparativa: CC/LOC vs Esforço de Correção
 Repositórios: pandas-dev/pandas | scikit-learn/scikit-learn | numpy/numpy
 Período: 2020-2024
================================================================================
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import spearmanr, kruskal, mannwhitneyu

plt.rcParams.update({
    "figure.dpi": 150, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.titlesize": 11, "axes.labelsize": 10,
})

REPOS = {
    "pandas":  "dataset_esforco_pandas_2020_2024.csv",
    "sklearn": "dataset_esforco_sklearn_2020_2024.csv",
    "numpy":   "dataset_esforco_numpy_2020_2024.csv",
}
CORES  = {"pandas": "#2a78d6", "sklearn": "#f97306", "numpy": "#1baf7a"}
METRICAS = ["T_resolucao", "T_latencia", "T_implementacao", "T_integracao"]
LABELS_M = {
    "T_resolucao":     "T_resolução (h)",
    "T_latencia":      "T_latência (h)",
    "T_implementacao": "T_implementação (h)",
    "T_integracao":    "T_integração (h)",
}
CC_BINS   = [0, 2, 5, 10, 9999]
CC_LABELS = ["Baixa\n(1-2)", "Média\n(3-5)", "Alta\n(6-10)", "Muito Alta\n(11+)"]

# ── Limpeza padrão ────────────────────────────────────────────────────────────
def limpar(df):
    df = df[df["T_latencia"] >= 0].copy()
    colunas = ["cc_antes","loc_antes","T_implementacao","T_resolucao","T_latencia","T_integracao"]
    mask = pd.Series([True]*len(df), index=df.index)
    for col in colunas:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask &= df[col].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    df = df[mask].copy()
    df["cc_faixa"] = pd.cut(df["cc_antes"], bins=CC_BINS, labels=CC_LABELS)
    return df

def agregar_pr(df):
    agg = df.groupby("pr_id").agg(
        cc_mean      = ("cc_antes",    "mean"),
        cc_max       = ("cc_antes",    "max"),
        loc_mean     = ("loc_antes",   "mean"),
        loc_max      = ("loc_antes",   "max"),
        n_funcoes    = ("funcao_nome", "count"),
        n_arquivos   = ("arquivo",     "nunique"),
        T_resolucao     = ("T_resolucao",     "first"),
        T_latencia      = ("T_latencia",      "first"),
        T_implementacao = ("T_implementacao", "first"),
        T_integracao    = ("T_integracao",    "first"),
    ).reset_index()
    return limpar(agg.rename(columns={"cc_mean":"cc_antes","loc_mean":"loc_antes"}))

# ── Carregar e limpar ─────────────────────────────────────────────────────────
print("=" * 70)
print("ANÁLISE COMPARATIVA — pandas | scikit-learn | numpy (2020-2024)")
print("=" * 70)

dados = {}
dados_pr = {}
for repo, arquivo in REPOS.items():
    df = pd.read_csv(arquivo)
    df_c = limpar(df)
    df_p = agregar_pr(df)
    dados[repo]    = df_c
    dados_pr[repo] = df_p
    print(f"\n{repo}:")
    print(f"  Bruto: {len(df):,} funções | {df['pr_id'].nunique():,} PRs")
    print(f"  Limpo (função): {len(df_c):,} funções | {df_c['pr_id'].nunique():,} PRs")
    print(f"  Limpo (PR):     {len(df_p):,} PRs")

# ── 1. Correlações de Spearman por repositório ───────────────────────────────
print("\n" + "─"*70)
print("[1] CORRELAÇÕES DE SPEARMAN — CC e LOC vs T_implementacao")
print("─"*70)

resultados = []
for repo, df in dados.items():
    for pred in ["cc_antes", "loc_antes"]:
        r, p = spearmanr(df[pred], df["T_implementacao"])
        sig  = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        resultados.append({"repo": repo, "preditor": pred,
                            "r": round(r,4), "p": round(p,4), "sig": sig,
                            "nivel": "função"})
        print(f"  [{repo:8s}] {pred:12s} × T_implementacao: r={r:+.4f}  p={p:.4f}  {sig}")

print()
for repo, df in dados_pr.items():
    for pred, orig in [("cc_antes","cc_mean"), ("loc_antes","loc_mean")]:
        r, p = spearmanr(df[pred], df["T_implementacao"])
        sig  = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        resultados.append({"repo": repo, "preditor": orig,
                            "r": round(r,4), "p": round(p,4), "sig": sig,
                            "nivel": "PR"})
        print(f"  [{repo:8s}] {orig:12s} × T_implementacao (PR): r={r:+.4f}  p={p:.4f}  {sig}")

df_res = pd.DataFrame(resultados)

# ── 2. Kruskal-Wallis por faixas de CC ───────────────────────────────────────
print("\n" + "─"*70)
print("[2] KRUSKAL-WALLIS — T_implementacao por faixa de CC")
print("─"*70)

kw_results = {}
for repo, df in dados.items():
    grupos = [df[df["cc_faixa"]==f]["T_implementacao"].values for f in CC_LABELS]
    grupos = [g for g in grupos if len(g) > 1]
    H, p = kruskal(*grupos)
    sig  = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
    kw_results[repo] = (H, p, sig)
    print(f"  [{repo:8s}] H={H:.3f}, p={p:.4f} {sig}")
    for f in CC_LABELS:
        sub = df[df["cc_faixa"]==f]["T_implementacao"]
        if len(sub): print(f"    {f.replace(chr(10),' '):20s}: mediana={sub.median():.1f}h (n={len(sub)})")

# ── 3. Estatísticas descritivas comparativas ──────────────────────────────────
print("\n" + "─"*70)
print("[3] DESCRITIVAS COMPARATIVAS")
print("─"*70)

desc_rows = []
for repo, df in dados.items():
    row = {"repo": repo, "n_funcoes": len(df), "n_prs": df["pr_id"].nunique()}
    for col in ["cc_antes","loc_antes","T_implementacao","T_resolucao","T_latencia","T_integracao"]:
        row[f"{col}_med"] = round(df[col].median(), 1)
        row[f"{col}_iqr"] = round(df[col].quantile(0.75) - df[col].quantile(0.25), 1)
    desc_rows.append(row)
df_desc = pd.DataFrame(desc_rows)
print(df_desc[["repo","n_funcoes","n_prs",
               "cc_antes_med","loc_antes_med",
               "T_implementacao_med","T_resolucao_med"]].to_string(index=False))

# ── 4. Figuras comparativas ───────────────────────────────────────────────────
print("\n[4] GERANDO FIGURAS...")

# Fig A: Heatmap de correlações lado a lado (função + PR)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, nivel, df_dict in zip(axes,
    ["Nível de Função", "Nível de PR"],
    [dados, dados_pr]
):
    mat = pd.DataFrame(index=list(REPOS.keys()), columns=["CC", "LOC"], dtype=float)
    pmat = pd.DataFrame(index=list(REPOS.keys()), columns=["CC", "LOC"], dtype=float)
    for repo, df in df_dict.items():
        for col_label, col in [("CC","cc_antes"),("LOC","loc_antes")]:
            r, p = spearmanr(df[col], df["T_implementacao"])
            mat.loc[repo, col_label]  = round(r, 3)
            pmat.loc[repo, col_label] = p
    sns.heatmap(mat.astype(float), annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, vmin=-0.3, vmax=0.3, ax=ax, linewidths=0.5)
    for i, repo in enumerate(REPOS.keys()):
        for j, col in enumerate(["CC","LOC"]):
            p = pmat.loc[repo, col]
            sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
            if sig:
                ax.text(j+0.5, i+0.8, sig, ha="center", fontsize=11)
    ax.set_title(f"Spearman r vs T_implementação\n{nivel}")
    ax.set_xlabel("")
fig.suptitle("Correlações CC/LOC × T_implementação — Comparativo entre repositórios\n(* p<0,05  ** p<0,01  *** p<0,001)", fontsize=11)
plt.tight_layout()
plt.savefig("figA_heatmap_comparativo.png", bbox_inches="tight")
plt.close()
print("  figA_heatmap_comparativo.png ✓")

# Fig B: Boxplots T_implementacao por repositório
fig, ax = plt.subplots(figsize=(10, 5))
data_box = [dados[r]["T_implementacao"].values for r in REPOS]
bp = ax.boxplot(data_box, patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                flierprops=dict(marker="o", markersize=2, alpha=0.3))
for patch, repo in zip(bp["boxes"], REPOS):
    patch.set_facecolor(CORES[repo])
    patch.set_alpha(0.8)
ax.set_xticklabels([f"{r}\n(n={len(dados[r])})" for r in REPOS])
ax.set_ylabel("T_implementação (horas)")
ax.set_title("Distribuição de T_implementação por repositório")
for i, repo in enumerate(REPOS, 1):
    med = dados[repo]["T_implementacao"].median()
    ax.text(i, med+5, f"{med:.0f}h", ha="center", fontsize=9,
            fontweight="bold", color=CORES[repo])
plt.tight_layout()
plt.savefig("figB_boxplot_repos.png", bbox_inches="tight")
plt.close()
print("  figB_boxplot_repos.png ✓")

# Fig C: Boxplots T_implementacao por faixa de CC — um painel por repo
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle("T_implementação por faixa de CC — comparativo entre repositórios", fontsize=12)
for ax, (repo, df) in zip(axes, dados.items()):
    grupos = [df[df["cc_faixa"]==f]["T_implementacao"].values for f in CC_LABELS]
    bp = ax.boxplot(grupos, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    flierprops=dict(marker="o", markersize=2, alpha=0.3))
    for patch in bp["boxes"]:
        patch.set_facecolor(CORES[repo])
        patch.set_alpha(0.7)
    H, p, sig = kw_results[repo]
    ax.set_title(f"{repo}\nKruskal-Wallis: H={H:.2f}, p={p:.3f} {sig}")
    ax.set_xticklabels(CC_LABELS, fontsize=8)
    ax.set_ylabel("T_implementação (h)")
    for i, f in enumerate(CC_LABELS, 1):
        sub = df[df["cc_faixa"]==f]["T_implementacao"]
        if len(sub):
            ax.text(i, sub.median()+2, f"{sub.median():.0f}h",
                    ha="center", fontsize=8, color=CORES[repo])
plt.tight_layout()
plt.savefig("figC_boxplots_cc_repos.png", bbox_inches="tight")
plt.close()
print("  figC_boxplots_cc_repos.png ✓")

# Fig D: Scatter CC vs T_implementacao — três painéis
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle("CC vs T_implementação — comparativo entre repositórios", fontsize=12)
from scipy.stats import theilslopes
for ax, (repo, df) in zip(axes, dados.items()):
    ax.scatter(df["cc_antes"], df["T_implementacao"],
               alpha=0.15, s=8, color=CORES[repo])
    slope, intercept, _, _ = theilslopes(df["T_implementacao"], df["cc_antes"])
    x_r = np.linspace(df["cc_antes"].min(), df["cc_antes"].max(), 100)
    ax.plot(x_r, intercept + slope*x_r, color="red", linewidth=1.5,
            label=f"slope: {slope:.2f}")
    r, p = spearmanr(df["cc_antes"], df["T_implementacao"])
    sig  = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
    ax.set_title(f"{repo}\nSpearman r={r:.3f} {sig}")
    ax.set_xlabel("CC")
    ax.set_ylabel("T_implementação (h)")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figD_scatter_cc_repos.png", bbox_inches="tight")
plt.close()
print("  figD_scatter_cc_repos.png ✓")

# ── 5. Sumário final ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMÁRIO COMPARATIVO")
print("="*70)
print("\nCorrelações CC × T_implementacao (nível função):")
for repo in REPOS:
    sub = df_res[(df_res["repo"]==repo) & (df_res["preditor"]=="cc_antes") & (df_res["nivel"]=="função")]
    row = sub.iloc[0]
    print(f"  {repo:10s}: r={row['r']:+.4f}  p={row['p']:.4f}  {row['sig']}")

print("\nCorrelações CC × T_implementacao (nível PR):")
for repo in REPOS:
    sub = df_res[(df_res["repo"]==repo) & (df_res["preditor"]=="cc_mean") & (df_res["nivel"]=="PR")]
    row = sub.iloc[0]
    print(f"  {repo:10s}: r={row['r']:+.4f}  p={row['p']:.4f}  {row['sig']}")

print("\nKruskal-Wallis (T_implementacao por faixa CC):")
for repo, (H, p, sig) in kw_results.items():
    print(f"  {repo:10s}: H={H:.3f}  p={p:.4f}  {sig}")

df_res.to_csv("resultados_correlacoes_comparativo.csv", index=False)
print("\nresultados_correlacoes_comparativo.csv ✓")
print("\nConcluído.")
