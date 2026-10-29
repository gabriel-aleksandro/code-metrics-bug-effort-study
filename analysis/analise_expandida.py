"""
================================================================================
 Análise Estatística Expandida: CC/LOC vs Esforço de Correção
 Dataset: pandas-dev/pandas (2020-2024)
 Novidades em relação à análise anterior:
   1. Análise em nível de PR (além de nível de função)
   2. Variáveis de controle: n_funcoes, n_arquivos alterados
   3. Correlação parcial de Spearman (controlando por n_funcoes e n_arquivos)
   4. Regressão quantílica (mediana) para T_implementacao
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, kruskal, mannwhitneyu

# ── Estilo global ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})
AZUL  = "#2a78d6"; VERDE = "#1baf7a"; AMARELO = "#eda100"; ROXO = "#4a3aa7"
CORES_FAIXA = [AZUL, VERDE, AMARELO, ROXO]
METRICAS = ["T_resolucao", "T_latencia", "T_implementacao", "T_integracao"]
LABELS_M  = {"T_resolucao": "T_resolução (h)", "T_latencia": "T_latência (h)",
              "T_implementacao": "T_implementação (h)", "T_integracao": "T_integração (h)"}

# ── 1. Carga e limpeza ────────────────────────────────────────────────────────
print("=" * 70)
print("ANÁLISE EXPANDIDA — pandas-dev/pandas 2020-2024")
print("=" * 70)

df_raw = pd.read_csv("dataset_esforco_pandas_2020_2024.csv")
df = df_raw[df_raw["T_latencia"] >= 0].copy()

def remover_outliers_iqr(df, colunas, fator=1.5):
    mask = pd.Series([True]*len(df), index=df.index)
    for col in colunas:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask &= df[col].between(Q1 - fator*IQR, Q3 + fator*IQR)
    return df[mask].copy()

df_func = remover_outliers_iqr(df,
    ["cc_antes","loc_antes","T_implementacao","T_resolucao","T_latencia","T_integracao"])

bins   = [0, 2, 5, 10, df_func["cc_antes"].max()+1]
labels = ["Baixa (1-2)", "Média (3-5)", "Alta (6-10)", "Muito Alta (11+)"]
df_func["cc_faixa"] = pd.cut(df_func["cc_antes"], bins=bins, labels=labels)

print(f"\n[dataset função] {len(df_func):,} funções | {df_func['pr_id'].nunique():,} PRs")

# ── 2. Agregação em nível de PR ───────────────────────────────────────────────
df_pr = df_func.groupby("pr_id").agg(
    cc_mean      = ("cc_antes",    "mean"),
    cc_max       = ("cc_antes",    "max"),
    cc_median    = ("cc_antes",    "median"),
    loc_mean     = ("loc_antes",   "mean"),
    loc_max      = ("loc_antes",   "max"),
    n_funcoes    = ("funcao_nome", "count"),
    n_arquivos   = ("arquivo",     "nunique"),
    T_resolucao     = ("T_resolucao",     "first"),
    T_latencia      = ("T_latencia",      "first"),
    T_implementacao = ("T_implementacao", "first"),
    T_integracao    = ("T_integracao",    "first"),
).reset_index()

# Remove outliers também no nível de PR
df_pr = remover_outliers_iqr(df_pr,
    ["cc_mean","cc_max","loc_mean","loc_max","n_funcoes","n_arquivos",
     "T_implementacao","T_resolucao","T_latencia","T_integracao"])

print(f"[dataset PR]     {len(df_pr):,} PRs únicas após limpeza")

# ── 3. Correlações em nível de PR ─────────────────────────────────────────────
print("\n" + "─"*70)
print("[3] CORRELAÇÕES DE SPEARMAN — NÍVEL DE PR")
print("─"*70)

preditores_pr = ["cc_mean", "cc_max", "loc_mean", "loc_max", "n_funcoes", "n_arquivos"]
resultados_pr = []
for pred in preditores_pr:
    for alvo in METRICAS:
        r, p = spearmanr(df_pr[pred], df_pr[alvo])
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        resultados_pr.append({"Preditor": pred, "Alvo": alvo,
                               "r": round(r,4), "p": round(p,4), "sig": sig})
        print(f"  {pred:15s} × {alvo:20s}: r={r:+.4f}  p={p:.4f}  {sig}")

df_corr_pr = pd.DataFrame(resultados_pr)

# ── 4. Correlação parcial de Spearman ─────────────────────────────────────────
print("\n" + "─"*70)
print("[4] CORRELAÇÃO PARCIAL DE SPEARMAN")
print("    Controlando por: n_funcoes, n_arquivos")
print("─"*70)

def spearman_parcial(df, x, y, controles):
    """
    Correlação parcial de Spearman via resíduos de regressão de ranks.
    Regride os ranks de x e y nos ranks dos controles; correlaciona os resíduos.
    """
    from numpy.linalg import lstsq
    cols = [x, y] + controles
    sub  = df[cols].dropna()
    ranks = sub.rank()

    def residuos(target):
        X = np.column_stack([ranks[c] for c in controles] + [np.ones(len(ranks))])
        y_vec = ranks[target].values
        coef, *_ = lstsq(X, y_vec, rcond=None)
        return y_vec - X @ coef

    res_x = residuos(x)
    res_y = residuos(y)
    r, p  = spearmanr(res_x, res_y)
    return round(r, 4), round(p, 4)

controles = ["n_funcoes", "n_arquivos"]
print(f"\n  Preditor          × Alvo                   r_parcial   p")
for pred in ["cc_mean", "cc_max", "loc_mean", "loc_max"]:
    for alvo in METRICAS:
        r, p = spearman_parcial(df_pr, pred, alvo, controles)
        sig  = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        print(f"  {pred:17s} × {alvo:20s}  r={r:+.4f}   p={p:.4f}  {sig}")

# ── 5. Regressão quantílica (mediana) ─────────────────────────────────────────
print("\n" + "─"*70)
print("[5] REGRESSÃO QUANTÍLICA (τ=0.5) — T_implementacao ~ CC + LOC + controles")
print("─"*70)

try:
    import statsmodels.formula.api as smf
    formula = "T_implementacao ~ cc_mean + loc_mean + n_funcoes + n_arquivos"
    mod = smf.quantreg(formula, df_pr).fit(q=0.5, max_iter=2000)
    print(mod.summary())
    coef_qr = mod.params
    pval_qr = mod.pvalues
except ImportError:
    print("  statsmodels não instalado. Pulando regressão quantílica.")
    coef_qr = None; pval_qr = None

# ── 6. Figuras expandidas ─────────────────────────────────────────────────────
print("\n[6] GERANDO FIGURAS EXPANDIDAS...")

# Fig 5: Heatmap de correlações em nível de PR
fig, ax = plt.subplots(figsize=(10, 4))
pred_plot = ["cc_mean", "cc_max", "loc_mean", "loc_max", "n_funcoes", "n_arquivos"]
labels_pred = ["CC médio", "CC máx", "LOC médio", "LOC máx", "N funções", "N arquivos"]
corr_mat = pd.DataFrame(index=pred_plot, columns=METRICAS, dtype=float)
pval_mat = pd.DataFrame(index=pred_plot, columns=METRICAS, dtype=float)
for pred in pred_plot:
    for alvo in METRICAS:
        r, p = spearmanr(df_pr[pred], df_pr[alvo])
        corr_mat.loc[pred, alvo] = round(r, 3)
        pval_mat.loc[pred, alvo] = p

sns.heatmap(corr_mat.astype(float), annot=True, fmt=".3f", cmap="RdBu_r",
            center=0, vmin=-0.4, vmax=0.4, ax=ax, linewidths=0.5,
            xticklabels=list(LABELS_M.values()), yticklabels=labels_pred)

for i, pred in enumerate(pred_plot):
    for j, alvo in enumerate(METRICAS):
        p = pval_mat.loc[pred, alvo]
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        if sig:
            ax.text(j+0.5, i+0.82, sig, ha="center", va="center", fontsize=10)

ax.set_title("Correlações de Spearman — Nível de PR\n(* p<0,05  ** p<0,01  *** p<0,001)", fontsize=11)
plt.tight_layout()
plt.savefig("fig5_correlacoes_pr.png", bbox_inches="tight")
plt.close()
print("  fig5_correlacoes_pr.png ✓")

# Fig 6: n_funcoes e n_arquivos vs T_implementacao
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (pred, titulo, cor) in zip(axes, [
    ("n_funcoes",  "N° de funções alteradas vs T_implementação", AZUL),
    ("n_arquivos", "N° de arquivos alterados vs T_implementação", VERDE),
]):
    ax.scatter(df_pr[pred], df_pr["T_implementacao"], alpha=0.2, s=12, color=cor)
    slope, intercept, _, _ = stats.theilslopes(df_pr["T_implementacao"], df_pr[pred])
    x_r = np.linspace(df_pr[pred].min(), df_pr[pred].max(), 100)
    ax.plot(x_r, intercept + slope*x_r, color="red", linewidth=1.5,
            label=f"Theil-Sen slope: {slope:.2f}")
    r, p = spearmanr(df_pr[pred], df_pr["T_implementacao"])
    sig  = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
    ax.set_title(f"{titulo}\nSpearman r={r:.3f} {sig}")
    ax.set_xlabel(titulo.split(" vs ")[0])
    ax.set_ylabel("T_implementação (h)")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("fig6_controles_pr.png", bbox_inches="tight")
plt.close()
print("  fig6_controles_pr.png ✓")

# Fig 7: Comparativo função vs PR — scatter CC vs T_implementacao
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("CC vs T_implementação — Nível de Função vs Nível de PR", fontsize=12)
for ax, (df_plot, xlabel, titulo, cor) in zip(axes, [
    (df_func, "cc_antes", "Nível de Função (n=2.404)", AZUL),
    (df_pr,   "cc_mean", "Nível de PR (n=844)",        ROXO),
]):
    ax.scatter(df_plot[xlabel], df_plot["T_implementacao"],
               alpha=0.15, s=10, color=cor)
    slope, intercept, _, _ = stats.theilslopes(
        df_plot["T_implementacao"], df_plot[xlabel])
    x_r = np.linspace(df_plot[xlabel].min(), df_plot[xlabel].max(), 100)
    ax.plot(x_r, intercept + slope*x_r, color="red", linewidth=1.5,
            label=f"slope: {slope:.2f}")
    r, p = spearmanr(df_plot[xlabel], df_plot["T_implementacao"])
    sig  = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
    ax.set_title(f"{titulo}\nSpearman r={r:.3f} {sig}")
    ax.set_xlabel("CC" if xlabel == "cc_antes" else "CC médio da PR")
    ax.set_ylabel("T_implementação (h)")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("fig7_funcao_vs_pr.png", bbox_inches="tight")
plt.close()
print("  fig7_funcao_vs_pr.png ✓")

# ── 7. Salva datasets agregados ───────────────────────────────────────────────
df_pr.to_csv("dataset_nivel_pr_pandas.csv", index=False)
print("\n  dataset_nivel_pr_pandas.csv ✓")

# ── 8. Sumário ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMÁRIO DOS RESULTADOS EXPANDIDOS")
print("="*70)
print(f"\nNível de PR ({len(df_pr)} PRs):")
sig_pr = df_corr_pr[df_corr_pr["sig"] != "ns"]
if len(sig_pr):
    for _, row in sig_pr.iterrows():
        print(f"  {row['Preditor']:15s} × {row['Alvo']:20s}: r={row['r']:+.4f} {row['sig']}")
else:
    print("  Nenhuma correlação significativa encontrada.")

print("\nConcluído. Arquivos gerados na pasta atual.")
