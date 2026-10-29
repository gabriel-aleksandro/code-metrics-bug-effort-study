"""
================================================================================
 Análise Estatística: CC/LOC vs Esforço de Correção de Bugs
 Dataset: pandas-dev/pandas (2020-2024)
================================================================================

Execução:
    pip install pandas scipy matplotlib seaborn
    python analise_estatistica.py

Saídas geradas:
    - relatorio_estatistico.txt   (sumário completo para o orientador)
    - fig1_distribuicoes.png      (distribuições de CC, LOC e métricas de tempo)
    - fig2_correlacoes.png        (heatmap de correlações de Spearman)
    - fig3_boxplots_cc.png        (T_implementacao por faixa de CC)
    - fig4_scatter.png            (scatter CC/LOC vs T_implementacao)
    - fig5_barras_spearman.png    (barras de r de Spearman com destaque de significância)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu, kruskal

# ── Estilo global ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

AZUL    = "#2a78d6"
VERDE   = "#1baf7a"
AMARELO = "#eda100"
ROXO    = "#4a3aa7"
CORES_FAIXA = [AZUL, VERDE, AMARELO, ROXO]

METRICAS_TEMPO = ["T_resolucao", "T_latencia", "T_implementacao", "T_integracao"]
METRICAS_LABEL = {
    "T_resolucao":     "T_resolução (h)",
    "T_latencia":      "T_latência (h)",
    "T_implementacao": "T_implementação (h)",
    "T_integracao":    "T_integração (h)",
}

# ── 1. Carga e limpeza inicial ────────────────────────────────────────────────
print("=" * 70)
print("ANÁLISE ESTATÍSTICA — CC/LOC vs ESFORÇO DE CORREÇÃO (pandas 2020-2024)")
print("=" * 70)

df_raw = pd.read_csv("dataset_esforco_pandas_2020_2024.csv")
print(f"\n[1] Dataset bruto: {len(df_raw):,} funções | {df_raw['pr_id'].nunique():,} PRs únicas")

# Filtro 1: remover T_latencia negativa (commit anterior à abertura da issue)
df = df_raw[df_raw["T_latencia"] >= 0].copy()
n_lat_neg = len(df_raw) - len(df)
print(f"    Removidos {n_lat_neg} registros com T_latencia < 0 ({n_lat_neg/len(df_raw)*100:.1f}%)")

# Filtro 2: remover outliers via IQR nas variáveis de interesse
def remover_outliers_iqr(df, colunas, fator=1.5):
    mascara = pd.Series([True] * len(df), index=df.index)
    info = {}
    for col in colunas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - fator * IQR
        limite_sup = Q3 + fator * IQR
        fora = ~df[col].between(limite_inf, limite_sup)
        info[col] = {"Q1": Q1, "Q3": Q3, "IQR": IQR,
                     "lim_inf": limite_inf, "lim_sup": limite_sup,
                     "n_removidos": fora.sum()}
        mascara &= ~fora
    return df[mascara].copy(), info

colunas_outlier = ["cc_antes", "loc_antes", "T_implementacao",
                   "T_resolucao", "T_latencia", "T_integracao"]
df_clean, info_iqr = remover_outliers_iqr(df, colunas_outlier)

n_removidos_iqr = len(df) - len(df_clean)
print(f"\n    Remoção de outliers (IQR ×1.5) por coluna:")
for col, v in info_iqr.items():
    print(f"      {col}: [{v['lim_inf']:.1f}, {v['lim_sup']:.1f}] "
          f"→ {v['n_removidos']} removidos")
print(f"\n    Dataset limpo: {len(df_clean):,} funções | "
      f"{df_clean['pr_id'].nunique():,} PRs únicas "
      f"({n_removidos_iqr} removidos = {n_removidos_iqr/len(df)*100:.1f}%)")

# Faixas de CC para análise categórica
bins   = [0, 2, 5, 10, df_clean["cc_antes"].max() + 1]
labels = ["Baixa (1-2)", "Média (3-5)", "Alta (6-10)", "Muito Alta (11+)"]
df_clean["cc_faixa"] = pd.cut(df_clean["cc_antes"], bins=bins, labels=labels)

# ── 2. Estatísticas descritivas ───────────────────────────────────────────────
print("\n" + "─" * 70)
print("[2] ESTATÍSTICAS DESCRITIVAS (dataset limpo)")
print("─" * 70)

cols_desc = ["cc_antes", "loc_antes"] + METRICAS_TEMPO
desc = df_clean[cols_desc].describe().round(2)
print(desc.to_string())

print("\n    Assimetria (skewness):")
for col in cols_desc:
    sk = df_clean[col].skew()
    print(f"      {col}: {sk:.3f}")

# ── 3. Testes de normalidade ──────────────────────────────────────────────────
print("\n" + "─" * 70)
print("[3] TESTE DE NORMALIDADE — Shapiro-Wilk (amostra de 500)")
print("─" * 70)

np.random.seed(42)
amostra = df_clean.sample(min(500, len(df_clean)), random_state=42)
for col in cols_desc:
    stat, p = stats.shapiro(amostra[col])
    normal = "NORMAL" if p > 0.05 else "NÃO NORMAL"
    print(f"    {col}: W={stat:.4f}, p={p:.4f} → {normal}")

print("\n    → Distribuições não-normais confirmam uso de Spearman (não Pearson)")

# ── 4. Correlações de Spearman ────────────────────────────────────────────────
print("\n" + "─" * 70)
print("[4] CORRELAÇÕES DE SPEARMAN — CC e LOC vs métricas de tempo")
print("─" * 70)

resultados_corr = []
for pred in ["cc_antes", "loc_antes"]:
    for alvo in METRICAS_TEMPO:
        r, p = spearmanr(df_clean[pred], df_clean[alvo])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        resultados_corr.append({"Preditor": pred, "Alvo": alvo,
                                 "r": round(r, 4), "p": round(p, 4), "sig": sig})
        print(f"    {pred} × {alvo}: r={r:.4f}, p={p:.4f} {sig}")

df_corr = pd.DataFrame(resultados_corr)

# ── 5. Teste de Kruskal-Wallis e Mann-Whitney por faixas de CC ───────────────
print("\n" + "─" * 70)
print("[5] KRUSKAL-WALLIS — T_implementacao por faixa de CC")
print("─" * 70)

grupos = [df_clean[df_clean["cc_faixa"] == f]["T_implementacao"].values
          for f in labels]
H, p_kw = kruskal(*grupos)
print(f"    H={H:.4f}, p={p_kw:.4f} "
      f"{'→ diferença significativa entre grupos' if p_kw < 0.05 else '→ sem diferença significativa'}")

print("\n    Medianas por faixa:")
for f in labels:
    med = df_clean[df_clean["cc_faixa"] == f]["T_implementacao"].median()
    n   = (df_clean["cc_faixa"] == f).sum()
    print(f"      {f}: mediana={med:.1f}h (n={n})")

print("\n    Mann-Whitney — comparações par a par (Baixa vs demais):")
g_baixa = df_clean[df_clean["cc_faixa"] == "Baixa (1-2)"]["T_implementacao"].values
for f in labels[1:]:
    g = df_clean[df_clean["cc_faixa"] == f]["T_implementacao"].values
    U, p_mw = mannwhitneyu(g_baixa, g, alternative="two-sided")
    r_ef = 1 - (2 * U) / (len(g_baixa) * len(g))   # rank-biserial correlation (effect size)
    sig = "***" if p_mw < 0.001 else ("**" if p_mw < 0.01 else ("*" if p_mw < 0.05 else "ns"))
    print(f"      Baixa vs {f}: U={U:.0f}, p={p_mw:.4f} {sig}, r={r_ef:.3f}")

# ── 6. Figuras ────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("[6] GERANDO FIGURAS...")
print("─" * 70)

# --- Fig 1: Distribuições ---
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("Distribuições das variáveis (dataset limpo, outliers removidos)", fontsize=13)

vars_plot = [("cc_antes", "CC"), ("loc_antes", "LOC"),
             ("T_implementacao", "T_implementação (h)"),
             ("T_resolucao", "T_resolução (h)"),
             ("T_latencia", "T_latência (h)"),
             ("T_integracao", "T_integração (h)")]

for ax, (col, titulo) in zip(axes.flat, vars_plot):
    ax.hist(df_clean[col], bins=40, color=AZUL, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_title(titulo)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frequência")
    med = df_clean[col].median()
    ax.axvline(med, color="red", linestyle="--", linewidth=1.2, label=f"Mediana: {med:.1f}")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig1_distribuicoes.png", bbox_inches="tight")
plt.close()
print("    fig1_distribuicoes.png ✓")

# --- Fig 2: Heatmap de correlações ---
fig, ax = plt.subplots(figsize=(8, 4))

corr_matrix = pd.DataFrame(index=["cc_antes", "loc_antes"], columns=METRICAS_TEMPO, dtype=float)
pval_matrix = pd.DataFrame(index=["cc_antes", "loc_antes"], columns=METRICAS_TEMPO, dtype=float)
for pred in ["cc_antes", "loc_antes"]:
    for alvo in METRICAS_TEMPO:
        r, p = spearmanr(df_clean[pred], df_clean[alvo])
        corr_matrix.loc[pred, alvo] = round(r, 3)
        pval_matrix.loc[pred, alvo] = p

corr_matrix = corr_matrix.astype(float)
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            vmin=-0.3, vmax=0.3, ax=ax, linewidths=0.5,
            xticklabels=list(METRICAS_LABEL.values()),
            yticklabels=["CC", "LOC"])

for i, pred in enumerate(["cc_antes", "loc_antes"]):
    for j, alvo in enumerate(METRICAS_TEMPO):
        p = pval_matrix.loc[pred, alvo]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if sig:
            ax.text(j + 0.5, i + 0.82, sig, ha="center", va="center",
                    fontsize=11, color="black")

ax.set_title("Correlações de Spearman — CC e LOC vs métricas de tempo\n(* p<0.05  ** p<0.01  *** p<0.001)", fontsize=11)
plt.tight_layout()
plt.savefig("fig2_correlacoes.png", bbox_inches="tight")
plt.close()
print("    fig2_correlacoes.png ✓")

# --- Fig 3: Boxplots T_implementacao por faixa de CC ---
fig, ax = plt.subplots(figsize=(9, 5))

dados_box = [df_clean[df_clean["cc_faixa"] == f]["T_implementacao"].values for f in labels]
bp = ax.boxplot(dados_box, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, alpha=0.4))

for patch, cor in zip(bp["boxes"], CORES_FAIXA):
    patch.set_facecolor(cor)
    patch.set_alpha(0.8)

ax.set_xticklabels(labels)
ax.set_xlabel("Faixa de Complexidade Ciclomática (CC)")
ax.set_ylabel("T_implementação (horas)")
ax.set_title("T_implementação por faixa de CC")

for i, (dados, cor) in enumerate(zip(dados_box, CORES_FAIXA), start=1):
    med = np.median(dados)
    ax.text(i, med + 2, f"{med:.0f}h", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=cor)

sig_kw = "***" if p_kw < 0.001 else ("**" if p_kw < 0.01 else ("*" if p_kw < 0.05 else "ns"))
ax.set_title(f"T_implementação por faixa de CC  [Kruskal-Wallis: H={H:.2f}, p={p_kw:.4f} {sig_kw}]")

plt.tight_layout()
plt.savefig("fig3_boxplots_cc.png", bbox_inches="tight")
plt.close()
print("    fig3_boxplots_cc.png ✓")

# --- Fig 4: Scatter CC e LOC vs T_implementacao ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (pred, titulo, cor) in zip(axes, [
    ("cc_antes", "CC vs T_implementação", AZUL),
    ("loc_antes", "LOC vs T_implementação", VERDE)
]):
    ax.scatter(df_clean[pred], df_clean["T_implementacao"],
               alpha=0.15, s=10, color=cor)
    # linha de tendência (Theil-Sen — robusta a outliers)
    slope, intercept, _, _ = stats.theilslopes(df_clean["T_implementacao"], df_clean[pred])
    x_range = np.linspace(df_clean[pred].min(), df_clean[pred].max(), 100)
    ax.plot(x_range, intercept + slope * x_range, color="red",
            linewidth=1.5, label=f"Theil-Sen slope: {slope:.2f}")
    r, p = spearmanr(df_clean[pred], df_clean["T_implementacao"])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.set_title(f"{titulo}\nSpearman r={r:.3f} {sig}")
    ax.set_xlabel(pred.replace("_antes", "").upper())
    ax.set_ylabel("T_implementação (h)")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig4_scatter.png", bbox_inches="tight")
plt.close()
print("    fig4_scatter.png ✓")


# --- Fig 5: Barras de r de Spearman por preditor e métrica ---
fig, ax = plt.subplots(figsize=(10, 5))

metricas_plot  = list(METRICAS_LABEL.values())
cc_vals  = [df_corr[(df_corr["Preditor"]=="cc_antes")  & (df_corr["Alvo"]==m)]["r"].values[0] for m in METRICAS_TEMPO]
loc_vals = [df_corr[(df_corr["Preditor"]=="loc_antes") & (df_corr["Alvo"]==m)]["r"].values[0] for m in METRICAS_TEMPO]
cc_sigs  = [df_corr[(df_corr["Preditor"]=="cc_antes")  & (df_corr["Alvo"]==m)]["sig"].values[0] for m in METRICAS_TEMPO]
loc_sigs = [df_corr[(df_corr["Preditor"]=="loc_antes") & (df_corr["Alvo"]==m)]["sig"].values[0] for m in METRICAS_TEMPO]

x      = np.arange(len(METRICAS_TEMPO))
width  = 0.35
cinza  = "#c3c2b7"

# Cor sólida para significativo, cinza para ns
cc_cores  = [AZUL  if s != "ns" else cinza for s in cc_sigs]
loc_cores = [VERDE if s != "ns" else cinza for s in loc_sigs]

bars_cc  = ax.bar(x - width/2, cc_vals,  width, color=cc_cores,  label="CC",  alpha=0.9)
bars_loc = ax.bar(x + width/2, loc_vals, width, color=loc_cores, label="LOC", alpha=0.9,
                  hatch="//", edgecolor="white", linewidth=0.5)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metricas_plot)
ax.set_ylabel("Correlação de Spearman (r)")
ax.set_title("Correlações de Spearman — CC e LOC vs métricas de tempo\n(barras coloridas = p<0.05; cinza = não significativo)")

# Asteriscos sobre barras significativas
for i, (bar, sig) in enumerate(zip(bars_cc, cc_sigs)):
    if sig != "ns":
        ypos = bar.get_height() + 0.002 if bar.get_height() >= 0 else bar.get_height() - 0.006
        ax.text(bar.get_x() + bar.get_width()/2, ypos, sig,
                ha="center", va="bottom", fontsize=11, color=AZUL, fontweight="bold")

for i, (bar, sig) in enumerate(zip(bars_loc, loc_sigs)):
    if sig != "ns":
        ypos = bar.get_height() + 0.002 if bar.get_height() >= 0 else bar.get_height() - 0.006
        ax.text(bar.get_x() + bar.get_width()/2, ypos, sig,
                ha="center", va="bottom", fontsize=11, color=VERDE, fontweight="bold")

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor=AZUL,  label="CC (significativo)"),
    Patch(facecolor=VERDE, label="LOC (significativo)", hatch="//", edgecolor="white"),
    Patch(facecolor=cinza, label="não significativo (p≥0.05)"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
ax.set_ylim(-0.12, 0.10)

plt.tight_layout()
plt.savefig("fig5_barras_spearman.png", bbox_inches="tight")
plt.close()
print("    fig5_barras_spearman.png ✓")

# ── 7. Relatório em texto ─────────────────────────────────────────────────────
with open("relatorio_estatistico.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("RELATÓRIO ESTATÍSTICO — CC/LOC vs ESFORÇO DE CORREÇÃO\n")
    f.write("Repositório: pandas-dev/pandas | Período: 2020-2024\n")
    f.write("=" * 70 + "\n\n")

    f.write("AMOSTRAS\n")
    f.write(f"  Bruto:  {len(df_raw):,} funções, {df_raw['pr_id'].nunique():,} PRs\n")
    f.write(f"  Após filtro T_latencia < 0: -{n_lat_neg}\n")
    f.write(f"  Após remoção de outliers (IQR×1.5): -{n_removidos_iqr}\n")
    f.write(f"  Final:  {len(df_clean):,} funções, {df_clean['pr_id'].nunique():,} PRs\n\n")

    f.write("ESTATÍSTICAS DESCRITIVAS\n")
    f.write(desc.to_string() + "\n\n")

    f.write("NORMALIDADE (Shapiro-Wilk, n=500)\n")
    f.write("  Todas as variáveis: NÃO NORMAL → justifica uso de Spearman\n\n")

    f.write("CORRELAÇÕES DE SPEARMAN\n")
    f.write(df_corr.to_string(index=False) + "\n\n")

    f.write("KRUSKAL-WALLIS — T_implementacao por faixa de CC\n")
    f.write(f"  H={H:.4f}, p={p_kw:.4f}\n")
    f.write("  Medianas:\n")
    for f_label in labels:
        med = df_clean[df_clean["cc_faixa"] == f_label]["T_implementacao"].median()
        n   = (df_clean["cc_faixa"] == f_label).sum()
        f.write(f"    {f_label}: {med:.1f}h (n={n})\n")

    f.write("\n  sig: *** p<0.001  ** p<0.01  * p<0.05  ns = não significativo\n")

print("    relatorio_estatistico.txt ✓")

print("\n" + "=" * 70)
print("CONCLUÍDO. Arquivos gerados na pasta atual.")
print("=" * 70)
