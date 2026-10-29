# Estimando o Esforço para Correção de Defeitos a partir de Métricas Estáticas de Código

Estudo empírico em repositórios Python open-source — Iniciação Científica UNICAMP (2024–2026)

**Autor:** Gabriel Aleksandro de Paula  
**Orientadores:** Prof. Dr. Plinio Roberto Souza Vilela · Prof. Dr. Pedro Ivo Garcia-Nunes  
**Instituição:** Faculdade de Tecnologia — UNICAMP, Limeira, SP

---

## Sobre o projeto

Este repositório contém os artefatos completos da IC que investigou empiricamente se a **Complexidade Ciclomática (CC)** e as **Linhas de Código (LOC)** de funções Python predizem o esforço necessário para corrigir bugs em projetos open-source de grande escala.

**Repositórios analisados:** pandas-dev/pandas · scikit-learn/scikit-learn · numpy/numpy  
**Período:** 2020–2024  
**Dataset:** 3.402 funções modificadas em 1.171 Pull Requests de correção de bug

### Resultado principal

A hipótese foi refutada: CC e LOC não são preditores robustos do esforço de correção. As correlações de Spearman com o tempo de implementação foram fracas e inconsistentes entre repositórios (pandas: r = −0,058\*\*; scikit-learn: r = +0,106\*; numpy: r = +0,081 ns). O efeito desaparece ao nível de Pull Request e não persiste quando controlado pelo escopo da PR.

---

## Estrutura do repositório

```
.
├── scripts/
│   └── miner.py                     # Pipeline de mineração via GitHub API (REST + GraphQL)
│
├── analysis/         # Scripts de análise estatística
│   ├── analise_estatistica.py   # Análise nível função: Spearman, KW, correlação parcial (pandas)
│   ├── analise_expandida.py     # Análise nível PR + variáveis de controle + regressão quantílica
│   └── analise_comparativa.py   # Análise comparativa dos 3 repositórios
│
├── data/             # Datasets coletados (CSV)
│   ├── dataset_esforco_pandas_2020_2024.csv
│   ├── dataset_esforco_sklearn_2020_2024.csv
│   └── dataset_esforco_numpy_2020_2024.csv
│
├── figures/          # Figuras geradas pela análise
│   ├── fig1_distribuicoes.png       # Distribuições de CC, LOC e métricas de tempo (pandas)
│   ├── fig2_correlacoes.png         # Heatmap de correlações de Spearman (pandas)
│   ├── fig3_boxplots_cc.png         # T_implementação por faixa de CC (pandas)
│   ├── fig4_scatter.png             # Scatter CC/LOC vs T_implementação (pandas)
│   ├── fig5_correlacoes_pr.png      # Correlações de Spearman nível PR (pandas)
│   ├── fig6_controles_pr.png        # Correlação parcial controlando escopo da PR
│   ├── fig7_funcao_vs_pr.png        # Comparação nível função vs nível PR
│   ├── figA_heatmap_comparativo.png # Correlações comparativas — 3 repositórios
│   ├── figB_boxplot_repos.png       # T_implementação por repositório
│   ├── figC_boxplots_cc_repos.png   # CC por faixa — 3 repositórios
│   └── figD_scatter_cc_repos.png    # Scatter comparativo — 3 repositórios
│
└── paper/
    └── artigo_esforco_bugs.tex      # Artigo científico completo (LaTeX)
```

---

## Metodologia

### Métricas de esforço (proxies de tempo)

| Métrica | Fórmula | Interpretação |
|---|---|---|
| T_resolucao | issue.closed_at − issue.created_at | Vida útil do bug no ecossistema |
| T_latencia | commit₁ − issue.created_at | Latência até o início da implementação |
| **T_implementacao** | commitₙ − commit₁ | **Janela de programação ativa (foco principal)** |
| T_integracao | PR.merged_at − commitₙ | Tempo de code review e CI/CD |

### Pipeline de mineração

O pipeline opera 100% via GitHub API (REST + GraphQL), sem clone local dos repositórios:

1. Busca issues de bug fechadas via GraphQL (labels: `Bug` / `00 - Bug`)
2. Vincula cada issue ao PR que a fechou via `ClosedEvent.closer` na timeline
3. Coleta commits originais do PR via `/pulls/{n}/commits` — contornando o squash-and-merge do pandas, que zeraria o T_implementacao se consultado pelo histórico da main
4. Recupera o código pré-fix via `/contents/{path}?ref={sha}~1`
5. Calcula CC com `radon` e LOC por contagem de linhas não-vazias via AST

### Análise estatística

- Teste de normalidade: Shapiro-Wilk (confirma distribuição não-normal em todas as variáveis)
- Correlação: Spearman (nível função e nível PR)
- Comparação de grupos: Kruskal-Wallis + Mann-Whitney por faixas de CC
- Controle de confundidores: correlação parcial controlando escopo da PR

---

## Dataset

Cada linha do CSV representa uma **função/método modificado** em uma correção de bug:

| Coluna | Descrição |
|---|---|
| issue_id | Número da issue no GitHub |
| pr_id | Número do Pull Request |
| funcao_nome | Nome da função/método |
| cc_antes | Complexidade Ciclomática antes do fix |
| loc_antes | Linhas de código antes do fix |
| T_resolucao | Tempo total de resolução (horas) |
| T_latencia | Tempo até início da implementação (horas) |
| T_implementacao | Janela de implementação ativa (horas) |
| T_integracao | Tempo de revisão e integração (horas) |

**Tamanho do dataset (após limpeza):**

| Repositório | Funções | PRs |
|---|---|---|
| pandas | 2.638 | 889 |
| scikit-learn | 519 | 167 |
| numpy | 245 | 115 |
| **Total** | **3.402** | **1.171** |

Limpeza aplicada: remoção de T_latencia < 0 e outliers via Tukey IQR × 1,5.

---

## Como reproduzir

### Pré-requisitos

```bash
pip install requests pandas numpy scipy matplotlib seaborn radon
```

Defina o token de acesso à GitHub API:

```bash
export GITHUB_TOKEN=ghp_seu_token_aqui
```

### Mineração

```bash
# pandas
python scripts/miner.py --owner pandas-dev --repo pandas --label Bug \
                         --output data/dataset_esforco_pandas_2020_2024.csv

# scikit-learn
python scripts/miner.py --owner scikit-learn --repo scikit-learn --label Bug \
                         --output data/dataset_esforco_sklearn_2020_2024.csv

# numpy
python scripts/miner.py --owner numpy --repo numpy --label "00 - Bug" \
                         --output data/dataset_esforco_numpy_2020_2024.csv
```

O script possui checkpoint incremental: se interrompido, retoma de onde parou.

### Análise

```bash
# Análise nível função (figuras fig1–fig4 + relatorio_estatistico.txt)
python analysis/analise_estatistica.py

# Análise nível PR + variáveis de controle (figuras fig5–fig7)
python analysis/analise_expandida.py

# Análise comparativa dos 3 repositórios (figuras figA–figD)
python analysis/analise_comparativa.py
```

---

## Resultados

### Correlações de Spearman — CC × T_implementacao

| Repositório | Nível | r | p | Sig. |
|---|---|---|---|---|
| pandas | Função | −0,058 | 0,003 | \*\* |
| scikit-learn | Função | +0,106 | 0,016 | \* |
| numpy | Função | +0,081 | 0,208 | ns |
| pandas | PR | +0,020 | 0,599 | ns |
| scikit-learn | PR | +0,126 | 0,105 | ns |
| numpy | PR | +0,057 | 0,548 | ns |

### Kruskal-Wallis — T_implementacao por faixa de CC

| Repositório | H | p | Padrão |
|---|---|---|---|
| pandas | 10,845 | 0,013 | Não-monótono (CC≥11: 69h < CC baixa: 91h) |
| scikit-learn | 5,512 | 0,138 | ns |
| numpy | 3,260 | 0,353 | ns |

### Interpretações

As correlações fracas e inconsistentes sugerem três explicações alternativas, não mutuamente exclusivas:

1. **Viés de expertise:** funções complexas podem ser atribuídas a desenvolvedores mais experientes, que as corrigem mais rápido — confundidor não capturado pelas métricas estáticas
2. **Cultura do projeto:** T_implementacao mediana difere drasticamente entre repositórios (numpy: 0h, pandas: 91h, scikit-learn: 197h), sugerindo que práticas de contribuição dominam sobre a complexidade do código
3. **Limitação do proxy:** o intervalo entre commits não equivale ao tempo efetivo de programação

---

## Apresentação

Resultados apresentados no **XXXIV Congresso de Iniciação Científica — UNICAMP**, outubro de 2026.

---

## Licença

Repositório de pesquisa acadêmica. Dados coletados via GitHub API conforme os Termos de Serviço do GitHub.
