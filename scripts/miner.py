"""
================================================================================
 miner.py — Mineração de Complexidade Ciclomática / LOC vs Esforço de Correção
================================================================================

Coleta issues de bug fechadas de um repositório Python no GitHub, vincula cada
issue à PR mergeada que a fechou, extrai métricas de código (CC, LOC) via
GitHub API + radon (sem clone local), e calcula 4 métricas de esforço/tempo.

Uso:
    python scripts/miner.py --owner pandas-dev --repo pandas --label Bug \\
                            --output data/dataset_esforco_pandas_2020_2024.csv

    python scripts/miner.py --owner scikit-learn --repo scikit-learn --label Bug \\
                            --output data/dataset_esforco_sklearn_2020_2024.csv

    python scripts/miner.py --owner numpy --repo numpy --label "00 - Bug" \\
                            --output data/dataset_esforco_numpy_2020_2024.csv

Requisitos:
    pip install requests radon pandas

Variável de ambiente necessária:
    export GITHUB_TOKEN="ghp_xxx..."   # Linux/macOS
    $env:GITHUB_TOKEN="ghp_xxx..."    # Windows PowerShell

O script possui checkpoint incremental: se interrompido, retoma de onde parou.
"""

import os
import ast
import csv
import time
import base64
import argparse
from datetime import datetime, timezone

import requests
import pandas as pd
from radon.complexity import cc_visit
from radon.raw import analyze as radon_analyze


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Minera CC/LOC vs esforço de correção de bugs via GitHub API"
    )
    parser.add_argument("--owner",  required=True, help="Dono do repositório (ex: pandas-dev)")
    parser.add_argument("--repo",   required=True, help="Nome do repositório (ex: pandas)")
    parser.add_argument("--label",  required=True, help="Label de bug (ex: Bug  ou  '00 - Bug')")
    parser.add_argument("--output", required=True, help="Caminho do CSV de saída")
    parser.add_argument("--max-issues", type=int, default=2000,
                        help="Máximo de issues a processar (padrão: 2000)")
    parser.add_argument("--period-start", default="2020-01-01",
                        help="Início do período (YYYY-MM-DD, padrão: 2020-01-01)")
    parser.add_argument("--period-end",   default="2024-12-31",
                        help="Fim do período (YYYY-MM-DD, padrão: 2024-12-31)")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Configuração de rede
# --------------------------------------------------------------------------- #

def build_clients(owner, repo, label):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Defina a variável de ambiente GITHUB_TOKEN antes de executar.")

    repo_full   = f"{owner}/{repo}"
    graphql_url = "https://api.github.com/graphql"
    rest_base   = f"https://api.github.com/repos/{repo_full}"

    headers_graphql = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    headers_rest = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Query GraphQL com label injetada em tempo de configuração
    issues_query = """
    query($owner: String!, $name: String!, $after: String) {
      repository(owner: $owner, name: $name) {
        issues(first: 30, after: $after, states: CLOSED,
               labels: ["%s"], orderBy: {field: CREATED_AT, direction: ASC}) {
          pageInfo { hasNextPage endCursor }
          nodes {
            number
            createdAt
            closedAt
            timelineItems(last: 10, itemTypes: [CLOSED_EVENT]) {
              nodes {
                __typename
                ... on ClosedEvent {
                  closer {
                    __typename
                    ... on PullRequest {
                      number
                      merged
                      mergedAt
                      createdAt
                      mergeCommit { oid }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """ % label

    return {
        "owner":          owner,
        "repo":           repo,
        "graphql_url":    graphql_url,
        "rest_base":      rest_base,
        "headers_gql":    headers_graphql,
        "headers_rest":   headers_rest,
        "issues_query":   issues_query,
    }


# --------------------------------------------------------------------------- #
# Utilitários HTTP (retry automático em rate-limit)
# --------------------------------------------------------------------------- #

def github_get(cfg, url, params=None):
    while True:
        resp = requests.get(url, headers=cfg["headers_rest"], params=params)
        if resp.status_code in (401, 403, 429):
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait  = max(reset - time.time(), 30)
            print(f"  [rate-limit REST {resp.status_code}] aguardando {wait:.0f}s...")
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            return None
        if resp.status_code >= 500:
            print(f"  [server error {resp.status_code}] pulando: {url}")
            return None
        time.sleep(0.5)
        resp.raise_for_status()
        return resp


def run_graphql(cfg, query, variables):
    while True:
        resp = requests.post(
            cfg["graphql_url"],
            json={"query": query, "variables": variables},
            headers=cfg["headers_gql"],
        )
        if resp.status_code in (401, 403, 429):
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait  = max(reset - time.time(), 30)
            print(f"  [rate-limit GraphQL {resp.status_code}] aguardando {wait:.0f}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"Erro GraphQL: {data['errors']}")
        return data["data"]


# --------------------------------------------------------------------------- #
# 1+2. Issues de bug fechadas + PR mergeada vinculada
# --------------------------------------------------------------------------- #

def fetch_bug_issues_with_pr(cfg, period_start, period_end, max_issues):
    results, after, total = [], None, 0

    while True:
        data = run_graphql(cfg, cfg["issues_query"],
                           {"owner": cfg["owner"], "name": cfg["repo"], "after": after})
        issues_block = data["repository"]["issues"]

        for issue in issues_block["nodes"]:
            total += 1

            issue_dt = parse_iso(issue["createdAt"])
            if issue_dt < period_start:
                continue
            if issue_dt > period_end:
                print(f"  [período] Fim do intervalo em {issue['createdAt']}, encerrando.")
                return results

            linked_pr = None
            for event in issue["timelineItems"]["nodes"]:
                closer = event.get("closer")
                if closer and closer.get("__typename") == "PullRequest" and closer.get("merged"):
                    linked_pr = closer
                    break

            if not linked_pr:
                continue
            merge_commit = linked_pr.get("mergeCommit")
            if not merge_commit:
                continue

            results.append({
                "issue_number":      issue["number"],
                "issue_created_at":  issue["createdAt"],
                "issue_closed_at":   issue["closedAt"],
                "pr_number":         linked_pr["number"],
                "pr_created_at":     linked_pr["createdAt"],
                "pr_merged_at":      linked_pr["mergedAt"],
                "merge_commit_sha":  merge_commit["oid"],
            })

            if max_issues and len(results) >= max_issues:
                return results

        if not issues_block["pageInfo"]["hasNextPage"]:
            break
        after = issues_block["pageInfo"]["endCursor"]
        print(f"  [issues] {total} varridas, {len(results)} candidatas...")

    return results


# --------------------------------------------------------------------------- #
# 3. Commits originais do PR (evita armadilha do squash-and-merge)
# --------------------------------------------------------------------------- #

def get_pr_original_commits(cfg, pr_number):
    timestamps, page = [], 1
    while True:
        resp = github_get(cfg, f"{cfg['rest_base']}/pulls/{pr_number}/commits",
                          params={"per_page": 100, "page": page})
        if resp is None:
            break
        batch = resp.json()
        if not batch:
            break
        for c in batch:
            timestamps.append(c["commit"]["author"]["date"])
        if len(batch) < 100:
            break
        page += 1
    return timestamps


# --------------------------------------------------------------------------- #
# 4. Métricas de código via API do GitHub + radon (sem clone local)
# --------------------------------------------------------------------------- #

def get_file_content_before(cfg, filepath, merge_commit_sha):
    """Conteúdo do arquivo imediatamente ANTES do merge commit ({sha}~1)."""
    ref  = f"{merge_commit_sha}~1"
    resp = github_get(cfg, f"{cfg['rest_base']}/contents/{filepath}", params={"ref": ref})
    if resp is None:
        return None
    data = resp.json()
    if data.get("encoding") != "base64" or not data.get("content"):
        return None
    return base64.b64decode(data["content"]).decode("utf-8", errors="replace")


def get_changed_python_files(cfg, merge_commit_sha):
    """Lista (filepath, patch) dos arquivos .py alterados no commit."""
    resp = github_get(cfg, f"{cfg['rest_base']}/commits/{merge_commit_sha}")
    if resp is None:
        return []
    files = resp.json().get("files", [])
    return [
        (f["filename"], f.get("patch", ""))
        for f in files
        if f["filename"].endswith(".py") and f.get("status") != "added"
    ]


def linhas_alteradas_no_patch(patch):
    """Números de linha do arquivo ANTES da modificação que aparecem no diff."""
    linhas = set()
    current_line = 0
    for linha in patch.splitlines():
        if linha.startswith("@@"):
            try:
                parte       = linha.split("@@")[1].strip()
                old_part    = parte.split(" ")[0]
                current_line = int(old_part.lstrip("-").split(",")[0])
            except (IndexError, ValueError):
                pass
            continue
        if linha.startswith("-"):
            linhas.add(current_line)
            current_line += 1
        elif not linha.startswith("+"):
            current_line += 1
    return linhas


def calcular_cc_loc_antes(source_code, linhas_afetadas):
    """CC e LOC das funções cujo corpo se sobrepõe com linhas_afetadas."""
    resultados = []

    try:
        blocos_cc = cc_visit(source_code)
    except SyntaxError:
        return resultados

    cc_por_nome    = {b.name: b.complexity for b in blocos_cc}
    linhas_codigo  = source_code.splitlines()

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return resultados

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        inicio = node.lineno
        fim    = node.end_lineno if hasattr(node, "end_lineno") else node.lineno

        if not set(range(inicio, fim + 1)) & linhas_afetadas:
            continue

        nome = node.name
        loc  = sum(
            1 for l in linhas_codigo[inicio - 1: fim]
            if l.strip() and not l.strip().startswith("#")
        )

        resultados.append({
            "funcao_nome": nome,
            "cc_antes":    cc_por_nome.get(nome, 1),
            "loc_antes":   loc,
        })

    return resultados


def analyze_commit_via_api(cfg, merge_commit_sha):
    """Orquestra análise de um commit: retorna lista de {funcao_nome, arquivo, cc_antes, loc_antes}."""
    funcoes = []
    for filepath, patch in get_changed_python_files(cfg, merge_commit_sha):
        if not patch:
            continue
        source_before = get_file_content_before(cfg, filepath, merge_commit_sha)
        if not source_before:
            continue
        linhas_afetadas = linhas_alteradas_no_patch(patch)
        if not linhas_afetadas:
            continue
        for m in calcular_cc_loc_antes(source_before, linhas_afetadas):
            funcoes.append({**m, "arquivo": filepath})
    return funcoes


# --------------------------------------------------------------------------- #
# 5. Métricas de tempo
# --------------------------------------------------------------------------- #

def parse_iso(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def calcular_metricas_tempo(issue_created_at, issue_closed_at,
                             pr_merged_at, commits_timestamps):
    if not commits_timestamps:
        return None

    commits_dt = sorted(parse_iso(t) for t in commits_timestamps)
    primeiro   = commits_dt[0]
    ultimo     = commits_dt[-1]
    t_criada   = parse_iso(issue_created_at)
    t_fechada  = parse_iso(issue_closed_at) if issue_closed_at else None
    t_merged   = parse_iso(pr_merged_at)    if pr_merged_at    else None

    def h(delta):
        return delta.total_seconds() / 3600.0

    return {
        "T_resolucao":     h(t_fechada - t_criada) if t_fechada else None,
        "T_latencia":      h(primeiro - t_criada),
        "T_implementacao": h(ultimo - primeiro),
        "T_integracao":    h(t_merged - ultimo) if t_merged else None,
    }


# --------------------------------------------------------------------------- #
# 6. Pipeline principal
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    period_start = datetime.fromisoformat(args.period_start).replace(tzinfo=timezone.utc)
    period_end   = datetime.fromisoformat(args.period_end).replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    cfg = build_clients(args.owner, args.repo, args.label)

    print(f"[config] {args.owner}/{args.repo}  label={args.label!r}  saída={args.output}")
    print(f"         período: {args.period_start} → {args.period_end}  max_issues={args.max_issues}")

    print("\n[1/3] Buscando issues de bug fechadas com PR mergeada...")
    candidatos = fetch_bug_issues_with_pr(cfg, period_start, period_end, args.max_issues)
    print(f"  -> {len(candidatos)} candidatas encontradas.")

    # Checkpoint
    prs_processadas = set()
    if os.path.exists(args.output):
        df_ex = pd.read_csv(args.output)
        prs_processadas = set(df_ex["pr_id"].astype(int).tolist())
        print(f"  [checkpoint] {len(prs_processadas)} PRs já processadas, serão puladas.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fieldnames = [
        "issue_id", "pr_id", "funcao_nome", "arquivo",
        "cc_antes", "loc_antes",
        "T_resolucao", "T_latencia", "T_implementacao", "T_integracao",
    ]
    write_header = not os.path.exists(args.output)
    csv_file = open(args.output, "a", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    print("\n[2/3] Analisando PRs (API GitHub + radon, sem clone local)...")
    for i, item in enumerate(candidatos, start=1):
        pr  = item["pr_number"]
        iss = item["issue_number"]
        print(f"  ({i}/{len(candidatos)}) issue #{iss} / PR #{pr}")

        if pr in prs_processadas:
            print("    -> já processada, pulando.")
            continue

        commits_ts = get_pr_original_commits(cfg, pr)
        if not commits_ts:
            print("    -> sem commits, pulando.")
            continue

        tempo = calcular_metricas_tempo(
            item["issue_created_at"], item["issue_closed_at"],
            item["pr_merged_at"], commits_ts,
        )
        if tempo is None:
            continue

        funcoes = analyze_commit_via_api(cfg, item["merge_commit_sha"])
        if not funcoes:
            print("    -> nenhuma função .py analisável, pulando.")
            continue

        for func in funcoes:
            writer.writerow({
                "issue_id":    iss,
                "pr_id":       pr,
                "funcao_nome": func["funcao_nome"],
                "arquivo":     func["arquivo"],
                "cc_antes":    func["cc_antes"],
                "loc_antes":   func["loc_antes"],
                **tempo,
            })

        csv_file.flush()
        prs_processadas.add(pr)
        print(f"    -> {len(funcoes)} função(ões) gravada(s).")

    csv_file.close()

    print("\n[3/3] Dataset final:")
    df = pd.read_csv(args.output)
    print(df.head(10))
    print(f"\nTotal de registros : {len(df)}")
    print(f"PRs únicas         : {df['pr_id'].nunique()}")
    print(f"Arquivo            : {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
