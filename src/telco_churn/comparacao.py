"""Roda a escada de modelos, a ablação e o experimento de calibração.

    uv run python -m telco_churn.comparacao

Grava a tabela de comparação em `models/comparacao.json` e as figuras em
`reports/figures/`. O notebook 02 lê esses artefatos em vez de retreinar tudo,
para que a narrativa e os números não possam divergir.

A escolha do campeão é feita por validação cruzada sobre o TREINO. O conjunto
de teste é medido uma única vez, no fim, e só para o campeão e seus vizinhos —
olhar para ele a cada comparação transformaria a estimativa final em otimismo.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # sem display: este módulo roda em terminal e em CI

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

from telco_churn import graficos  # noqa: E402
from telco_churn.catalogo import catalogo  # noqa: E402
from telco_churn.dados import RAIZ_PROJETO, carregar_tratado  # noqa: E402
from telco_churn.modelo import (  # noqa: E402
    SEED_PADRAO,
    avaliar,
    avaliar_cv,
    construir_pipeline,
    curva_calibracao,
    dividir,
    lift_por_decil,
)

FIGURAS_PADRAO = RAIZ_PROJETO / "reports" / "figures"
SAIDA_PADRAO = RAIZ_PROJETO / "models"


def comparar_por_cv(X, y, usar_total_charges: bool = True, seed: int = SEED_PADRAO) -> pd.DataFrame:
    """Validação cruzada 5×3 de cada modelo do catálogo."""
    linhas = []
    for nome, fabrica in catalogo(usar_total_charges=usar_total_charges, seed=seed).items():
        print(f"  {nome} ...", end="", flush=True)
        resultado = avaliar_cv(fabrica, X, y, seed=seed)
        print(
            f" ROC {resultado['roc_auc_media']:.4f} ±{resultado['roc_auc_desvio']:.4f}"
            f" | Brier {resultado['brier_media']:.4f}"
        )
        linhas.append({"modelo": nome, **resultado})
    return pd.DataFrame(linhas)


def comparar_pareado(
    X, y, referencia: str = "Regressão logística", seed: int = SEED_PADRAO
) -> pd.DataFrame:
    """Compara cada modelo contra uma referência nos MESMOS folds.

    Por que pareado: a variação entre folds (±0,019 de PR-AUC) é muito maior que
    a diferença entre modelos (~0,010). Comparando médias com seus desvios, toda
    diferença parece afogada no ruído. Mas os folds são compartilhados — um fold
    difícil é difícil para todos —, e a diferença *dentro de cada fold* cancela
    essa variação comum.

    A pergunta certa não é "a diferença é maior que o desvio entre folds?", e sim
    "o sinal da diferença se mantém fold a fold?". Um modelo que vence em 15 de
    15 folds por pouco é melhor com mais confiança do que a comparação de médias
    sugere.
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    folds = list(cv.split(X, y))
    fabricas = catalogo(seed=seed)

    scores: dict[str, list[float]] = {nome: [] for nome in fabricas}
    tempos: dict[str, float] = dict.fromkeys(fabricas, 0.0)

    for idx_treino, idx_val in folds:
        X_a, X_b = X.iloc[idx_treino], X.iloc[idx_val]
        y_a, y_b = y.iloc[idx_treino], y.iloc[idx_val]
        for nome, fabrica in fabricas.items():
            modelo = fabrica()
            inicio = time.perf_counter()
            modelo.fit(X_a, y_a)
            tempos[nome] += time.perf_counter() - inicio
            scores[nome].append(average_precision_score(y_b, modelo.predict_proba(X_b)[:, 1]))

    base = np.array(scores[referencia])
    linhas = []
    for nome, valores in scores.items():
        diferenca = np.array(valores) - base
        linhas.append(
            {
                "modelo": nome,
                "pr_auc_media": float(np.mean(valores)),
                "dif_media": float(diferenca.mean()),
                "dif_desvio": float(diferenca.std()),
                "folds_vencidos": int((diferenca > 0).sum()),
                "n_folds": len(folds),
                "segundos_por_ajuste": tempos[nome] / len(folds),
            }
        )
    return pd.DataFrame(linhas)


def experimento_calibracao(X_treino, y_treino, X_teste, y_teste, seed: int = SEED_PADRAO) -> dict:
    """Mede o preço do balanceamento de classe sobre a calibração.

    Três variantes da mesma regressão logística:

    1. como está — sem balanceamento;
    2. com `class_weight="balanced"`, que é o reflexo comum diante de 26,5% de
       positivos;
    3. a balanceada, recalibrada por regressão isotônica.

    A hipótese a testar: (2) melhora o recall e piora o Brier, porque reponderar
    a classe positiva infla sistematicamente a probabilidade prevista. Se for o
    caso, a Fase 4 não pode usar (2) — ela calcula valor esperado a partir do
    nível da probabilidade, não do ranking.
    """
    variantes = {
        "Sem balanceamento": construir_pipeline(LogisticRegression(max_iter=1000)),
        "class_weight='balanced'": construir_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "Balanceada + isotônica": CalibratedClassifierCV(
            construir_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
            method="isotonic",
            cv=5,
        ),
    }

    metricas, calibracoes = [], {}
    for nome, modelo in variantes.items():
        modelo.fit(X_treino, y_treino)
        resultado = avaliar(modelo, X_teste, y_teste)
        p = modelo.predict_proba(X_teste)[:, 1]
        metricas.append(
            {
                "variante": nome,
                "roc_auc": resultado["roc_auc"],
                "pr_auc": resultado["pr_auc"],
                "brier": resultado["brier"],
                "prob_media": float(p.mean()),
            }
        )
        calibracoes[nome] = curva_calibracao(y_teste, p)
        print(
            f"  {nome:26s} ROC {resultado['roc_auc']:.4f}"
            f" | Brier {resultado['brier']:.4f} | p média {p.mean():.3f}"
        )

    return {"metricas": pd.DataFrame(metricas), "calibracoes": calibracoes}


def executar(
    seed: int = SEED_PADRAO,
    figuras: Path = FIGURAS_PADRAO,
    saida: Path = SAIDA_PADRAO,
) -> dict:
    df = carregar_tratado()
    X_treino, X_teste, y_treino, y_teste = dividir(df, seed=seed)
    taxa_base = float(y_teste.mean())
    print(f"treino {len(X_treino)} | teste {len(X_teste)} | taxa base {taxa_base:.4f}\n")

    print("Validação cruzada 5x3, com TotalCharges:")
    cv_com = comparar_por_cv(X_treino, y_treino, usar_total_charges=True, seed=seed)

    print("\nValidação cruzada 5x3, sem TotalCharges (ablação):")
    cv_sem = comparar_por_cv(X_treino, y_treino, usar_total_charges=False, seed=seed)

    # Campeão pelo PR-AUC: com 26,5% de positivos é a métrica que reflete o uso
    # real (uma lista de contatos), enquanto o ROC-AUC é otimista por incluir a
    # facilidade de acertar os negativos, que são a maioria.
    lider = cv_com.loc[cv_com["pr_auc_media"].idxmax(), "modelo"]

    print("\nComparação pareada nos mesmos folds (referência: regressão logística):")
    pareado = comparar_pareado(X_treino, y_treino, seed=seed)
    for linha in pareado.sort_values("pr_auc_media", ascending=False).itertuples():
        print(
            f"  {linha.modelo:22s} PR {linha.pr_auc_media:.4f}"
            f" | dif {linha.dif_media:+.4f} ±{linha.dif_desvio:.4f}"
            f" | vence {linha.folds_vencidos}/{linha.n_folds}"
            f" | {linha.segundos_por_ajuste:5.2f}s/ajuste"
        )

    # Regra de desempate declarada: entre os modelos que empatam com o líder — o
    # sinal da diferença pareada contra ele não se sustenta nos folds —, fica o
    # mais barato de treinar. Complexidade que não paga em métrica é custo puro:
    # mais tempo de treino, mais superfície de falha no deploy, menos
    # explicabilidade numa entrevista.
    scores_lider = pareado.set_index("modelo").loc[lider]
    empatados = []
    for linha in pareado.itertuples():
        if linha.modelo == lider:
            continue
        dif = linha.dif_media - scores_lider["dif_media"]
        if abs(dif) < linha.dif_desvio:
            empatados.append(linha.modelo)

    candidatos = [lider, *empatados]
    custos = pareado.set_index("modelo")["segundos_por_ajuste"]
    campeao = min(candidatos, key=lambda nome: custos[nome])

    print(f"\n  Líder por PR-AUC:        {lider} ({custos[lider]:.2f}s/ajuste)")
    print(f"  Empatam com ele:         {', '.join(empatados) or 'nenhum'}")
    print(f"  Campeão (mais barato):   {campeao} ({custos[campeao]:.2f}s/ajuste)")

    print("\nMedição única no teste:")
    fabricas = catalogo(usar_total_charges=True, seed=seed)
    curvas, calibracoes, lifts, brier, metricas_teste = {}, {}, {}, {}, []
    for nome, fabrica in fabricas.items():
        modelo = fabrica()
        modelo.fit(X_treino, y_treino)
        p = modelo.predict_proba(X_teste)[:, 1]
        resultado = avaliar(modelo, X_teste, y_teste)

        curvas[nome] = (y_teste.to_numpy(), p)
        calibracoes[nome] = curva_calibracao(y_teste, p)
        lifts[nome] = lift_por_decil(y_teste, p)
        brier[nome] = resultado["brier"]
        metricas_teste.append({"modelo": nome, **resultado})
        print(
            f"  {nome:22s} ROC {resultado['roc_auc']:.4f}"
            f" | PR {resultado['pr_auc']:.4f} | Brier {resultado['brier']:.4f}"
            f" | lift decil 1 {lifts[nome]['lift'].iloc[0]:.2f}"
        )

    print("\nExperimento de calibração:")
    calibracao = experimento_calibracao(X_treino, y_treino, X_teste, y_teste, seed=seed)

    figuras.mkdir(parents=True, exist_ok=True)
    saidas = {
        "comparacao_metricas.png": graficos.comparacao_metricas(cv_com, campeao),
        "curvas_roc.png": graficos.curvas_roc(curvas, taxa_base, campeao),
        "curvas_precisao_recall.png": graficos.curvas_precisao_recall(curvas, taxa_base, campeao),
        "curvas_calibracao.png": graficos.curvas_calibracao(calibracoes, brier, campeao),
        "comparacao_lift.png": graficos.comparacao_lift(lifts, campeao),
        "ablacao_total_charges.png": graficos.ablacao_total_charges(cv_com, cv_sem),
        "efeito_class_weight.png": graficos.efeito_class_weight(
            calibracao["calibracoes"], calibracao["metricas"]
        ),
    }
    for nome_arquivo, figura in saidas.items():
        figura.savefig(figuras / nome_arquivo, dpi=140, bbox_inches="tight")
        print(f"  figura: {figuras / nome_arquivo}")

    relatorio = {
        "seed": seed,
        "taxa_base_teste": taxa_base,
        "lider_pr_auc": lider,
        "campeao_cv": campeao,
        "empatados_com_lider": empatados,
        "pareado": pareado.to_dict(orient="records"),
        "cv_com_total_charges": cv_com.to_dict(orient="records"),
        "cv_sem_total_charges": cv_sem.to_dict(orient="records"),
        "teste": metricas_teste,
        "lift_por_decil": {
            nome: lift.astype({"decil": int}).to_dict(orient="records")
            for nome, lift in lifts.items()
        },
        "calibracao": calibracao["metricas"].to_dict(orient="records"),
    }
    saida.mkdir(parents=True, exist_ok=True)
    caminho = saida / "comparacao.json"
    caminho.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nComparação salva em {caminho}")
    return relatorio


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m telco_churn.comparacao",
        description="Compara a escada de modelos por validação cruzada e gera as figuras.",
    )
    parser.add_argument("--seed", type=int, default=SEED_PADRAO)
    parser.add_argument("--figuras", type=Path, default=FIGURAS_PADRAO)
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    args = parser.parse_args(argv)
    executar(seed=args.seed, figuras=args.figuras, saida=args.saida)


if __name__ == "__main__":
    main()
