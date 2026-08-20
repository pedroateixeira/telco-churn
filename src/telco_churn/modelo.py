"""Split, pipeline, avaliação e lift.

A regra que não se quebra: `dividir` acontece antes de qualquer `fit`, e o
pré-processador vive dentro do `Pipeline`. O tratamento de `dados.tratar` pode
rodar antes do split porque é determinístico linha a linha — ver a docstring
daquela função.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from telco_churn.dados import ALVO
from telco_churn.features import construir_preprocessador

SEED_PADRAO = 42
TEST_SIZE_PADRAO = 0.2


def dividir(
    df: pd.DataFrame, test_size: float = TEST_SIZE_PADRAO, seed: int = SEED_PADRAO
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split estratificado por `Churn`.

    Estratificado porque a taxa base é 26,5%: um split aleatório simples pode
    deslocar a proporção entre treino e teste o suficiente para bagunçar a
    comparação de métricas.

    Ressalva declarada: o Telco não tem nenhuma coluna de data — `tenure` é
    duração, não data de entrada do cliente —, então não há como fazer split
    temporal por safra, que seria o correto. A performance medida aqui é
    otimista em relação ao que se veria em produção.
    """
    X = df.drop(columns=[ALVO])
    y = df[ALVO]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def construir_pipeline(estimador: BaseEstimator, usar_total_charges: bool = True) -> Pipeline:
    """Encaixa o pré-processador antes do estimador."""
    return Pipeline(
        [
            ("pre", construir_preprocessador(usar_total_charges=usar_total_charges)),
            ("clf", estimador),
        ]
    )


def avaliar(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float | int]:
    """Métricas de um pipeline já treinado, sobre um conjunto qualquer.

    ROC-AUC ordena; PR-AUC é a que importa com classe desbalanceada; Brier mede
    se a probabilidade é *calibrada*, e não só se a ordem está certa — a camada
    de decisão da Fase 4 depende do valor absoluto da probabilidade, não do
    ranking, então esta é a métrica que não pode degradar em silêncio.
    """
    p = pipeline.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "taxa_base": float(np.mean(y)),
        "n": int(len(y)),
    }


def lift_por_decil(y: pd.Series, p: np.ndarray, n_decis: int = 10) -> pd.DataFrame:
    """Taxa de churn e lift por decil de risco, do mais arriscado para o menos.

    É a métrica que a operação de retenção realmente usa: a campanha atinge um
    número limitado de clientes, então o que importa é acertar no topo da lista,
    não a qualidade média da predição em toda a base.

    Lift de 2,5 no primeiro decil significa: entre os 10% mais arriscados, a
    taxa de churn é 2,5x a da carteira inteira.

    Levanta `ValueError` em recortes pequenos ou sem nenhum churn. Os dois casos
    são plausíveis no dashboard, onde a base é filtrada interativamente, e nos
    dois a função devolveria uma tabela sem sentido: com menos linhas que decis
    o `qcut` produz faixas vazias e rótulos salteados (1, 5, 10); sem nenhum
    churn, lift e captura viram `NaN` por divisão por zero. Falhar alto é melhor
    que devolver um gráfico que parece certo.
    """
    ranking = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})

    if len(ranking) < n_decis:
        raise ValueError(
            f"lift por decil precisa de ao menos {n_decis} clientes, recebeu {len(ranking)}"
        )
    if ranking["y"].sum() == 0:
        raise ValueError("lift por decil não é definido em um recorte sem nenhum churn")

    # rank antes de cortar: probabilidades empatadas são comuns e o qcut puro
    # falharia ao tentar criar bordas iguais.
    ranking["decil"] = pd.qcut(
        ranking["p"].rank(method="first", ascending=False),
        n_decis,
        labels=range(1, n_decis + 1),
    )

    taxa_base = ranking["y"].mean()
    resumo = (
        ranking.groupby("decil", observed=True)
        .agg(n=("y", "size"), churns=("y", "sum"), taxa_churn=("y", "mean"))
        .reset_index()
    )
    resumo["lift"] = resumo["taxa_churn"] / taxa_base
    resumo["churns_acumulados"] = resumo["churns"].cumsum()
    resumo["captura_acumulada"] = resumo["churns_acumulados"] / ranking["y"].sum()
    return resumo
