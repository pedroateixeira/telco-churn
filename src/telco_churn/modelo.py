"""Split, pipeline, avaliação e lift.

A regra que não se quebra: `dividir` acontece antes de qualquer `fit`, e o
pré-processador vive dentro do `Pipeline`. O tratamento de `dados.tratar` pode
rodar antes do split porque é determinístico linha a linha — ver a docstring
daquela função.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
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


class RegraContrato(BaseEstimator, ClassifierMixin):
    """Baseline de uma linha: quem está em contrato mês a mês é quem cancela.

    Não é formalidade. A EDA mostrou que `Contract` sozinho separa 42,7% de
    churn contra 11,3% e 2,8% — um classificador razoável escrito em uma
    condição. Um modelo que não supere isso é um resultado, e é melhor
    descobrir antes da entrevista do que durante.

    `predict` aplica a regra crua. `predict_proba` devolve a taxa de churn
    observada no treino dentro de cada grupo, o que faz do baseline um
    competidor honesto também em Brier e PR-AUC — comparar uma regra 0/1 com
    modelos probabilísticos usando métricas de probabilidade seria uma
    vitória de régua torta.
    """

    def __init__(self, coluna: str = "Contract", categoria: str = "Month-to-month"):
        self.coluna = coluna
        self.categoria = categoria

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegraContrato":
        marcado = X[self.coluna] == self.categoria
        y = pd.Series(np.asarray(y), index=X.index)
        self.classes_ = np.array([0, 1])
        self.taxa_marcado_ = float(y[marcado].mean())
        self.taxa_resto_ = float(y[~marcado].mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        marcado = (X[self.coluna] == self.categoria).to_numpy()
        p = np.where(marcado, self.taxa_marcado_, self.taxa_resto_)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X[self.coluna] == self.categoria).astype(int).to_numpy()


def construir_pipeline(estimador: BaseEstimator, usar_total_charges: bool = True) -> Pipeline:
    """Encaixa o pré-processador antes do estimador.

    A `RegraContrato` é a exceção: ela lê a coluna `Contract` como texto e não
    passa por aqui, porque o one-hot destruiria justamente o que ela usa.
    """
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


def avaliar_cv(
    construtor: Callable[[], BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = SEED_PADRAO,
) -> dict[str, float]:
    """Média e desvio das métricas em validação cruzada repetida, sobre o TREINO.

    Existe por um motivo específico: o conjunto de teste tem ~1.400 linhas, e o
    desvio-padrão do ROC-AUC nesse tamanho é da ordem de ±0,01. Comparar seis
    modelos por um único split significaria decidir na quarta casa decimal de
    uma amostra — boa parte das diferenças cairia dentro do ruído.

    Por isso a escolha do campeão se faz aqui, com 5 folds repetidos 3 vezes, e
    o conjunto de teste fica intocado para uma única medição final. Olhar para o
    teste a cada comparação é como ajustá-lo a olho: ele deixa de ser uma
    estimativa honesta do desempenho fora da amostra.

    `construtor` é uma função que devolve um estimador novo, e não um estimador
    já pronto — cada fold precisa treinar do zero.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    resultados: dict[str, list[float]] = {"roc_auc": [], "pr_auc": [], "brier": []}

    for idx_treino, idx_val in cv.split(X, y):
        X_treino, X_val = X.iloc[idx_treino], X.iloc[idx_val]
        y_treino, y_val = y.iloc[idx_treino], y.iloc[idx_val]

        modelo = clone(construtor())
        modelo.fit(X_treino, y_treino)
        p = modelo.predict_proba(X_val)[:, 1]

        resultados["roc_auc"].append(roc_auc_score(y_val, p))
        resultados["pr_auc"].append(average_precision_score(y_val, p))
        resultados["brier"].append(brier_score_loss(y_val, p))

    resumo: dict[str, float] = {}
    for metrica, valores in resultados.items():
        resumo[f"{metrica}_media"] = float(np.mean(valores))
        resumo[f"{metrica}_desvio"] = float(np.std(valores))
    resumo["n_ajustes"] = n_splits * n_repeats
    return resumo


def curva_calibracao(y: pd.Series, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Frequência observada contra probabilidade prevista, por faixa.

    Um modelo calibrado põe os pontos sobre a diagonal: entre os clientes a quem
    ele atribui 30% de risco, 30% de fato cancelam. A Fase 4 calcula valor
    esperado a partir do valor absoluto dessa probabilidade — se ela estiver
    inflada, a campanha inteira é dimensionada errado, por mais que o ranking
    (e portanto o ROC-AUC) esteja perfeito.
    """
    observado, previsto = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"previsto": previsto, "observado": observado})


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
