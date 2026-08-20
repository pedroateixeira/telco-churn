"""Catálogo dos modelos comparados na Fase 3.

Cada entrada é uma *fábrica* — uma função que devolve um estimador novo — e não
um estimador pronto. A validação cruzada treina do zero a cada fold, e reusar a
mesma instância vazaria estado de um fold para o outro.

Os hiperparâmetros são deliberadamente conservadores. Esta fase compara famílias
de modelo sobre um dado pequeno (5.634 linhas de treino); busca de
hiperparâmetro entraria como mais uma fonte de variância entre candidatos que
já se distinguem por menos que o ruído do split.
"""

from collections.abc import Callable

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from telco_churn.modelo import SEED_PADRAO, RegraContrato, construir_pipeline

Fabrica = Callable[[], BaseEstimator]


def _logistica(seed: int = SEED_PADRAO) -> BaseEstimator:
    return LogisticRegression(max_iter=1000)


def _adaboost(seed: int = SEED_PADRAO) -> BaseEstimator:
    return AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=seed)


def _xgboost(seed: int = SEED_PADRAO) -> BaseEstimator:
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=seed,
    )


def _lightgbm(seed: int = SEED_PADRAO) -> BaseEstimator:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )


def _base_do_ensemble(seed: int = SEED_PADRAO) -> list[tuple[str, BaseEstimator]]:
    """Os quatro modelos que aprendem, que é o que o ensemble combina."""
    return [
        ("logistica", _logistica(seed)),
        ("adaboost", _adaboost(seed)),
        ("xgboost", _xgboost(seed)),
        ("lightgbm", _lightgbm(seed)),
    ]


def _votacao(seed: int = SEED_PADRAO) -> BaseEstimator:
    """Média simples das probabilidades dos quatro.

    Votação *soft*, não *hard*: a Fase 4 precisa de probabilidade, e votar em
    rótulos 0/1 jogaria fora exatamente a informação que a camada de decisão
    consome. A média de probabilidades também tende a melhorar a calibração,
    porque erros independentes dos modelos se cancelam em parte.
    """
    return VotingClassifier(estimators=_base_do_ensemble(seed), voting="soft")


def _stacking(seed: int = SEED_PADRAO) -> BaseEstimator:
    """Combina os quatro com uma logística aprendendo os pesos.

    A diferença para a votação: em vez de assumir que os quatro merecem peso
    igual, o meta-modelo aprende quanto confiar em cada um, a partir de
    predições fora da amostra (`cv=5`). Custa mais treino e arrisca sobreajuste
    no nível de cima, e é por isso que os dois entram na comparação em vez de
    eu escolher no palpite.
    """
    return StackingClassifier(
        estimators=_base_do_ensemble(seed),
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method="predict_proba",
    )


def catalogo(usar_total_charges: bool = True, seed: int = SEED_PADRAO) -> dict[str, Fabrica]:
    """Nome -> fábrica de estimador, na ordem em que a escada é lida.

    A `RegraContrato` é a única que não recebe o pré-processador: ela lê a
    coluna `Contract` como texto, e o one-hot destruiria justamente o que ela
    usa. Todas as outras vão embrulhadas no `Pipeline`, o que garante que a
    escala e o encoding só enxergam o fold de treino.
    """

    def com_pipeline(construtor: Callable[[int], BaseEstimator]) -> Fabrica:
        return lambda: construir_pipeline(
            construtor(seed), usar_total_charges=usar_total_charges
        )

    return {
        "Regra de contrato": lambda: RegraContrato(),
        "Regressão logística": com_pipeline(_logistica),
        "AdaBoost": com_pipeline(_adaboost),
        "XGBoost": com_pipeline(_xgboost),
        "LightGBM": com_pipeline(_lightgbm),
        "Ensemble (votação)": com_pipeline(_votacao),
        "Ensemble (stacking)": com_pipeline(_stacking),
    }
