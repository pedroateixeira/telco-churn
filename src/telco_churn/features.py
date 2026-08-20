"""Pré-processamento das features.

Tudo que aprende algo a partir dos dados vive aqui — e, por consequência,
vive dentro do `Pipeline`, onde só enxerga o conjunto de treino. É a diferença
entre escalar com a média do treino e escalar com a média de tudo, que é o
vazamento mais comum de quem está começando.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_churn.dados import (
    ALVO,
    COLUNAS_CATEGORICAS,
    COLUNAS_NUMERICAS,
    SCHEMA_BRUTO,
)

# Colunas 0/1 depois de `dados.tratar`: tudo que sobra quando se tiram as
# numéricas, as categóricas de texto, o alvo e o identificador. Derivar isto do
# schema, em vez de escrever a lista à mão, garante que uma coluna nova no dado
# não passe despercebida — ela apareceria aqui e quebraria o teste de schema.
COLUNAS_BINARIAS_MODELO = [
    col
    for col in SCHEMA_BRUTO
    if col not in {*COLUNAS_NUMERICAS, *COLUNAS_CATEGORICAS, ALVO, "customerID"}
]


def construir_preprocessador(usar_total_charges: bool = True) -> ColumnTransformer:
    """Monta o `ColumnTransformer` aplicado ao DataFrame já tratado.

    `usar_total_charges` existe para a ablação da Fase 3: a EDA demonstrou que
    `TotalCharges` é reconstituível de `tenure x MonthlyCharges` e ainda mistura
    dois sinais de direções opostas. Deixar isso como parâmetro transforma o
    achado em experimento reproduzível, em vez de uma edição manual de código.
    """
    numericas = [
        col for col in COLUNAS_NUMERICAS if usar_total_charges or col != "TotalCharges"
    ]

    return ColumnTransformer(
        [
            ("num", StandardScaler(), numericas),
            # handle_unknown="ignore": categoria nunca vista em produção vira um
            # vetor de zeros em vez de derrubar a predição.
            #
            # Sem drop="first": medi as duas versões (26 features contra 23) e o
            # resultado é o mesmo até a terceira casa. Sem o drop, o mesmo
            # pré-processador serve modelo linear e árvore — a regularização L2
            # da logística já lida com a colinearidade, e para árvore o drop
            # atrapalha de leve. Um caminho de código em vez de dois.
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                COLUNAS_CATEGORICAS,
            ),
            # Já são 0/1: escalar não mudaria nada e só atrapalharia a leitura
            # dos coeficientes.
            ("bin", "passthrough", COLUNAS_BINARIAS_MODELO),
        ],
        # Explícito de propósito: coluna nova no dado não entra no modelo sem
        # alguém ter decidido o que fazer com ela.
        remainder="drop",
    )


def nomes_features(preprocessador: ColumnTransformer) -> list[str]:
    """Nomes das features na saída do pré-processador, para ler coeficientes."""
    return list(preprocessador.get_feature_names_out())
