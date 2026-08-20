"""Carga, validação de schema e tratamento dos dados brutos.

Esta é a fronteira do sistema: nada entra sem passar por `validar_schema`.
O erro que não for pego aqui vira modelo treinado em lixo três semanas depois.

O tratamento reproduz, em forma importável, as células 8 a 27 de
`notebooks/01_eda_telco.ipynb`, onde cada decisão está justificada em texto.
"""

from pathlib import Path

import pandas as pd
from pandas.api import types as pdt

ALVO = "Churn"

COLUNAS_BINARIAS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

COLUNAS_ADDON = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Depois de `tratar`, só estas três continuam como texto — as demais viram 0/1.
COLUNAS_CATEGORICAS = ["InternetService", "Contract", "PaymentMethod"]

COLUNAS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]

_SIM_NAO = ("Yes", "No")

# Schema do arquivo BRUTO, como ele sai do Kaggle. Cada entrada é
# (tipo esperado, domínio esperado) — domínio `None` para colunas contínuas.
#
# Repare que `TotalCharges` é esperado como TEXTO: no arquivo original ela vem
# com espaços em branco no lugar de valores ausentes, e é por isso que o
# tratamento precisa convertê-la explicitamente.
SCHEMA_BRUTO: dict[str, tuple[str, tuple[str, ...] | None]] = {
    "customerID": ("texto", None),
    "gender": ("texto", ("Male", "Female")),
    "SeniorCitizen": ("inteiro", None),
    "Partner": ("texto", _SIM_NAO),
    "Dependents": ("texto", _SIM_NAO),
    "tenure": ("inteiro", None),
    "PhoneService": ("texto", _SIM_NAO),
    "MultipleLines": ("texto", ("Yes", "No", "No phone service")),
    "InternetService": ("texto", ("DSL", "Fiber optic", "No")),
    **{col: ("texto", ("Yes", "No", "No internet service")) for col in COLUNAS_ADDON},
    "Contract": ("texto", ("Month-to-month", "One year", "Two year")),
    "PaperlessBilling": ("texto", _SIM_NAO),
    "PaymentMethod": (
        "texto",
        (
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ),
    ),
    "MonthlyCharges": ("decimal", None),
    "TotalCharges": ("texto", None),
    "Churn": ("texto", _SIM_NAO),
}


class ErroDeSchema(ValueError):
    """O dado de entrada não é o que o pipeline espera."""


def caminho_padrao() -> Path:
    """Caminho do CSV bruto, resolvido a partir da localização do pacote.

    Deliberadamente não usa `Path.cwd()`: o diretório de trabalho muda conforme
    quem inicia o processo — notebook, `python -m`, ou o Streamlit Cloud — e
    ancorar no arquivo é a única forma de funcionar nos três casos.
    """
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_customer_churn.csv"


def carregar_brutos(caminho: Path | str | None = None) -> pd.DataFrame:
    """Lê o CSV original, sem tratar nada."""
    return pd.read_csv(Path(caminho) if caminho is not None else caminho_padrao())


def _tipo_confere(serie: pd.Series, esperado: str) -> bool:
    if esperado == "inteiro":
        return pdt.is_integer_dtype(serie)
    if esperado == "decimal":
        return pdt.is_float_dtype(serie)
    # pandas 3 entrega texto como StringDtype, não como object: uma checagem
    # escrita como `dtype == object` passaria silenciosamente por tudo.
    return pdt.is_string_dtype(serie)


def validar_schema(df: pd.DataFrame, schema: dict | None = None) -> None:
    """Confere colunas, tipos e domínios do DataFrame bruto.

    Levanta `ErroDeSchema` listando **todos** os problemas de uma vez. Um
    validador que falha um por vez obriga a rodar o pipeline N vezes para
    descobrir N erros.
    """
    schema = schema if schema is not None else SCHEMA_BRUTO
    problemas: list[str] = []

    faltando = [col for col in schema if col not in df.columns]
    if faltando:
        problemas.append(f"colunas ausentes: {faltando}")

    sobrando = [col for col in df.columns if col not in schema]
    if sobrando:
        problemas.append(f"colunas inesperadas: {sobrando}")

    for col, (tipo_esperado, dominio) in schema.items():
        if col not in df.columns:
            continue

        if not _tipo_confere(df[col], tipo_esperado):
            problemas.append(f"{col}: esperava {tipo_esperado}, veio {df[col].dtype}")
            continue  # domínio de uma coluna com tipo errado não diz nada útil

        if dominio is not None:
            fora = sorted(set(df[col].dropna().unique()) - set(dominio))
            if fora:
                problemas.append(f"{col}: valores fora do domínio {list(dominio)}: {fora}")

    if problemas:
        raise ErroDeSchema(
            "O dado bruto não bate com o schema esperado:\n  - " + "\n  - ".join(problemas)
        )


def tratar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica o tratamento definido na EDA. Puro: não altera o DataFrame recebido.

    Todas as transformações aqui são determinísticas linha a linha — mapeamento
    de texto para 0/1 e preenchimento com constante. Nenhuma estatística é
    aprendida a partir do conjunto, e é por isso que esta função pode rodar
    ANTES do split treino/teste sem causar vazamento. Qualquer transformação
    que calcule um número olhando várias linhas (média, escala, lista de
    categorias) tem de ficar dentro do `Pipeline` — ver `features.py`.
    """
    tratado = df.copy()

    # customerID identifica o cliente, não descreve o comportamento dele.
    tratado = tratado.drop(columns=["customerID"])

    for col in COLUNAS_BINARIAS:
        tratado[col] = tratado[col].map({"Yes": 1, "No": 0})
    tratado["gender"] = tratado["gender"].map({"Male": 1, "Female": 0})

    # "No internet service" é implicado por InternetService == "No": a crosstab
    # da célula 15 do notebook mostra que a categoria nunca aparece junto de um
    # "Yes". Como terceira categoria ela não carrega informação nova.
    for col in COLUNAS_ADDON:
        tratado[col] = tratado[col].replace("No internet service", "No").map({"Yes": 1, "No": 0})

    # Mesmo caso, para telefone: PhoneService == 0 <=> "No phone service",
    # implicação perfeita em 682 linhas. O notebook deixou esta coluna como
    # texto de três categorias; aqui ela segue o mesmo padrão dos add-ons.
    tratado["MultipleLines"] = (
        tratado["MultipleLines"].replace("No phone service", "No").map({"Yes": 1, "No": 0})
    )

    # No arquivo bruto TotalCharges é texto, com espaço em branco onde deveria
    # haver valor. São 11 clientes, todos com tenure == 0: entraram e ainda não
    # fecharam o primeiro ciclo de cobrança, então 0 é o valor correto e não uma
    # imputação — a constante não depende dos dados, e por isso não vaza.
    tratado["TotalCharges"] = pd.to_numeric(tratado["TotalCharges"], errors="coerce").fillna(0)

    return tratado


def carregar_tratado(caminho: Path | str | None = None) -> pd.DataFrame:
    """Carrega, valida e trata. É o ponto de entrada do resto do pacote."""
    brutos = carregar_brutos(caminho)
    validar_schema(brutos)
    return tratar(brutos)
