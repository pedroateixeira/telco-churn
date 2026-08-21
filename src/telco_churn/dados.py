"""Carga, validação de schema e tratamento dos dados brutos.

Esta é a fronteira do sistema: nada entra sem passar por `validar_schema`.
O erro que não for pego aqui vira modelo treinado em lixo três semanas depois.

O tratamento reproduz, em forma importável, as células 8 a 27 de
`notebooks/01_eda_telco.ipynb`, onde cada decisão está justificada em texto.
"""

from pathlib import Path

import pandas as pd
from pandas.api import types as pdt

RAIZ_PROJETO = Path(__file__).resolve().parents[2]

ALVO = "Churn"

# Colunas mapeadas de "Yes"/"No" para 1/0 por `tratar`. O alvo está incluído de
# propósito: `Churn` passa pelo mesmo mapeamento.
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


# Colunas 0/1 depois de `tratar`: tudo que sobra quando se tiram as numéricas,
# as categóricas de texto, o alvo e o identificador. Derivar do schema, em vez
# de escrever a lista à mão, evita que as duas versões divirjam.
COLUNAS_BINARIAS_MODELO = [
    col
    for col in SCHEMA_BRUTO
    if col not in {*COLUNAS_NUMERICAS, *COLUNAS_CATEGORICAS, ALVO, "customerID"}
]


class ErroDeSchema(ValueError):
    """O dado de entrada não é o que o pipeline espera."""


def caminho_padrao() -> Path:
    """Caminho do CSV bruto, resolvido a partir da localização do pacote.

    Deliberadamente não usa `Path.cwd()`: o diretório de trabalho muda conforme
    quem inicia o processo — notebook, `python -m`, ou o Streamlit Cloud — e
    ancorar no arquivo é a única forma de funcionar nos três casos.
    """
    return RAIZ_PROJETO / "data" / "raw" / "telco_customer_churn.csv"


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


def validar_tratado(df: pd.DataFrame) -> None:
    """Confere que o DataFrame tratado cumpre o contrato que `features.py` assume.

    `features.py` monta o pré-processador declarando que tudo fora das numéricas,
    das categóricas e do alvo é 0/1, e manda esse resto por `passthrough`. Quem
    torna isso verdade é `tratar` — ou seja, é uma correspondência entre dois
    módulos que nada verificava.

    O caso que essa checagem pega: alguém acrescenta uma coluna ao `SCHEMA_BRUTO`
    (o passo natural depois que a validação do bruto acusa "coluna inesperada").
    Se for texto, o `fit` estoura alto e o erro aparece. Mas se for numérica
    contínua — uma latitude, um valor de reembolso —, ela entra no `passthrough`
    sem escala, ao lado de features escaladas, e degrada o modelo linear em
    silêncio. Silêncio é o que não pode acontecer.
    """
    problemas: list[str] = []

    for col in COLUNAS_BINARIAS_MODELO:
        if col not in df.columns:
            problemas.append(f"{col}: ausente depois de tratar")
            continue
        valores = set(df[col].dropna().unique())
        if not valores <= {0, 1}:
            # Uma coluna contínua intrusa tem milhares de valores distintos:
            # mostrar uma amostra, ou a mensagem de erro vira um despejo ilegível.
            amostra = sorted(valores - {0, 1}, key=str)[:5]
            resumo = ", ".join(str(v) for v in amostra)
            if len(valores) > len(amostra):
                resumo += f", ... ({len(valores)} valores distintos)"
            problemas.append(f"{col}: deveria ser 0/1 depois de tratar, veio {resumo}")

    if problemas:
        raise ErroDeSchema(
            "O dado tratado não cumpre o contrato esperado por features.py:\n  - "
            + "\n  - ".join(problemas)
        )


def carregar_tratado(caminho: Path | str | None = None) -> pd.DataFrame:
    """Carrega, valida e trata. É o ponto de entrada do resto do pacote."""
    brutos = carregar_brutos(caminho)
    validar_schema(brutos)
    tratado = tratar(brutos)
    validar_tratado(tratado)
    return tratado

# Cliente de partida do simulador do dashboard: valores medianos da base. Abrir
# o formulário preenchido, em vez de vazio, evita que a primeira predição saia
# de uma combinação que ninguém escolheu.
CLIENTE_EXEMPLO: dict = {
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "tenure": 29,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": "Fiber optic",
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "Contract": "Month-to-month",
    "PaperlessBilling": 1,
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 1394.55,
}

# Rótulos em português para a interface. O dado vem em inglês do Kaggle e as
# colunas mantêm o nome original — traduzir no schema quebraria a rastreabilidade
# com a fonte, então a tradução fica só na camada de apresentação.
ROTULOS_PT: dict[str, str] = {
    "gender": "Gênero",
    "SeniorCitizen": "Idoso (65+)",
    "Partner": "Tem parceiro(a)",
    "Dependents": "Tem dependentes",
    "tenure": "Tempo de casa (meses)",
    "PhoneService": "Serviço de telefone",
    "MultipleLines": "Múltiplas linhas",
    "InternetService": "Tipo de internet",
    "OnlineSecurity": "Segurança online",
    "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de aparelho",
    "TechSupport": "Suporte técnico",
    "StreamingTV": "Streaming de TV",
    "StreamingMovies": "Streaming de filmes",
    "Contract": "Contrato",
    "PaperlessBilling": "Fatura digital",
    "PaymentMethod": "Forma de pagamento",
    "MonthlyCharges": "Mensalidade (R$)",
    "TotalCharges": "Total já pago (R$)",
    "Churn": "Cancelou",
}

VALORES_PT: dict[str, dict] = {
    "InternetService": {"DSL": "DSL", "Fiber optic": "Fibra ótica", "No": "Sem internet"},
    "Contract": {
        "Month-to-month": "Mês a mês",
        "One year": "Um ano",
        "Two year": "Dois anos",
    },
    "PaymentMethod": {
        "Electronic check": "Débito eletrônico",
        "Mailed check": "Boleto pelo correio",
        "Bank transfer (automatic)": "Transferência automática",
        "Credit card (automatic)": "Cartão de crédito automático",
    },
}
