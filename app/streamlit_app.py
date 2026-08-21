"""Dashboard de retenção — Telco Churn.

    uv run streamlit run app/streamlit_app.py

Quatro abas: segmentos de risco, simulador individual, simulação de campanha e
metodologia. Nada é recalculado em lote a cada interação porque nada precisa:
o caminho inteiro (carregar, prever 1.409 clientes, simular as quatro filas)
leva menos de 100 ms, então os controles recalculam ao vivo.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# O Streamlit Cloud inicia o processo da raiz do repositório, sem o pacote
# instalado. Ancorar em __file__ é o que faz o import funcionar nos dois lugares.
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from telco_churn import decisao, graficos  # noqa: E402
from telco_churn.dados import (  # noqa: E402
    ALVO,
    CLIENTE_EXEMPLO,
    COLUNAS_ADDON,
    ROTULOS_PT,
    VALORES_PT,
    carregar_tratado,
)
from telco_churn.modelo import dividir  # noqa: E402

st.set_page_config(page_title="Dashboard - Retenção de Clientes", page_icon="📉", layout="wide")

REPO = "https://github.com/pedroateixeira/telco-churn"
NOTEBOOKS = f"{REPO}/blob/main/notebooks"
KAGGLE = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"


# --------------------------------------------------------------------------
# Carga (em cache)
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def carregar_base() -> pd.DataFrame:
    return carregar_tratado()


@st.cache_resource(show_spinner=False)
def carregar_modelo():
    """`cache_resource`, não `cache_data`: o pipeline sklearn não é serializável."""
    return joblib.load(RAIZ / "models" / "modelo.joblib")


@st.cache_data(show_spinner=False)
def pontuar_teste() -> pd.DataFrame:
    """Conjunto de teste com a probabilidade prevista.

    Teste, e não a base inteira: o modelo já viu o treino, e simular campanha
    sobre dados memorizados produziria lucro de fantasia.
    """
    _, X_teste, _, y_teste = dividir(carregar_base())
    pontuado = X_teste.copy()
    pontuado["p_churn"] = carregar_modelo().predict_proba(X_teste)[:, 1]
    pontuado[ALVO] = y_teste
    return pontuado


def _pt(coluna: str, valor):
    """Traduz um valor para exibição, mantendo o dado original em inglês."""
    return VALORES_PT.get(coluna, {}).get(valor, valor)


def reais(valor: float, casas: int = 0) -> str:
    inteiro = f"{valor:,.{casas}f}"
    return "R$ " + inteiro.replace(",", "@").replace(".", ",").replace("@", ".")


# --------------------------------------------------------------------------
# Controles de negócio (barra lateral, compartilhados)
# --------------------------------------------------------------------------


def controles_negocio() -> decisao.ParametrosNegocio:
    padrao = decisao.ParametrosNegocio()
    st.sidebar.header("Premissas de negócio")
    st.sidebar.caption(
        "Aqui, fazemos algumas premissas de negócios que não vem diretamente dos "
        "dados, para ilustrar uma situação real."
    )

    # Sliders em pontos percentuais inteiros: com valor fracionário, o format
    # "%.0f%%" do Streamlit imprimiria "0%" para 0,30 — ele não multiplica por 100.
    margem = (
        st.sidebar.slider(
            "Margem de contribuição (%)",
            10,
            60,
            int(padrao.margem * 100),
            1,
            help="Fração da mensalidade que é lucro. Usar a receita cheia "
            "superestimaria o ganho em mais de 3x.",
        )
        / 100
    )
    desconto = (
        st.sidebar.slider(
            "Desconto ofertado (%)",
            1,
            35,
            int(padrao.desconto * 100),
            1,
            help="Fração da mensalidade concedida a quem aceita a oferta",
        )
        / 100
    )
    taxa_aceite = (
        st.sidebar.slider(
            "Taxa de aceite (%)",
            5,
            95,
            int(padrao.taxa_aceite * 100),
            5,
            help="P(aceita | contatado).",
        )
        / 100
    )
    horizonte = st.sidebar.slider("Horizonte (meses)", 1, 36, padrao.horizonte_meses, 1)
    custo_contato = st.sidebar.slider(
        "Custo de contato (R$)",
        0.0,
        100.0,
        padrao.custo_contato,
        1.0,
        help="Custo fixo por cliente abordado, pago mesmo quando ele recusa.",
    )

    params = decisao.ParametrosNegocio(
        margem=margem,
        desconto=desconto,
        taxa_aceite=taxa_aceite,
        horizonte_meses=horizonte,
        custo_contato=custo_contato,
    )

    piso = decisao.piso_do_limiar(params)
    st.sidebar.divider()
    st.sidebar.metric("Piso do limiar", f"{piso:.1%}", help="desconto ÷ margem")
    if piso >= 1.0:
        st.sidebar.error(
            "Desconto ≥ margem: nenhum cliente pode justificar contato, "
            "por melhor que seja o modelo."
        )
    elif piso > 0.6:
        st.sidebar.warning("Piso alto: só clientes de risco extremo se pagam.")
    return params


# --------------------------------------------------------------------------
# Aba 1 — Segmentos de risco
# --------------------------------------------------------------------------


def aba_segmentos(df: pd.DataFrame) -> None:
    st.subheader("Onde o churn se concentra")
    total = f"{len(df):,}".replace(",", ".")
    st.caption(
        f"Churn **observado** por categoria, sobre a base inteira ({total} clientes), "
        "contra a taxa base da carteira. É descrição do passado."
    )

    colunas = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
        "SeniorCitizen",
        "Dependents",
        "Partner",
        "PaperlessBilling",
    ]
    escolha = st.selectbox("Variável", colunas, format_func=lambda c: ROTULOS_PT.get(c, c))

    exibicao = df.copy()
    if escolha in VALORES_PT:
        exibicao[escolha] = exibicao[escolha].map(lambda v: _pt(escolha, v))
    elif exibicao[escolha].dropna().isin([0, 1]).all():
        exibicao[escolha] = exibicao[escolha].map({1: "Sim", 0: "Não"})

    rotulo = ROTULOS_PT.get(escolha, escolha)

    esq, dir_ = st.columns([3, 2])
    with esq:
        st.pyplot(
            graficos.churn_por_categoria(exibicao, escolha, titulo=rotulo),
            use_container_width=True,
        )
    with dir_:
        resumo = (
            exibicao.groupby(escolha, observed=True)[ALVO]
            .agg(Clientes="size", **{"Taxa de churn": "mean"})
            .sort_values("Taxa de churn", ascending=False)
            .rename_axis(rotulo)
        )
        resumo["Taxa de churn"] = resumo["Taxa de churn"].map("{:.1%}".format)
        st.dataframe(resumo, use_container_width=True)
        st.markdown(
            "**Contrato é o sinal mais forte da base**: 42,7% de churn no mês a mês "
            "contra 2,8% em dois anos, uma variação de 15x, maior que a de qualquer "
            "outra variável."
        )


# --------------------------------------------------------------------------
# Aba 2 — Simulador individual
# --------------------------------------------------------------------------


def formulario_cliente() -> dict:
    """Formulário que impõe as implicações que existem na base.

    Um formulário com 19 controles independentes deixaria montar clientes
    impossíveis como sem internet mas com add-ons, por exemplo. O modelo responde
    sem reclamar e a probabilidade muda 15 pontos, mas a combinação não existe
    em nenhuma das 7.043 linhas sendo uma extrapolação com cara de predição.
    """
    cliente = dict(CLIENTE_EXEMPLO)

    st.markdown("##### Contrato e cobrança")
    c1, c2, c3 = st.columns(3)
    with c1:
        cliente["Contract"] = st.selectbox(
            ROTULOS_PT["Contract"],
            list(VALORES_PT["Contract"]),
            format_func=lambda v: _pt("Contract", v),
        )
    with c2:
        cliente["PaymentMethod"] = st.selectbox(
            ROTULOS_PT["PaymentMethod"],
            list(VALORES_PT["PaymentMethod"]),
            format_func=lambda v: _pt("PaymentMethod", v),
        )
    with c3:
        cliente["PaperlessBilling"] = int(st.checkbox(ROTULOS_PT["PaperlessBilling"], value=True))

    c1, c2 = st.columns(2)
    with c1:
        cliente["tenure"] = st.slider(ROTULOS_PT["tenure"], 0, 72, CLIENTE_EXEMPLO["tenure"])
    with c2:
        cliente["MonthlyCharges"] = st.slider(
            ROTULOS_PT["MonthlyCharges"],
            18.25,
            118.75,
            float(CLIENTE_EXEMPLO["MonthlyCharges"]),
            0.05,
        )

    st.markdown("##### Serviços")
    c1, c2 = st.columns(2)
    with c1:
        cliente["InternetService"] = st.selectbox(
            ROTULOS_PT["InternetService"],
            list(VALORES_PT["InternetService"]),
            format_func=lambda v: _pt("InternetService", v),
        )
    with c2:
        cliente["PhoneService"] = int(st.checkbox(ROTULOS_PT["PhoneService"], value=True))

    # Implicação verificada em 682 linhas: sem telefone, não há múltiplas linhas.
    if cliente["PhoneService"]:
        cliente["MultipleLines"] = int(st.checkbox(ROTULOS_PT["MultipleLines"]))
    else:
        cliente["MultipleLines"] = 0
        st.caption("Sem telefone: múltiplas linhas não se aplica.")

    # Implicação verificada em 1.526 linhas: sem internet, nenhum add-on.
    if cliente["InternetService"] == "No":
        for col in COLUNAS_ADDON:
            cliente[col] = 0
        st.caption("Sem internet: os seis serviços adicionais não se aplicam e ficam em 0.")
    else:
        colunas_ui = st.columns(3)
        for i, col in enumerate(COLUNAS_ADDON):
            with colunas_ui[i % 3]:
                cliente[col] = int(st.checkbox(ROTULOS_PT[col], key=f"addon_{col}"))

    st.markdown("##### Perfil")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cliente["gender"] = (
            1
            if st.radio(ROTULOS_PT["gender"], ["Feminino", "Masculino"], horizontal=True)
            == "Masculino"
            else 0
        )
    with c2:
        cliente["SeniorCitizen"] = int(st.checkbox(ROTULOS_PT["SeniorCitizen"]))
    with c3:
        cliente["Partner"] = int(st.checkbox(ROTULOS_PT["Partner"]))
    with c4:
        cliente["Dependents"] = int(st.checkbox(ROTULOS_PT["Dependents"]))

    # TotalCharges é reconstituível de tenure x MonthlyCharges (achado nº 4 da
    # EDA: mediana do resíduo 0,00). Derivar em vez de perguntar evita que os
    # três números entrem incoerentes entre si.
    derivado = round(cliente["tenure"] * cliente["MonthlyCharges"], 2)
    with st.expander(f"Total já pago: {reais(derivado, 2)} — derivado, clique para sobrescrever"):
        st.caption(
            "A EDA mostrou que esta coluna é reconstituível de tempo de casa × "
            "mensalidade (mediana do resíduo 0,00). Sobrescrever só faz sentido "
            "para representar mudança de plano ao longo do contrato."
        )
        cliente["TotalCharges"] = st.number_input(
            ROTULOS_PT["TotalCharges"], 0.0, 9000.0, float(derivado), 10.0
        )
    if "TotalCharges" not in cliente or cliente["TotalCharges"] is None:
        cliente["TotalCharges"] = derivado
    return cliente


def aba_simulador(params: decisao.ParametrosNegocio) -> None:
    st.subheader("Simulador individual")
    st.caption(
        "Monte um cliente e veja a decisão. O formulário impõe as combinações que "
        "existem na base, sem isso, dá para montar cliente impossível e receber "
        "uma probabilidade com cara de confiável."
    )

    esq, dir_ = st.columns([3, 2])
    with esq:
        cliente = formulario_cliente()

    linha = pd.DataFrame([cliente])[list(CLIENTE_EXEMPLO)]
    p = float(carregar_modelo().predict_proba(linha)[0, 1])
    mensalidade = float(cliente["MonthlyCharges"])
    limiar = float(decisao.limiar_por_cliente(np.array([mensalidade]), params)[0])
    ve = float(decisao.valor_esperado(np.array([p]), np.array([mensalidade]), params)[0])

    with dir_:
        st.markdown("##### Decisão")
        st.metric("Probabilidade de churn", f"{p:.1%}")
        st.metric(
            "Limiar deste cliente",
            f"{limiar:.1%}",
            help="Depende da mensalidade: cliente caro compensa ser abordado com "
            "menos certeza. Não existe um limiar único.",
        )
        st.metric("Valor esperado do contato", reais(ve, 2))

        if ve > 0:
            st.success(f"**Contatar.** O contato se paga em {reais(ve, 2)}.")
        else:
            st.error("**Não contatar.** O contato destrói valor com estas premissas.")

        folga = p - limiar
        st.caption(
            f"Probabilidade {'acima' if folga >= 0 else 'abaixo'} do limiar por "
            f"{abs(folga):.1f} ponto percentual."
            if abs(folga) < 0.01
            else f"Probabilidade {'acima' if folga >= 0 else 'abaixo'} do limiar por "
            f"{abs(folga):.1%}."
        )
        st.info(
            "**Dois clientes com a mesma probabilidade podem receber decisões "
            "opostas.** Mude a mensalidade e veja o limiar se mover — é o achado "
            "central da camada de decisão."
        )


# --------------------------------------------------------------------------
# Aba 3 — Campanha
# --------------------------------------------------------------------------


def aba_campanha(params: decisao.ParametrosNegocio) -> None:
    pontuado = pontuar_teste()
    p = pontuado["p_churn"].to_numpy()
    mensalidade = pontuado["MonthlyCharges"].to_numpy()
    mes_a_mes = (pontuado["Contract"] == "Month-to-month").to_numpy()

    st.subheader("Quantos contatar e quais")
    total = f"{len(p):,}".replace(",", ".")
    st.caption(
        f"Simulado sobre os {total} clientes do **conjunto de teste** — dados que o "
        "modelo não viu no treino. A aba de segmentos usa a base inteira porque "
        "descreve o passado; aqui a decisão é sobre o futuro, e só dado fora da "
        "amostra é honesto."
    )

    otima = decisao.campanha_otima(p, mensalidade, params)
    curvas = decisao.comparar_estrategias(p, mensalidade, mes_a_mes, params)
    n = otima["n_contatados"]

    if n == 0:
        st.error(
            f"**Com estas premissas a campanha não se paga para nenhum cliente.** "
            f"O desconto exige probabilidade de churn acima de "
            f"{otima['piso_do_limiar']:.0%}, e ninguém na base chega lá. "
            "O que precisa mudar é o desenho da oferta, não o modelo."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes a contatar", f"{n}", f"{otima['fracao_da_base']:.0%} da base")
    c2.metric("Lucro esperado", reais(otima["lucro_esperado"]))
    c3.metric("Por contato", reais(otima["lucro_por_contato"], 2))
    c4.metric("Limiar mediano", f"{otima['limiar_mediano']:.1%}")

    st.pyplot(graficos.curva_orcamento(curvas, otima), use_container_width=True)

    esq, dir_ = st.columns([2, 3])
    with esq:
        st.markdown("##### No mesmo orçamento, mudando só a ordem da fila")
        comparativo = pd.DataFrame(
            {
                "Estratégia": list(curvas),
                "Lucro esperado": [
                    reais(float(c["lucro_acumulado"].iloc[n - 1])) for c in curvas.values()
                ],
            }
        )
        st.dataframe(comparativo, use_container_width=True, hide_index=True)
        st.caption(
            "A regra de contrato costuma dar **negativo**: acerta o segmento mas "
            "não ordena dentro dele, e o desconto pago a quem ia ficar come o ganho."
        )

    with dir_:
        st.markdown("##### Lista de contato")
        ve = decisao.valor_esperado(p, mensalidade, params)
        lista = pontuado.assign(valor_esperado=ve).nlargest(n, "valor_esperado")
        exibir = lista[["Contract", "tenure", "MonthlyCharges", "p_churn", "valor_esperado"]].copy()
        exibir["Contract"] = exibir["Contract"].map(lambda v: _pt("Contract", v))
        exibir.columns = ["Contrato", "Tempo de casa", "Mensalidade", "P(churn)", "Valor esperado"]
        st.dataframe(
            exibir.style.format(
                {"Mensalidade": "R$ {:.2f}", "P(churn)": "{:.1%}", "Valor esperado": "R$ {:.2f}"}
            ),
            use_container_width=True,
            height=320,
        )
        st.download_button(
            "Baixar lista em CSV",
            lista.to_csv(index=True).encode("utf-8"),
            file_name=f"campanha_{n}_clientes.csv",
            mime="text/csv",
        )

    with st.expander("Sensibilidade às premissas"):
        st.pyplot(
            graficos.sensibilidade_parametros(
                decisao.sensibilidade(p, mensalidade, params), params
            ),
            use_container_width=True,
        )
        st.caption(
            "Horizonte e margem têm a maior amplitude, mas são características do "
            "negócio. Entre o que se controla, o **desconto** é o mais decisivo — "
            "e no sentido contrário à intuição: reduzi-lo aumenta o lucro."
        )


# --------------------------------------------------------------------------
# Aba 4 — Metodologia
# --------------------------------------------------------------------------


def aba_metodologia() -> None:
    figuras = RAIZ / "reports" / "figures"
    st.subheader("Como este número foi construído")

    st.markdown(
        "**1. Sete modelos comparados por validação cruzada, não por um split.** "
        "O conjunto de teste tem 1.409 linhas e o desvio do ROC-AUC nesse tamanho "
        "é ~0,012 — decidir entre sete modelos por um split só seria escolher na "
        "quarta casa decimal de uma amostra. O campeão saiu de uma regra de "
        "desempate declarada em código: entre os que empatam com o líder, fica o "
        "mais barato de treinar."
    )
    st.image(str(figuras / "comparacao_metricas.png"), use_container_width=True)

    st.markdown(
        "**2. A probabilidade precisa estar calibrada, não só bem ordenada.** "
        "`class_weight='balanced'` deixa o ROC-AUC intacto (0,8424 → 0,8418) e "
        "infla a probabilidade média de 0,268 para 0,416, contra taxa base real "
        "de 0,265. Como o valor esperado usa o **nível** da probabilidade e não o "
        "ranking, um projeto que reportasse só AUC entregaria um modelo que infla "
        "o retorno de toda campanha."
    )
    st.image(str(figuras / "efeito_class_weight.png"), use_container_width=True)

    st.markdown(
        "**3. O desenho da oferta importa mais que o modelo.** O limiar tem um "
        "piso igual a `desconto ÷ margem` que nenhuma qualidade de modelo vence. "
        "Um ROC-AUC de 0,99 não salvaria uma campanha cujo desconto consome a "
        "margem inteira."
    )
    st.image(str(figuras / "piso_do_desconto.png"), use_container_width=True)

    st.divider()
    st.markdown(
        "**Análise completa:** "
        f"[EDA]({NOTEBOOKS}/01_eda_telco.ipynb) · "
        f"[Modelagem]({NOTEBOOKS}/02_modelagem.ipynb) · "
        f"[Camada de decisão]({NOTEBOOKS}/03_decisao.ipynb) · "
        f"[Decisões e razões]({REPO}/tree/main/docs) · "
        f"[Código]({REPO})"
    )


# --------------------------------------------------------------------------


def main() -> None:
    st.title("Dashboard de Retenção de clientes: quantos contatar e quais")
    st.markdown(
        "Esse dashboard consiste em um simulador para uma campanha de redução de "
        "churn, objetivando aumentar a retenção. Para isso, avalia os grupos mais "
        "propensos a desistirem de continuar contratando os serviços. Também "
        "adiciona algumas premissas de negócios (fictícias) para incrementar o "
        "processo de avaliação",
    )

    params = controles_negocio()
    df = carregar_base()

    abas = st.tabs(["Segmentos de risco", "Simulador individual", "Campanha", "Metodologia"])
    with abas[0]:
        aba_segmentos(df)
    with abas[1]:
        aba_simulador(params)
    with abas[2]:
        aba_campanha(params)
    with abas[3]:
        aba_metodologia()

    st.sidebar.divider()
    st.sidebar.caption(
        f"Dados: [Telco Customer Churn]({KAGGLE}) (IBM, via Kaggle). Modelo: XGBoost calibrado."
    )


if __name__ == "__main__":
    main()
