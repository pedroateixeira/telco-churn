"""Camada de decisão: de probabilidade de churn para lista de campanha.

    uv run python -m telco_churn.decisao

A Fase 3 entregou um modelo calibrado e uma frase que ainda não é decisão —
"lift de 2,89 no primeiro decil". Nenhum gerente de retenção sabe o que fazer
com isso. As perguntas reais são *quantos* clientes contatar e *quais*, e
nenhuma métrica estatística responde: F1, ponto de Youden e joelho da curva ROC
escolhem um limiar sem nunca perguntar quanto custa a oferta nem quanto vale o
cliente.

Aqui o limiar sai de valor esperado.
"""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # sem display: roda em terminal e em CI

import joblib  # noqa: E402

from telco_churn import graficos  # noqa: E402
from telco_churn.dados import RAIZ_PROJETO, carregar_tratado  # noqa: E402
from telco_churn.modelo import SEED_PADRAO, dividir  # noqa: E402

FIGURAS_PADRAO = RAIZ_PROJETO / "reports" / "figures"
SAIDA_PADRAO = RAIZ_PROJETO / "models"


@dataclass(frozen=True)
class ParametrosNegocio:
    """As suposições de negócio, reunidas e nomeadas.

    Nenhuma delas vem dos dados — são premissas, e é por isso que estão num
    objeto próprio, entram no relatório e passam por análise de sensibilidade.
    Escondê-las dentro do código as faria parecer fatos.
    """

    margem: float = 0.30  # margem de contribuição sobre a mensalidade
    desconto: float = 0.10  # desconto ofertado, como fração da mensalidade
    taxa_aceite: float = 0.40  # P(aceita a oferta | contatado)
    horizonte_meses: int = 12  # por quanto tempo valem retenção e desconto
    custo_contato: float = 15.0  # custo fixo por cliente abordado


def valor_esperado(
    p: np.ndarray, mensalidade: np.ndarray, params: ParametrosNegocio
) -> np.ndarray:
    """Valor esperado de contatar cada cliente.

    Três componentes:

    | Componente         | Quando ocorre                        | Valor                   |
    |--------------------|--------------------------------------|-------------------------|
    | Margem preservada  | ia cancelar (p) E aceitou (a)        | p · a · margem · mc · H |
    | Custo de contato   | sempre que se aborda                 | custo_contato           |
    | Custo do desconto  | aceitou (a), tendo ou não intenção   | a · desconto · mc · H   |
                           de sair

    A terceira linha é a que a formulação ingênua omite, e é a mais importante:
    **o desconto vai também para quem não ia cancelar**. É o custo real de toda
    campanha de retenção, e ignorá-lo infla o resultado.

    Agrupando:  VE = a · mc · H · (p · margem − desconto) − custo_contato
    """
    p = np.asarray(p, dtype=float)
    mensalidade = np.asarray(mensalidade, dtype=float)
    escala = params.taxa_aceite * mensalidade * params.horizonte_meses
    return escala * (p * params.margem - params.desconto) - params.custo_contato


def piso_do_limiar(params: ParametrosNegocio) -> float:
    """A probabilidade de churn abaixo da qual nenhum contato se paga.

    Isolando p em VE > 0:

        p* = desconto/margem + custo_contato / (a · margem · mc · H)
             └── piso ───┘     └──── termo por cliente ────┘

    O primeiro termo não depende do cliente, do horizonte, da taxa de aceite nem
    da qualidade do modelo — só da razão entre o desconto e a margem. Com
    desconto igual à margem o piso vai a 1,0 e **nenhum cliente jamais justifica
    contato**, por melhor que o modelo seja. É um resultado sobre o desenho da
    oferta, e é a primeira coisa a checar antes de qualquer campanha.
    """
    return params.desconto / params.margem


def limiar_por_cliente(
    mensalidade: np.ndarray, params: ParametrosNegocio
) -> np.ndarray:
    """O limiar de contato, que é diferente para cada cliente.

    O termo por cliente cai com a mensalidade: cliente caro compensa ser
    abordado com muito menos certeza de que vai sair. Por isso não existe *um*
    limiar — e por isso ordenar por valor esperado não é ordenar por p.
    """
    mensalidade = np.asarray(mensalidade, dtype=float)
    denominador = params.taxa_aceite * params.margem * mensalidade * params.horizonte_meses
    return piso_do_limiar(params) + params.custo_contato / denominador


def simular_campanha(
    p: np.ndarray,
    mensalidade: np.ndarray,
    params: ParametrosNegocio,
    ordem: np.ndarray | None = None,
) -> pd.DataFrame:
    """Lucro esperado acumulado conforme se contata mais gente.

    `ordem` é a fila de contato — os índices dos clientes, do primeiro ao
    último. Trocando só a ordem, com a **mesma** conta de valor esperado, dá
    para comparar estratégias de segmentação e isolar o ganho da camada de
    decisão do ganho do modelo. Sem `ordem`, ordena pelo próprio VE, que é o
    ótimo.
    """
    ve = valor_esperado(p, mensalidade, params)
    if ordem is None:
        ordem = np.argsort(-ve)
    ordem = np.asarray(ordem)

    return pd.DataFrame(
        {
            "contatados": np.arange(1, len(ordem) + 1),
            "lucro_acumulado": np.cumsum(ve[ordem]),
            "ve_marginal": ve[ordem],
        }
    )


def campanha_otima(
    p: np.ndarray, mensalidade: np.ndarray, params: ParametrosNegocio
) -> dict:
    """O ponto onde contatar mais um cliente passa a destruir valor.

    Contatar todo mundo com VE > 0 e mais ninguém. Como a fila está ordenada por
    VE decrescente, esse ponto é o último índice com VE positivo — não é preciso
    otimizar nada, e é por isso que o teste de verificação compara este
    resultado com uma varredura em força bruta.
    """
    ve = valor_esperado(p, mensalidade, params)
    selecionados = ve > 0
    n = int(selecionados.sum())

    return {
        "n_contatados": n,
        "fracao_da_base": float(n / len(ve)) if len(ve) else 0.0,
        "lucro_esperado": float(ve[selecionados].sum()) if n else 0.0,
        "lucro_por_contato": float(ve[selecionados].mean()) if n else 0.0,
        "piso_do_limiar": piso_do_limiar(params),
        "limiar_mediano": float(np.median(limiar_por_cliente(mensalidade, params))),
        "selecionados": selecionados,
    }


def comparar_estrategias(
    p: np.ndarray,
    mensalidade: np.ndarray,
    contrato_mes_a_mes: np.ndarray,
    params: ParametrosNegocio,
    seed: int = SEED_PADRAO,
) -> dict[str, pd.DataFrame]:
    """As mesmas contas, quatro filas de contato diferentes.

    A comparação isola o que cada camada acrescenta: a regra de contrato é o que
    dá para fazer sem modelo nenhum; ordenar por `p` é o que o modelo entrega
    sozinho; ordenar por VE é o que a camada de decisão acrescenta ao modelo.
    A aleatória é o piso — o que se consegue sem informação alguma.
    """
    ve = valor_esperado(p, mensalidade, params)
    aleatoria = np.random.default_rng(seed).permutation(len(p))

    # A regra de contrato não ordena dentro dos grupos: quem é mês a mês vem
    # primeiro, na ordem em que aparece. É a limitação real de uma regra de uma
    # linha, e não faria sentido dar a ela um desempate que ela não tem.
    ordem_contrato = np.argsort(-np.asarray(contrato_mes_a_mes, dtype=float), kind="stable")

    ordens = {
        "Valor esperado": np.argsort(-ve),
        "Probabilidade de churn": np.argsort(-np.asarray(p)),
        "Regra de contrato": ordem_contrato,
        "Aleatória": aleatoria,
    }
    return {
        nome: simular_campanha(p, mensalidade, params, ordem)
        for nome, ordem in ordens.items()
    }


def sensibilidade(
    p: np.ndarray,
    mensalidade: np.ndarray,
    params: ParametrosNegocio,
    grade: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Como o resultado ótimo se move quando cada suposição varia.

    Um parâmetro por vez, os demais no padrão. Não é análise de interação — é a
    checagem mínima de quais premissas o resultado tolera e quais o dominam.
    """
    grade = grade or {
        "margem": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        "desconto": [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "taxa_aceite": [0.10, 0.20, 0.30, 0.40, 0.60, 0.80],
        "horizonte_meses": [3, 6, 12, 18, 24, 36],
        "custo_contato": [0.0, 5.0, 15.0, 30.0, 50.0, 100.0],
    }

    linhas = []
    for parametro, valores in grade.items():
        for valor in valores:
            variante = replace(params, **{parametro: valor})
            otima = campanha_otima(p, mensalidade, variante)
            linhas.append(
                {
                    "parametro": parametro,
                    "valor": valor,
                    "n_contatados": otima["n_contatados"],
                    "lucro_esperado": otima["lucro_esperado"],
                    "piso_do_limiar": otima["piso_do_limiar"],
                    "e_o_padrao": valor == getattr(params, parametro),
                }
            )
    return pd.DataFrame(linhas)


def _reais(valor: float) -> str:
    """Formata em padrão brasileiro: milhar com ponto, decimal com vírgula."""
    return f"{valor:,.0f}".replace(",", ".")


def frase_de_negocio(otima: dict, curvas: dict[str, pd.DataFrame], n_base: int) -> str:
    """O resultado traduzido para a linguagem de quem decide o orçamento."""
    n = otima["n_contatados"]
    if n == 0:
        return (
            "Com estes parâmetros a campanha não se paga para nenhum cliente: o desconto "
            f"ofertado exige probabilidade de churn acima de {otima['piso_do_limiar']:.0%}, "
            "e o desenho da oferta precisa mudar antes do modelo."
        )

    lucro_ve = otima["lucro_esperado"]
    lucro_contrato = float(curvas["Regra de contrato"]["lucro_acumulado"].iloc[n - 1])
    # Só os números são reformatados; um replace na frase inteira comeria a
    # vírgula da própria sentença.
    return (
        f"Contatando os {n} clientes de maior valor esperado ({n / n_base:.0%} da base), "
        f"a campanha rende R$ {_reais(lucro_ve)} de lucro esperado — contra "
        f"R$ {_reais(lucro_contrato)} da regra de contrato no mesmo orçamento."
    )


def executar(
    params: ParametrosNegocio | None = None,
    seed: int = SEED_PADRAO,
    figuras: Path = FIGURAS_PADRAO,
    saida: Path = SAIDA_PADRAO,
) -> dict:
    params = params or ParametrosNegocio()

    df = carregar_tratado()
    # O conjunto de teste, sempre: simular campanha sobre dados que o modelo já
    # viu no treino produziria lucro de fantasia.
    _, X_teste, _, y_teste = dividir(df, seed=seed)
    modelo = joblib.load(saida / "modelo.joblib")

    p = modelo.predict_proba(X_teste)[:, 1]
    mensalidade = X_teste["MonthlyCharges"].to_numpy()
    mes_a_mes = (X_teste["Contract"] == "Month-to-month").to_numpy()

    print(f"Base de simulação: {len(p)} clientes do conjunto de teste")
    print(f"Parâmetros: {asdict(params)}\n")

    piso = piso_do_limiar(params)
    limiares = limiar_por_cliente(mensalidade, params)
    print(f"Piso do limiar (desconto/margem): {piso:.3f}")
    print(f"Limiar por cliente: mediana {np.median(limiares):.3f}, "
          f"faixa {limiares.min():.3f} a {limiares.max():.3f}")

    otima = campanha_otima(p, mensalidade, params)
    print(f"\nCampanha ótima: {otima['n_contatados']} clientes "
          f"({otima['fracao_da_base']:.1%} da base)")
    print(f"  lucro esperado    R$ {otima['lucro_esperado']:,.0f}")
    print(f"  por contato       R$ {otima['lucro_por_contato']:,.2f}")

    curvas = comparar_estrategias(p, mensalidade, mes_a_mes, params, seed=seed)
    n = otima["n_contatados"]
    if n:
        print("\nNo mesmo orçamento, por estratégia de fila:")
        for nome, curva in curvas.items():
            print(f"  {nome:24s} R$ {curva['lucro_acumulado'].iloc[n - 1]:>9,.0f}")

    # Quanto a ordenação por VE difere da ordenação por probabilidade
    ve = valor_esperado(p, mensalidade, params)
    corte = max(n, 1)
    coincidem = len(set(np.argsort(-ve)[:corte]) & set(np.argsort(-p)[:corte]))
    print(f"\nTop-{corte} por VE vs por p: {coincidem} em comum "
          f"({coincidem / corte:.0%}) — {corte - coincidem} clientes trocam de lista")

    tabela_sens = sensibilidade(p, mensalidade, params)

    figuras.mkdir(parents=True, exist_ok=True)
    saidas = {
        "limiar_por_cliente.png": graficos.limiar_por_cliente(
            p, mensalidade, limiares, otima["selecionados"], piso
        ),
        "curva_orcamento.png": graficos.curva_orcamento(curvas, otima),
        "sensibilidade.png": graficos.sensibilidade_parametros(tabela_sens, params),
        "piso_do_desconto.png": graficos.piso_do_desconto(tabela_sens, params),
    }
    for nome_arquivo, figura in saidas.items():
        figura.savefig(figuras / nome_arquivo, dpi=140, bbox_inches="tight")
        print(f"  figura: {figuras / nome_arquivo}")

    frase = frase_de_negocio(otima, curvas, len(p))
    print(f"\n{frase}")

    relatorio = {
        "parametros": asdict(params),
        "seed": seed,
        "n_base_simulacao": len(p),
        "taxa_base_teste": float(np.mean(y_teste)),
        "piso_do_limiar": piso,
        "limiar_mediano": otima["limiar_mediano"],
        "limiar_min": float(limiares.min()),
        "limiar_max": float(limiares.max()),
        "campanha_otima": {k: v for k, v in otima.items() if k != "selecionados"},
        "coincidencia_ve_vs_p": {
            "corte": int(corte),
            "em_comum": int(coincidem),
            "trocam": int(corte - coincidem),
        },
        "curvas": {
            nome: curva[["contatados", "lucro_acumulado"]].to_dict(orient="records")
            for nome, curva in curvas.items()
        },
        "lucro_no_orcamento_otimo": {
            nome: float(curva["lucro_acumulado"].iloc[n - 1]) if n else 0.0
            for nome, curva in curvas.items()
        },
        "sensibilidade": tabela_sens.to_dict(orient="records"),
        "frase_de_negocio": frase,
    }
    caminho = saida / "decisao.json"
    caminho.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nDecisão salva em {caminho}")
    return relatorio


def main(argv: list[str] | None = None) -> None:
    padrao = ParametrosNegocio()
    parser = argparse.ArgumentParser(
        prog="python -m telco_churn.decisao",
        description="Traduz probabilidade de churn em lista de campanha por valor esperado.",
    )
    parser.add_argument("--margem", type=float, default=padrao.margem)
    parser.add_argument("--desconto", type=float, default=padrao.desconto)
    parser.add_argument("--taxa-aceite", type=float, default=padrao.taxa_aceite)
    parser.add_argument("--horizonte", type=int, default=padrao.horizonte_meses)
    parser.add_argument("--custo-contato", type=float, default=padrao.custo_contato)
    parser.add_argument("--seed", type=int, default=SEED_PADRAO)
    parser.add_argument("--figuras", type=Path, default=FIGURAS_PADRAO)
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    args = parser.parse_args(argv)

    executar(
        params=ParametrosNegocio(
            margem=args.margem,
            desconto=args.desconto,
            taxa_aceite=args.taxa_aceite,
            horizonte_meses=args.horizonte,
            custo_contato=args.custo_contato,
        ),
        seed=args.seed,
        figuras=args.figuras,
        saida=args.saida,
    )


if __name__ == "__main__":
    main()
