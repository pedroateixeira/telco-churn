"""Ponto de entrada do treino: um comando treina do zero e salva o artefato.

    uv run python -m telco_churn.treinar

Notebook explora; script entrega. Só o segundo pode ser testado e implantado.
"""

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from telco_churn import __version__
from telco_churn.dados import RAIZ_PROJETO, caminho_padrao, carregar_tratado
from telco_churn.features import nomes_features
from telco_churn.modelo import (
    SEED_PADRAO,
    TEST_SIZE_PADRAO,
    avaliar,
    construir_pipeline,
    dividir,
    lift_por_decil,
)

SAIDA_PADRAO = RAIZ_PROJETO / "models"


def _coeficientes(pipeline) -> dict[str, float]:
    """Coeficientes do modelo linear, nomeados. Vazio para estimadores sem `coef_`."""
    estimador = pipeline.named_steps["clf"]
    if not hasattr(estimador, "coef_"):
        return {}
    nomes = nomes_features(pipeline.named_steps["pre"])
    return {
        nome: float(valor)
        for nome, valor in zip(nomes, estimador.coef_[0], strict=True)
    }


def _formatar_metricas(metricas: dict[str, float | int]) -> str:
    linhas = [
        f"  ROC-AUC    {metricas['roc_auc']:.4f}",
        f"  PR-AUC     {metricas['pr_auc']:.4f}",
        f"  Brier      {metricas['brier']:.4f}",
        f"  taxa base  {metricas['taxa_base']:.4f}   (n = {metricas['n']})",
    ]
    return "\n".join(linhas)


def treinar(
    usar_total_charges: bool = True,
    seed: int = SEED_PADRAO,
    saida: Path = SAIDA_PADRAO,
    caminho_dados: Path | None = None,
    test_size: float = TEST_SIZE_PADRAO,
) -> dict:
    """Carrega, valida, trata, divide, treina, avalia e salva."""
    print(f"Lendo {caminho_dados or caminho_padrao()}")
    df = carregar_tratado(caminho_dados)
    print(f"  {df.shape[0]} clientes, {df.shape[1] - 1} features + alvo")

    X_treino, X_teste, y_treino, y_teste = dividir(df, test_size=test_size, seed=seed)
    print(f"  treino {len(X_treino)} | teste {len(X_teste)}")
    if not usar_total_charges:
        print("  ablação: TotalCharges fora do modelo")

    # A Fase 2 entrega um estimador só. A escada completa — regra de uma linha,
    # logística, boosting — é a Fase 3; aqui o objetivo é o encanamento rodando
    # de ponta a ponta, não escolher o campeão.
    #
    # Sem random_state: o solver lbfgs é determinístico, e o parâmetro só teria
    # efeito com sag/saga/liblinear. Passá-lo sugeriria uma escolha aleatória
    # que não existe — a reprodutibilidade aqui vem do seed do split.
    pipeline = construir_pipeline(
        LogisticRegression(max_iter=1000),
        usar_total_charges=usar_total_charges,
    )
    pipeline.fit(X_treino, y_treino)

    metricas_teste = avaliar(pipeline, X_teste, y_teste)
    metricas_treino = avaliar(pipeline, X_treino, y_treino)

    print("\nTeste:")
    print(_formatar_metricas(metricas_teste))
    print("\nTreino:")
    print(_formatar_metricas(metricas_treino))

    probabilidades = pipeline.predict_proba(X_teste)[:, 1]
    lift = lift_por_decil(y_teste, probabilidades)
    print("\nLift por decil de risco (teste):")
    print(lift.round(3).to_string(index=False))

    saida.mkdir(parents=True, exist_ok=True)

    # O nome do arquivo reflete a variante: sem isto, rodar a ablação para
    # conferir o achado nº 4 da EDA substituiria o modelo de produção pelo do
    # experimento, e o dashboard passaria a servir a ablação sem nenhum sinal.
    sufixo = "" if usar_total_charges else "_sem_total_charges"
    caminho_modelo = saida / f"modelo{sufixo}.joblib"
    joblib.dump(pipeline, caminho_modelo)

    preprocessador = pipeline.named_steps["pre"]
    relatorio = {
        "versao_pacote": __version__,
        "treinado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "test_size": test_size,
        "usar_total_charges": usar_total_charges,
        "modelo": type(pipeline.named_steps["clf"]).__name__,
        # joblib grava um pickle de objetos sklearn. Carregar isso com outra
        # versão da biblioteca dá, no melhor caso, InconsistentVersionWarning;
        # no pior, um objeto que desserializa e prediz diferente. Registrar o
        # ambiente é o que torna esse desencontro diagnosticável.
        "versoes": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "colunas_entrada": list(X_treino.columns),
        "n_features": len(nomes_features(preprocessador)),
        "features": nomes_features(preprocessador),
        "teste": metricas_teste,
        "treino": metricas_treino,
        "lift_por_decil": lift.astype({"decil": int}).to_dict(orient="records"),
        # Ressalva de leitura: as numéricas passaram por StandardScaler e as
        # binárias vieram por passthrough, então os coeficientes estão em
        # unidades diferentes — "por desvio-padrão de tenure" contra "de 0 para
        # 1 em TechSupport". Cada um é interpretável no seu grupo; ordenar por
        # abs(coef) misturando os dois seria enganoso.
        "coeficientes": _coeficientes(pipeline),
    }
    caminho_metricas = saida / f"metricas{sufixo}.json"
    # Sem `default=str`: ele não protegia nada aqui (o pandas já devolve tipos
    # nativos em to_dict) e, se algum dia aparecer um valor não serializável,
    # o que se quer é o TypeError, não uma string silenciosa no lugar do número.
    caminho_metricas.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nModelo salvo em   {caminho_modelo}")
    print(f"Métricas salvas em {caminho_metricas}")
    # O JSON existe para que README e dashboard citem o número de uma fonte só,
    # em vez de valor copiado à mão que envelhece calado.
    return relatorio


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m telco_churn.treinar",
        description="Treina o modelo de churn e salva o artefato.",
    )
    parser.add_argument(
        "--sem-total-charges",
        action="store_true",
        help="Treina sem TotalCharges (ablação do achado nº 4 da EDA).",
    )
    parser.add_argument("--seed", type=int, default=SEED_PADRAO, help="Semente do split.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE_PADRAO,
        help="Fração da base reservada para teste.",
    )
    parser.add_argument(
        "--dados", type=Path, default=None, help="CSV bruto alternativo (padrão: data/raw/)."
    )
    parser.add_argument(
        "--saida", type=Path, default=SAIDA_PADRAO, help="Onde salvar os artefatos."
    )
    args = parser.parse_args(argv)

    treinar(
        usar_total_charges=not args.sem_total_charges,
        seed=args.seed,
        saida=args.saida,
        caminho_dados=args.dados,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
