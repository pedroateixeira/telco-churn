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

from telco_churn import __version__
from telco_churn.catalogo import catalogo
from telco_churn.dados import RAIZ_PROJETO, caminho_padrao, carregar_tratado
from telco_churn.features import nomes_features
from telco_churn.modelo import (
    SEED_PADRAO,
    TEST_SIZE_PADRAO,
    avaliar,
    dividir,
    lift_por_decil,
)

SAIDA_PADRAO = RAIZ_PROJETO / "models"


def modelo_campeao(saida: Path = SAIDA_PADRAO) -> str:
    """Nome do modelo escolhido pela comparação da Fase 3.

    Lê `models/comparacao.json` em vez de trazer o nome escrito no código: quem
    decide o campeão é a validação cruzada, e o treino apenas persiste a
    decisão. Se a comparação ainda não rodou, cai na regressão logística, que é
    o modelo mais simples da escada.
    """
    caminho = saida / "comparacao.json"
    if not caminho.exists():
        return "Regressão logística"
    return json.loads(caminho.read_text(encoding="utf-8"))["campeao_cv"]


def _coeficientes(modelo) -> dict[str, float]:
    """Coeficientes do modelo linear, nomeados. Vazio para quem não tem `coef_`."""
    passos = getattr(modelo, "named_steps", None)
    if passos is None or "clf" not in passos:
        return {}
    estimador = passos["clf"]
    if not hasattr(estimador, "coef_"):
        return {}
    nomes = nomes_features(passos["pre"])
    return {
        nome: float(valor) for nome, valor in zip(nomes, estimador.coef_[0], strict=True)
    }


def _n_features(modelo, X) -> int:
    passos = getattr(modelo, "named_steps", None)
    if passos is None or "pre" not in passos:
        return X.shape[1]
    return len(nomes_features(passos["pre"]))


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
    modelo: str | None = None,
) -> dict:
    """Carrega, valida, trata, divide, treina, avalia e salva."""
    print(f"Lendo {caminho_dados or caminho_padrao()}")
    df = carregar_tratado(caminho_dados)
    print(f"  {df.shape[0]} clientes, {df.shape[1] - 1} features + alvo")

    X_treino, X_teste, y_treino, y_teste = dividir(df, test_size=test_size, seed=seed)
    print(f"  treino {len(X_treino)} | teste {len(X_teste)}")
    if not usar_total_charges:
        print("  ablação: TotalCharges fora do modelo")

    nome_modelo = modelo or modelo_campeao(saida)
    fabricas = catalogo(usar_total_charges=usar_total_charges, seed=seed)
    if nome_modelo not in fabricas:
        raise SystemExit(
            f"modelo desconhecido: {nome_modelo!r}. Disponíveis: {list(fabricas)}"
        )
    print(f"  modelo: {nome_modelo}")

    pipeline = fabricas[nome_modelo]()
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

    relatorio = {
        "versao_pacote": __version__,
        "treinado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "test_size": test_size,
        "usar_total_charges": usar_total_charges,
        "modelo": nome_modelo,
        "estimador": type(getattr(pipeline, "named_steps", {"clf": pipeline})["clf"]).__name__,
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
        "n_features": _n_features(pipeline, X_treino),
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
    parser.add_argument(
        "--modelo",
        default=None,
        help="Modelo do catálogo (padrão: o campeão registrado em comparacao.json).",
    )
    args = parser.parse_args(argv)

    treinar(
        usar_total_charges=not args.sem_total_charges,
        seed=args.seed,
        saida=args.saida,
        caminho_dados=args.dados,
        test_size=args.test_size,
        modelo=args.modelo,
    )


if __name__ == "__main__":
    main()
