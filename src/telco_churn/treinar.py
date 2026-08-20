"""Ponto de entrada do treino: um comando treina do zero e salva o artefato.

    uv run python -m telco_churn.treinar

Notebook explora; script entrega. Só o segundo pode ser testado e implantado.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from telco_churn import __version__
from telco_churn.dados import caminho_padrao, carregar_tratado
from telco_churn.modelo import (
    SEED_PADRAO,
    avaliar,
    construir_pipeline,
    dividir,
    lift_por_decil,
)

SAIDA_PADRAO = Path(__file__).resolve().parents[2] / "models"


def _formatar_metricas(metricas: dict[str, float]) -> str:
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
) -> dict:
    """Carrega, valida, trata, divide, treina, avalia e salva."""
    print(f"Lendo {caminho_dados or caminho_padrao()}")
    df = carregar_tratado(caminho_dados)
    print(f"  {df.shape[0]} clientes, {df.shape[1] - 1} features + alvo")

    X_treino, X_teste, y_treino, y_teste = dividir(df, seed=seed)
    print(f"  treino {len(X_treino)} | teste {len(X_teste)}")
    if not usar_total_charges:
        print("  ablação: TotalCharges fora do modelo")

    # A Fase 2 entrega um estimador só. A escada completa — regra de uma linha,
    # logística, boosting — é a Fase 3; aqui o objetivo é o encanamento rodando
    # de ponta a ponta, não escolher o campeão.
    pipeline = construir_pipeline(
        LogisticRegression(max_iter=1000, random_state=seed),
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
    caminho_modelo = saida / "modelo.joblib"
    joblib.dump(pipeline, caminho_modelo)

    relatorio = {
        "versao_pacote": __version__,
        "treinado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "usar_total_charges": usar_total_charges,
        "modelo": type(pipeline.named_steps["clf"]).__name__,
        "n_features": int(pipeline[:-1].transform(X_treino).shape[1]),
        "teste": metricas_teste,
        "treino": metricas_treino,
        "lift_por_decil": lift.to_dict(orient="records"),
    }
    caminho_metricas = saida / "metricas.json"
    caminho_metricas.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False, default=str) + "\n",
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
    parser.add_argument(
        "--seed", type=int, default=SEED_PADRAO, help="Semente do split e do modelo."
    )
    parser.add_argument(
        "--saida", type=Path, default=SAIDA_PADRAO, help="Onde salvar os artefatos."
    )
    args = parser.parse_args(argv)

    treinar(
        usar_total_charges=not args.sem_total_charges,
        seed=args.seed,
        saida=args.saida,
    )


if __name__ == "__main__":
    main()
