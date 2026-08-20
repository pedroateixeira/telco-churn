"""Gráficos de comparação entre modelos.

Todas as funções recebem dados já calculados e devolvem uma `Figure`. Nenhuma
delas treina nada nem salva arquivo: quem orquestra é `comparacao.py`, e manter
essa separação é o que permite reusar os mesmos gráficos no notebook e no
dashboard sem retreinar.

Duas decisões de forma valem explicação, porque as duas contam o resultado:

1. **Ponto com barra de erro, não barra.** Barra pressupõe linha de base no
   zero, e no zero todos os modelos aprendidos viram a mesma barra de 0,85 —
   some exatamente a diferença que se quer ler. Ponto não promete zero, então
   a escala pode focar a faixa onde a decisão acontece.

2. **Ênfase, não sete cores.** O achado desta fase é que os modelos aprendidos
   empatam entre si. Sete curvas coloridas quase sobrepostas pedem que o leitor
   distinga o que é indistinguível; destacar o campeão e o baseline, com o
   resto em cinza, é a forma que corresponde à conclusão.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

# Paleta validada da referência de data viz (modo claro).
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"
MUTADO = "#898781"
GRADE = "#e1e0d9"

AZUL = "#2a78d6"      # slot 1 — o campeão
LARANJA = "#eb6834"   # slot 2 — o modelo que serve de contraexemplo
AGUA = "#1baf7a"      # slot 3
CINZA_FUNDO = "#c9c8c2"  # os demais modelos, quando são contexto

BASE = {
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "axes.edgecolor": GRADE,
    "axes.labelcolor": TINTA_SUAVE,
    "text.color": TINTA,
    "xtick.color": MUTADO,
    "ytick.color": MUTADO,
    "grid.color": GRADE,
    "axes.grid": True,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


def _estilo():
    return plt.rc_context(BASE)


def _limpar(ax, eixo_grade: str = "x"):
    ax.grid(axis=eixo_grade, linewidth=0.8, color=GRADE)
    ax.set_axisbelow(True)


def comparacao_metricas(
    resumo: pd.DataFrame, campeao: str, baseline: str = "Regra de contrato"
) -> Figure:
    """Três painéis — ROC-AUC, PR-AUC e Brier — com o intervalo da validação cruzada.

    A barra de erro é o conteúdo do gráfico, não enfeite: ela mostra quais
    diferenças entre modelos são reais e quais cabem dentro do ruído do split.

    O baseline sai das linhas e vira uma marca de referência no eixo. Ele está
    tão longe (PR-AUC 0,41 contra 0,66) que, mantido como ponto, esticaria a
    escala e comprimiria num décimo do gráfico justamente a comparação que se
    quer ler. São duas perguntas diferentes — "os modelos batem a regra?" e "os
    modelos diferem entre si?" —, e a marca responde a primeira sem sacrificar
    a segunda.
    """
    metricas = [
        ("roc_auc", "ROC-AUC", "maior é melhor"),
        ("pr_auc", "PR-AUC", "maior é melhor"),
        ("brier", "Brier", "menor é melhor"),
    ]
    aprendidos = resumo[resumo["modelo"] != baseline]
    referencia = resumo[resumo["modelo"] == baseline]

    with _estilo():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

        for ax, (chave, titulo, sentido) in zip(axes, metricas, strict=True):
            dados = aprendidos.sort_values(f"{chave}_media", ascending=(chave == "brier"))
            media = dados[f"{chave}_media"].to_numpy()
            desvio = dados[f"{chave}_desvio"].to_numpy()
            y = np.arange(len(dados))

            cores = [AZUL if nome == campeao else MUTADO for nome in dados["modelo"]]

            ax.errorbar(
                media, y, xerr=desvio, fmt="none", ecolor=CINZA_FUNDO, elinewidth=2, capsize=0
            )
            ax.scatter(media, y, s=64, color=cores, zorder=3, edgecolor=SUPERFICIE, linewidth=1.5)

            for i, valor in enumerate(media):
                ax.text(valor, i - 0.34, f"{valor:.3f}", ha="center", fontsize=8.5,
                        color=TINTA_SUAVE)

            ax.set_yticks(y, dados["modelo"], fontsize=9)
            ax.invert_yaxis()
            # Folga acima da primeira linha: o rótulo dela fica sobre o ponto e,
            # sem isso, encosta no subtítulo.
            ax.set_ylim(len(dados) - 0.4, -0.85)

            margem = (media.max() - media.min()) * 0.18 + desvio.max()
            ax.set_xlim(media.min() - margem, media.max() + margem)
            _limpar(ax)

            # O valor do baseline vai no subtítulo, e não como anotação dentro
            # dos eixos: fora da escala ele colidiria com os rótulos do eixo x,
            # e dentro dela esticaria a escala. No subtítulo não faz nem um nem
            # outro, e continua respondendo "os modelos batem a regra?".
            subtitulo = f"{titulo}  ·  {sentido}"
            if not referencia.empty:
                valor_base = float(referencia[f"{chave}_media"].iloc[0])
                inicio, fim = ax.get_xlim()
                dentro = inicio <= valor_base <= fim
                if dentro:
                    ax.axvline(valor_base, color=MUTADO, lw=1.2, ls=(0, (4, 3)), zorder=0)
                marca = "linha tracejada" if dentro else "fora da escala"
                subtitulo += f"\n{baseline}: {valor_base:.3f} ({marca})"
            ax.set_title(subtitulo, fontsize=10.5, color=TINTA, pad=10)

        fig.suptitle(
            "Comparação de modelos — validação cruzada 5×3 sobre o treino",
            fontsize=13,
            y=1.04,
            color=TINTA,
        )
        fig.tight_layout()
    return fig


def _sobrepostas(
    curvas: dict[str, tuple[np.ndarray, np.ndarray]],
    campeao: str,
    baseline: str,
    calcular,
    ax,
):
    """Desenha o campeão e o baseline em destaque; o resto em cinza, como faixa."""
    rotulado_cinza = False
    for nome, (y, p) in curvas.items():
        if nome in (campeao, baseline):
            continue
        x_vals, y_vals = calcular(y, p)
        ax.plot(
            x_vals,
            y_vals,
            color=CINZA_FUNDO,
            lw=1.4,
            zorder=1,
            label="Demais modelos aprendidos" if not rotulado_cinza else None,
        )
        rotulado_cinza = True

    y, p = curvas[baseline]
    x_vals, y_vals = calcular(y, p)
    ax.plot(x_vals, y_vals, color=MUTADO, lw=2, zorder=2, label=baseline)

    y, p = curvas[campeao]
    x_vals, y_vals = calcular(y, p)
    ax.plot(x_vals, y_vals, color=AZUL, lw=2.4, zorder=3, label=f"{campeao} (campeão)")


def curvas_roc(
    curvas: dict[str, tuple[np.ndarray, np.ndarray]],
    taxa_base: float,
    campeao: str,
    baseline: str = "Regra de contrato",
) -> Figure:
    """ROC no teste. Campeão e baseline em destaque, os demais como faixa cinza."""

    def calcular(y, p):
        fpr, tpr, _ = roc_curve(y, p)
        return fpr, tpr

    with _estilo():
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], color=GRADE, lw=1.5, zorder=0, label="Acaso")
        _sobrepostas(curvas, campeao, baseline, calcular, ax)

        ax.set_xlabel("Taxa de falso positivo")
        ax.set_ylabel("Taxa de verdadeiro positivo")
        ax.set_title(
            f"Curvas ROC no teste  ·  taxa base {taxa_base:.1%}",
            fontsize=12,
            color=TINTA,
            pad=10,
        )
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        _limpar(ax, "both")
        fig.tight_layout()
    return fig


def curvas_precisao_recall(
    curvas: dict[str, tuple[np.ndarray, np.ndarray]],
    taxa_base: float,
    campeao: str,
    baseline: str = "Regra de contrato",
) -> Figure:
    """Precisão-recall no teste. Com classe desbalanceada, é esta curva que importa."""

    def calcular(y, p):
        precisao, recall, _ = precision_recall_curve(y, p)
        return recall, precisao

    with _estilo():
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.axhline(taxa_base, color=GRADE, lw=1.5, zorder=0, label=f"Acaso ({taxa_base:.1%})")
        _sobrepostas(curvas, campeao, baseline, calcular, ax)

        ax.set_xlabel("Recall — fração dos cancelamentos encontrada")
        ax.set_ylabel("Precisão — acerto entre os apontados")
        ax.set_title("Curvas de precisão-recall no teste", fontsize=12, color=TINTA, pad=10)
        ax.legend(loc="upper right", fontsize=9, frameon=False)
        _limpar(ax, "both")
        fig.tight_layout()
    return fig


def curvas_calibracao(
    calibracoes: dict[str, pd.DataFrame],
    brier: dict[str, float],
    campeao: str,
    destaque_ruim: str = "AdaBoost",
) -> Figure:
    """Previsto contra observado. A diagonal é o alvo.

    Aqui a ênfase muda de dono: o que este gráfico precisa mostrar não é quem
    ganha, e sim que um modelo com ROC-AUC competitivo pode mentir sobre o nível
    do risco. A Fase 4 usa o valor absoluto da probabilidade para calcular valor
    esperado, então é este gráfico — e não o ROC — que decide em quem confiar.
    """
    with _estilo():
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], color=GRADE, lw=1.5, zorder=0, label="Calibração perfeita")

        rotulado = False
        for nome, curva in calibracoes.items():
            if nome in (campeao, destaque_ruim):
                continue
            ax.plot(
                curva["previsto"],
                curva["observado"],
                color=CINZA_FUNDO,
                lw=1.4,
                zorder=1,
                label="Demais modelos" if not rotulado else None,
            )
            rotulado = True

        for nome, cor, largura in ((destaque_ruim, LARANJA, 2.2), (campeao, AZUL, 2.4)):
            if nome not in calibracoes:
                continue
            curva = calibracoes[nome]
            ax.plot(
                curva["previsto"],
                curva["observado"],
                marker="o",
                ms=6,
                lw=largura,
                color=cor,
                zorder=3,
                markeredgecolor=SUPERFICIE,
                markeredgewidth=1.2,
                label=f"{nome} — Brier {brier[nome]:.3f}",
            )

        ax.set_xlabel("Probabilidade prevista")
        ax.set_ylabel("Frequência observada de churn")
        ax.set_title(
            "Calibração no teste\nabaixo da diagonal = superestima o risco  ·  acima = subestima",
            fontsize=11.5,
            color=TINTA,
            pad=10,
        )
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        _limpar(ax, "both")
        fig.tight_layout()
    return fig


def comparacao_lift(
    lifts: dict[str, pd.DataFrame], campeao: str, baseline: str = "Regra de contrato"
) -> Figure:
    """Lift por decil e captura acumulada — a leitura da operação de retenção."""
    with _estilo():
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

        rotulado = False
        for nome, lift in lifts.items():
            if nome in (campeao, baseline):
                continue
            rotulo = "Demais modelos" if not rotulado else None
            axes[0].plot(lift["decil"], lift["lift"], color=CINZA_FUNDO, lw=1.4, label=rotulo)
            axes[1].plot(
                lift["decil"], lift["captura_acumulada"], color=CINZA_FUNDO, lw=1.4, label=rotulo
            )
            rotulado = True

        for nome, cor, largura in ((baseline, MUTADO, 2), (campeao, AZUL, 2.4)):
            lift = lifts[nome]
            rotulo = f"{nome} (campeão)" if nome == campeao else nome
            axes[0].plot(
                lift["decil"], lift["lift"], color=cor, lw=largura, marker="o", ms=5, label=rotulo
            )
            axes[1].plot(
                lift["decil"],
                lift["captura_acumulada"],
                color=cor,
                lw=largura,
                marker="o",
                ms=5,
                label=rotulo,
            )

        axes[0].axhline(1, color=GRADE, lw=1.5, zorder=0, label="Sem modelo")
        axes[0].set_xlabel("Decil de risco  ·  1 = mais arriscado")
        axes[0].set_ylabel("Lift sobre a taxa base")
        axes[0].set_title("Lift por decil", fontsize=11, color=TINTA, pad=8)

        axes[1].plot([1, 10], [0.1, 1.0], color=GRADE, lw=1.5, zorder=0, label="Sem modelo")
        axes[1].set_xlabel("Decis contatados, do mais arriscado")
        axes[1].set_ylabel("Cancelamentos capturados")
        axes[1].set_title("Captura acumulada", fontsize=11, color=TINTA, pad=8)

        for ax in axes:
            ax.set_xticks(range(1, 11))
            ax.legend(fontsize=8.5, frameon=False)
            _limpar(ax, "both")

        fig.suptitle(
            "O que a operação enxerga: acertar no topo da lista",
            fontsize=13,
            y=1.02,
            color=TINTA,
        )
        fig.tight_layout()
    return fig


def ablacao_total_charges(
    com: pd.DataFrame, sem: pd.DataFrame, baseline: str = "Regra de contrato"
) -> Figure:
    """Efeito de remover `TotalCharges`, modelo a modelo.

    Forma de "antes e depois por item": halteres. As duas pontas são o mesmo
    modelo com e sem a variável, e a barra cinza atrás é o desvio da validação
    cruzada — a régua contra a qual o deslocamento tem de ser lido. Quando o
    haltere é mais curto que a régua, não houve efeito.

    O baseline fica de fora: a regra de contrato nunca usou `TotalCharges`, então
    suas duas pontas são idênticas por construção. Mantê-lo só esticaria o eixo.
    """
    metricas = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]
    com = com[com["modelo"] != baseline].reset_index(drop=True)
    sem = sem[sem["modelo"] != baseline].reset_index(drop=True)

    with _estilo():
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

        for ax, (chave, titulo) in zip(axes, metricas, strict=True):
            modelos = list(com["modelo"])
            y = np.arange(len(modelos))
            v_com = com[f"{chave}_media"].to_numpy()
            v_sem = sem[f"{chave}_media"].to_numpy()
            desvio = com[f"{chave}_desvio"].to_numpy()

            for i in range(len(modelos)):
                ax.plot(
                    [v_com[i] - desvio[i], v_com[i] + desvio[i]],
                    [i, i],
                    color=GRADE,
                    lw=7,
                    solid_capstyle="round",
                    zorder=0,
                    label="Desvio da validação cruzada" if i == 0 else None,
                )
                ax.plot([v_com[i], v_sem[i]], [i, i], color=CINZA_FUNDO, lw=1.6, zorder=1)

            ax.scatter(v_com, y, s=58, color=AZUL, zorder=3, label="com TotalCharges")
            ax.scatter(v_sem, y, s=58, color=LARANJA, zorder=3, label="sem TotalCharges")

            ax.set_yticks(y, modelos, fontsize=9)
            ax.set_title(titulo, fontsize=11, color=TINTA, pad=8)
            ax.invert_yaxis()
            _limpar(ax)

        axes[0].legend(fontsize=8.5, frameon=False, loc="lower left")
        fig.suptitle(
            "Ablação: remover TotalCharges desloca menos que o ruído do split",
            fontsize=13,
            y=1.02,
            color=TINTA,
        )
        fig.tight_layout()
    return fig


def efeito_class_weight(calibracoes: dict[str, pd.DataFrame], metricas: pd.DataFrame) -> Figure:
    """A armadilha do balanceamento: o ranking não muda, a probabilidade desanda."""
    cores = [AZUL, LARANJA, AGUA]

    with _estilo():
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

        axes[0].plot([0, 1], [0, 1], color=GRADE, lw=1.5, zorder=0, label="Calibração perfeita")
        for (nome, curva), cor in zip(calibracoes.items(), cores, strict=False):
            axes[0].plot(
                curva["previsto"],
                curva["observado"],
                marker="o",
                ms=6,
                lw=2.2,
                color=cor,
                markeredgecolor=SUPERFICIE,
                markeredgewidth=1.2,
                label=nome,
            )
        axes[0].set_xlabel("Probabilidade prevista")
        axes[0].set_ylabel("Frequência observada")
        axes[0].set_title("Calibração", fontsize=11, color=TINTA, pad=8)
        axes[0].legend(fontsize=9, loc="upper left", frameon=False)
        _limpar(axes[0], "both")

        # ROC-AUC praticamente constante, Brier e probabilidade média disparando:
        # é a demonstração de que ordenar bem e acertar o nível são coisas
        # diferentes. Uma escala só para as três, porque as três vivem em [0,1].
        x = np.arange(len(metricas))
        largura = 0.26
        series = [
            ("roc_auc", "ROC-AUC", AZUL, -largura),
            ("brier", "Brier", LARANJA, 0.0),
            ("prob_media", "Prob. média prevista", AGUA, largura),
        ]
        for chave, rotulo, cor, deslocamento in series:
            axes[1].bar(x + deslocamento, metricas[chave], largura * 0.9, label=rotulo, color=cor)
            for i, valor in enumerate(metricas[chave]):
                axes[1].text(
                    i + deslocamento,
                    valor + 0.012,
                    f"{valor:.3f}",
                    ha="center",
                    fontsize=8.5,
                    color=TINTA_SUAVE,
                )

        axes[1].axhline(
            metricas["prob_media"].iloc[0],
            color=MUTADO,
            lw=1.2,
            ls=(0, (4, 3)),
            zorder=0,
            label="Taxa base real",
        )
        axes[1].set_xticks(x, metricas["variante"], fontsize=9)
        axes[1].set_ylim(0, 0.95)
        axes[1].set_title(
            "Ordenar bem e acertar o nível são coisas diferentes",
            fontsize=11,
            color=TINTA,
            pad=8,
        )
        # Legenda abaixo do eixo: no canto superior ela cobria o rótulo da
        # terceira barra de ROC-AUC.
        axes[1].legend(
            fontsize=8.5,
            frameon=False,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.09),
        )
        _limpar(axes[1], "y")

        fig.suptitle(
            "class_weight='balanced' preserva o ranking e infla a probabilidade",
            fontsize=13,
            y=1.01,
            color=TINTA,
        )
        fig.tight_layout()
    return fig
