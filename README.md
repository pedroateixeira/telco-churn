# telco-churn

Previsão de cancelamento (churn) de clientes de telecomunicações, com uma camada de decisão que traduz probabilidade em recomendação de campanha, e um dashboard público.

### ▶️ [Abrir o dashboard](https://telco-churn-dashboard-pagtx.streamlit.app/)

Simulador de campanha de retenção: segmentos de risco, simulador individual, quantos clientes contatar e a metodologia por trás do número.

> ⚠️ **Projeto em construção.** O dashboard está no ar e as fases 0 a 5 estão concluídas. Faltam testes automatizados, integração contínua e a reescrita final deste README — o roteiro está em [Estado do projeto](#estado-do-projeto).

## O problema

Uma operadora com 7.043 clientes perde 26,5% deles. Um time de retenção não consegue — nem deveria — ligar para todo mundo: cada contato tem custo, e a maior parte dos clientes não ia cancelar de qualquer forma. A pergunta útil não é *"quem vai cancelar?"* e sim **"para quantos clientes vale a pena ligar, e para quais?"**.

É por isso que o projeto não termina no modelo. Um ROC-AUC não diz quantas pessoas contatar. A entrega final é um limiar de decisão derivado de valor esperado — custo da oferta contra receita preservada — e um dashboard onde os parâmetros de negócio são ajustáveis, porque eles são suposições e devem ser tratados como tal.

## Achados preliminares da EDA

Já demonstrados no notebook, contra uma taxa base de **26,5%**:

- **Contrato é o sinal dominante.** Mês a mês: 42,7% · Um ano: 11,3% · Dois anos: 2,8%.
- **O churn se concentra no começo da vida do cliente.** Tenure mediano de 10 meses entre quem cancela, contra 38 entre quem fica.
- **Fibra ótica cancela mais que DSL** (41,9% vs 19,0%) — e a explicação óbvia, preço, **não se sustenta**: dentro da fibra, quanto mais caro o plano, *menor* o churn (55,7% no quartil barato → 26,3% no caro). O que a mensalidade baixa marca é cliente novo, sem serviços adicionais e sem fidelidade.
- **`TotalCharges` é redundante.** É reconstituível a partir de `tenure × MonthlyCharges` — a mediana do resíduo é exatamente 0,00.



## Dados

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — distribuído no Kaggle por *blastchar*, originalmente um conjunto de amostra da IBM. 7.043 clientes, 21 colunas, rótulo `Churn` já pronto.

O arquivo está versionado neste repositório (977 KB) de propósito: é o que permite clonar e rodar o notebook do início ao fim sem nenhum passo manual de download. **Os dados são dos autores originais** — a licença MIT deste repositório cobre apenas o código.

## Como rodar

```bash
git clone https://github.com/pedroateixeira/telco-churn.git
```

```bash
cd telco-churn && uv sync && uv run jupyter lab
```

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12. O `uv sync` lê o `uv.lock` e reproduz exatamente as mesmas versões de biblioteca usadas aqui.

## O que é cada arquivo

### Existe hoje

| Caminho | Para que serve |
|---|---|
| `notebooks/01_eda_telco.ipynb` | **O artefato principal por enquanto.** Análise exploratória completa: tratamento dos dados justificado passo a passo, taxa base, análise univariada, checagem de categorias raras, churn por categoria contra a base e relação com as variáveis numéricas. O prefixo `01_` é ordenação — o notebook de modelagem será o `02_`. |
| `data/raw/telco_customer_churn.csv` | O dataset original, sem nenhuma alteração. Nada neste projeto escreve nesta pasta: dado bruto é imutável, e essa separação é o que garante que a análise seja refazível do zero. |
| `data/processed/` | Destino dos dados já tratados, gerados pelo pipeline. Vazio no Git de propósito (ver `.gitignore`) — conteúdo derivado se regenera, não se versiona. |
| `pyproject.toml` | Declara o projeto e suas dependências (pandas, scikit-learn, LightGBM, Streamlit) em faixas de versão, mais um grupo `dev` com pytest e ruff. Também guarda a configuração do ruff e do pytest, para não espalhar arquivo de config pela raiz. |
| `uv.lock` | Trava as versões **exatas** de todas as dependências, incluindo as transitivas. É a diferença entre "instalei as bibliotecas" e "outra pessoa consegue reproduzir o meu resultado". Versionado sempre. |
| `.python-version` | Fixa o Python em 3.12. O `uv` lê este arquivo e baixa a versão certa sozinho se ela não existir na máquina. |
| `.gitignore` | Mantém fora do Git o ambiente virtual, caches, checkpoints do Jupyter e dados derivados. Tem duas exceções deliberadas e comentadas no próprio arquivo: o CSV bruto e o `models/modelo.joblib` **são** versionados — o primeiro para o notebook rodar em máquina limpa, o segundo porque o Streamlit Cloud precisa do modelo dentro do repositório. |
| `LICENSE` | MIT. Cobre o código deste repositório, não o dataset. |
| `models/` | Onde o modelo treinado será salvo. Vazio por enquanto. |
| `reports/figures/` | Gráficos exportados para uso no README e no dashboard. Vazio por enquanto. |
| `*/.gitkeep` | Arquivos vazios que existem por uma limitação do Git: ele não versiona diretório vazio. Sem eles, a estrutura de pastas não sobreviveria ao clone. |

### Entra nas próximas fases

| Caminho | Para que vai servir |
|---|---|
| `src/telco_churn/dados.py` | Carregar, validar o schema e aplicar o tratamento que hoje está espalhado pelas células do notebook. Falha alto se uma coluna esperada sumir. |
| `src/telco_churn/features.py` | O `ColumnTransformer`: codificação de categóricas e escala, ajustados **só** no conjunto de treino. |
| `src/telco_churn/modelo.py` | Montagem do `Pipeline`, treino e avaliação (ROC-AUC, PR-AUC, lift por decil, calibração). |
| `src/telco_churn/decisao.py` | A camada de valor esperado: limiar ótimo por custo, curva orçamento × lucro, análise de sensibilidade. |
| `src/telco_churn/treinar.py` | Ponto de entrada: um comando treina do zero e salva o artefato. Notebook explora; script entrega. |
| `notebooks/02_modelagem.ipynb` | A narrativa da modelagem — baseline, comparação de modelos, ablação de `TotalCharges`. |
| `app/streamlit_app.py` | O dashboard: segmentos de risco, simulador individual e simulação de campanha. |
| `tests/` | pytest para o que quebra em silêncio: validação de schema, ausência de vazamento no pipeline, corretude do limiar. |
| `.github/workflows/ci.yml` | Roda ruff e pytest a cada push. |
| `requirements.txt` | Gerado a partir do `uv.lock` só para o deploy — o Streamlit Community Cloud lê este formato. |

## Estado do projeto

- [x] **Fase 0** — Repositório, ambiente reprodutível e licença
- [x] **Fase 1** — Fechar a EDA: os cinco achados escritos
- [x] **Fase 2** — Do notebook ao pacote `src/`
- [x] **Fase 3** — Modelagem: baseline → regressão logística → LightGBM, com calibração
- [x] **Fase 4** — Camada de decisão por valor esperado
- [x] **Fase 5** — Dashboard Streamlit — [no ar](https://telco-churn-dashboard-pagtx.streamlit.app/)
- [ ] **Fase 6** — Testes e integração contínua
- [ ] **Fase 7** — README final e deploy público

## Decisões e limitações

- **Sem split temporal.** O dataset não tem nenhuma coluna de data — `tenure` é duração, não data de entrada do cliente. Isso torna impossível separar treino e teste por safra, que seria o correto. A consequência: a performance medida é otimista em relação ao que se veria em produção. 

