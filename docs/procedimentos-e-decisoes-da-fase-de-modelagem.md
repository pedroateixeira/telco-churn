# Procedimentos e decisões da fase de modelagem

Registro de **todas** as decisões tomadas na fase de modelagem do projeto, com a razão de cada uma. O que este documento tenta evitar é o problema comum de projeto de portfólio: números certos, escolhas invisíveis. Aqui cada escolha aparece com o motivo, a alternativa descartada e — quando existe — o número que a sustenta.

Complementa, sem repetir: a narrativa está em [`02_modelagem.ipynb`](../notebooks/02_modelagem.ipynb), o resumo dos resultados no [README](../README.md), e o código em [`src/telco_churn/`](../src/telco_churn/).

**Ambiente:** Python 3.12.2 · scikit-learn 1.9.0 · XGBoost 3.4.1 · LightGBM 4.7.0 · pandas 3.0.5
**Reprodução:** `uv run python -m telco_churn.comparacao` e `uv run python -m telco_churn.treinar`

---

## 1. Preparação dos dados

### 1.1. O tratamento roda antes do split — e por que isso não é vazamento

Esta é a decisão mais sutil da fase, e a que mais se erra.

`dados.tratar()` é chamada sobre a base inteira, **antes** de separar treino e teste. Isso parece vazamento e não é, por um motivo específico: todas as suas transformações são **determinísticas linha a linha** — mapeamento de `"Yes"`/`"No"` para 1/0, colapso de categorias, e um `fillna(0)` com constante. Nenhuma delas olha para mais de uma linha por vez, e nenhuma aprende um número a partir do conjunto.

A regra que separa um caso do outro:

> Qualquer transformação que calcule um valor **olhando várias linhas** — média, desvio, lista de categorias, mediana de imputação — tem de estar dentro do `Pipeline`. As que operam linha a linha podem vir antes.

É por isso que `StandardScaler` e `OneHotEncoder` estão dentro do `Pipeline` (§1.5) e o tratamento não.

### 1.2. `TotalCharges`: `fillna(0)` e não imputação por média

São 11 clientes com valor ausente, todos com `tenure == 0`: entraram e ainda não fecharam o primeiro ciclo de cobrança. Zero é o **valor correto**, não uma estimativa.

A distinção importa para o vazamento: uma imputação por média seria um parâmetro aprendido do conjunto e teria de ir para dentro do `Pipeline`. Uma constante justificada pelo domínio não depende dos dados e pode ficar em `tratar()`.

### 1.3. `MultipleLines` colapsada para 0/1

O notebook de EDA deixou esta coluna como texto de três categorias (`Yes` / `No` / `No phone service`), enquanto os seis add-ons de internet foram colapsados. Apliquei o mesmo padrão.

**Razão, com o dado:** `PhoneService == 0` ⟺ `MultipleLines == "No phone service"`, exatamente 682 linhas, implicação perfeita. A terceira categoria não carrega nenhuma informação além da que `PhoneService` já dá — mantê-la geraria uma coluna one-hot perfeitamente colinear com uma binária existente.

Depois disso sobram três colunas de texto genuinamente multivaloradas: `InternetService`, `Contract`, `PaymentMethod`.

### 1.4. Validação de schema nas duas pontas

- `validar_schema()` roda no dado **bruto**, antes do tratamento: confere colunas, tipos e domínio das categóricas, e reporta **todos** os problemas de uma vez em vez do primeiro. Um validador que falha um por vez obriga a rodar o pipeline N vezes para descobrir N erros.
- `validar_tratado()` roda **depois**, e fecha um contrato entre módulos: `features.py` declara que toda coluna fora das listas conhecidas é 0/1 e a manda por `passthrough`; quem torna isso verdade é `tratar()`. Sem a checagem, acrescentar uma coluna numérica contínua ao schema a faria entrar **sem escala**, ao lado de features escaladas, degradando o modelo linear em silêncio.

*Detalhe de ambiente:* em pandas 3, texto vem como `StringDtype` e não `object`. Uma validação escrita como `dtype == object` passaria silenciosamente por tudo — por isso o código usa `pandas.api.types`.

### 1.5. O pré-processador

| Bloco | Transformação | Razão |
|---|---|---|
| Numéricas | `StandardScaler` | `tenure` (0–72), `MonthlyCharges` (18–119) e `TotalCharges` (0–8684) vivem em escalas muito diferentes; sem escala a regularização L2 da logística penaliza as três de forma desigual. Árvores são indiferentes, mas um pré-processador só serve os dois. |
| Categóricas | `OneHotEncoder(handle_unknown="ignore")` | Categoria nunca vista em produção vira vetor de zeros em vez de derrubar a predição. |
| Binárias | `passthrough` | Já são 0/1. Escalar não mudaria a informação e atrapalharia a leitura dos coeficientes. |
| Resto | `remainder="drop"` **explícito** | Coluna nova no dado não entra no modelo sem alguém ter decidido o que fazer com ela. |

**Sem `drop="first"` no one-hot.** Medi as duas versões: 26 features contra 23, ROC-AUC **0,8424 contra 0,8422** — diferença nula. Sem o drop, o mesmo pré-processador serve modelo linear e árvore sem ressalva (a regularização L2 já lida com a colinearidade, e para árvore o drop atrapalha de leve, por esconder uma categoria de um split direto). Um caminho de código em vez de dois.

**Colunas binárias derivadas do schema, não escritas à mão.** `COLUNAS_BINARIAS_MODELO` é calculada por subtração a partir de `SCHEMA_BRUTO`. Uma lista manual divergiria em silêncio na primeira mudança de dado.

---

## 2. Desenho da avaliação

### 2.1. Não há split temporal — e isso é um limite do dado

A convenção correta em churn é split temporal: treinar em quem entrou antes de uma data, testar nos posteriores, porque é assim que o modelo é usado.

**O Telco não permite.** Não há nenhuma coluna de data na base. `tenure` é *duração*, não data de entrada — dois clientes com `tenure = 10` podem ter entrado com anos de diferença, e o dado não diz qual. Sem data de originação não há safra, e sem safra não há split temporal.

Restou split aleatório estratificado. **Consequência declarada: a performance medida aqui é otimista** em relação a produção, porque treino e teste vêm da mesma janela e compartilham qualquer efeito de período. Não é defeito do código; é limite do dado, e vale mais dizer do que esconder.

### 2.2. Split estratificado, `test_size=0.2`, `seed=42`

Estratificado por `Churn` porque a taxa base é 26,5%: um split aleatório simples pode deslocar a proporção entre treino e teste o suficiente para bagunçar a comparação. Resultado: 5.634 de treino, 1.409 de teste.

`test_size` virou **parâmetro registrado no relatório** — é um número que muda todas as métricas e, antes, não aparecia em lugar nenhum do artefato.

### 2.3. A escolha do campeão não se faz no conjunto de teste

Duas razões, e a segunda é a mais séria:

1. **Ruído.** Com 1.409 linhas, o desvio-padrão do ROC-AUC é da ordem de ±0,012, e as diferenças entre os modelos aprendidos são de ~0,005. Escolher por um split só seria decidir na quarta casa decimal de uma amostra.
2. **Contaminação.** Olhar o teste a cada comparação o transforma em conjunto de validação. Depois de sete espiadas ele deixa de ser estimativa honesta de desempenho fora da amostra — é uma forma lenta de ajustar ao teste.

Por isso: **seleção por validação cruzada sobre o treino; teste medido uma única vez, no fim.**

### 2.4. `RepeatedStratifiedKFold`, 5 folds × 3 repetições

15 ajustes por modelo. Cinco folds é o compromisso usual entre viés e custo; as três repetições existem para que a estimativa não dependa de um único sorteio de partição. Estratificado pela mesma razão do split principal.

`avaliar_cv` recebe uma **fábrica** (função que devolve um estimador novo), não um estimador pronto — cada fold precisa treinar do zero, e reusar a mesma instância vazaria estado entre folds.

---

## 3. Métricas

| Métrica | Papel | Razão |
|---|---|---|
| **PR-AUC** | Critério de escolha | Com 26,5% de positivos, é a que reflete o uso real (uma lista de contatos). |
| **ROC-AUC** | Comparabilidade | Reportado por convenção, mas é otimista: inclui a facilidade de acertar os negativos, que são a maioria. |
| **Brier** | Portão de calibração | Mede se a probabilidade está certa em **nível**, não só em ordem. A camada de decisão depende disso. |
| **Lift por decil** | Leitura da operação | A campanha alcança um número limitado de clientes; o que importa é acertar no topo da lista. |

**Acurácia foi descartada.** Prever "ninguém cancela" acerta 73,5% — um número que parece bom e não contém informação nenhuma.

**Por que Brier é um portão e não um detalhe:** o ROC-AUC só enxerga a *ordem*. Multiplicar todas as probabilidades por dois não o altera em nada. Mas a Fase 4 calcula `VE = p_churn × taxa_aceite × margem × horizonte − custo_da_oferta`, que usa o **valor absoluto** de `p_churn`. Probabilidade inflada dimensiona a campanha errada por mais perfeito que o ranking esteja.

---

## 4. A escada de modelos

### 4.1. O baseline é um estimador de verdade

`RegraContrato` implementa `Contract == "Month-to-month" → churn`. Não é formalidade: a EDA mostrou que `Contract` sozinho separa 42,7% contra 11,3% e 2,8%.

**Decisão de desenho:** seu `predict_proba` devolve a **taxa de churn observada no treino dentro de cada grupo**, e não 0/1. Comparar uma regra que só emite 0 e 1 contra modelos probabilísticos usando Brier ou PR-AUC seria vitória de régua torta — a regra perderia por não estar tentando responder a mesma pergunta. `predict` continua aplicando a regra crua.

É também o único modelo que **não** passa pelo pré-processador: ele lê `Contract` como texto, e o one-hot destruiria justamente o que usa.

### 4.2. Os modelos que aprendem

| Modelo | Configuração | Papel |
|---|---|---|
| Regressão logística | `max_iter=1000` | Referência interpretável |
| AdaBoost | `n_estimators=300`, `learning_rate=0.5` | Família de boosting clássica |
| XGBoost | `n_estimators=300`, `lr=0.05`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0` | Candidato |
| LightGBM | `n_estimators=300`, `lr=0.05`, `num_leaves=15`, `subsample=0.8`, `colsample_bytree=0.8` | Candidato |

**Hiperparâmetros deliberadamente conservadores, sem busca.** Com 5.634 linhas de treino e candidatos que se distinguem por menos que o ruído do split, uma busca de hiperparâmetro entraria como mais uma fonte de variância entre modelos que já empatam — e o vencedor seria, em boa parte, quem teve sorte na busca. Árvores rasas (`max_depth=3`, `num_leaves=15`) e `learning_rate` baixo são a escolha defensiva para dado pequeno.

**Sem `random_state` na logística.** O solver `lbfgs` é determinístico; o parâmetro só teria efeito com `sag`, `saga` ou `liblinear`. Passá-lo sugeriria uma escolha aleatória que não existe — a reprodutibilidade vem do seed do split.

### 4.3. Os dois ensembles

**Votação *soft*, não *hard*.** A média é de probabilidades, não de rótulos: votar em 0/1 jogaria fora exatamente a informação que a camada de decisão consome. A média de probabilidades também tende a melhorar a calibração, porque erros independentes se cancelam em parte.

**Stacking com meta-logística, `cv=5`, `stack_method="predict_proba"`.** Em vez de assumir peso igual para os quatro, o meta-modelo aprende quanto confiar em cada um, a partir de predições fora da amostra.

**Por que os dois entram, em vez de eu escolher um:** stacking é mais poderoso e arrisca sobreajustar no nível de cima. Qual dos dois ganha nesta base é pergunta empírica, e responder por palpite seria desperdiçar a comparação que já estava montada.

---

## 5. A decisão do campeão

### 5.1. Comparar médias não bastou

| Modelo | ROC-AUC | PR-AUC | Brier | s/ajuste |
|---|---|---|---|---|
| Regra de contrato | 0,729 ±0,011 | 0,409 ±0,010 | 0,163 | 0,00 |
| Regressão logística | 0,846 ±0,012 | 0,661 ±0,017 | 0,135 | 0,02 |
| AdaBoost | 0,847 ±0,012 | 0,660 ±0,019 | **0,171** | 0,70 |
| **XGBoost** | 0,847 ±0,012 | 0,668 ±0,020 | 0,134 | 0,24 |
| LightGBM | 0,841 ±0,013 | 0,657 ±0,021 | 0,137 | 0,61 |
| Ensemble (votação) | 0,850 ±0,012 | 0,671 ±0,020 | 0,136 | 1,53 |
| Ensemble (stacking) | 0,851 ±0,012 | 0,672 ±0,018 | 0,134 | 8,77 |

Contra o baseline, a diferença é enorme e inequívoca. Entre si, os desvios se sobrepõem quase inteiramente. Lida assim, a tabela diz "tanto faz" — e essa leitura estaria errada.

### 5.2. A comparação pareada, e por que ela muda a conclusão

O erro é comparar médias com desvios quando os desvios medem a coisa errada.

A variação **entre folds** (±0,019 de PR-AUC) é o dobro da diferença entre modelos (~0,010). Mas essa variação é **comum a todos**: um fold difícil é difícil para os sete. Como os folds são compartilhados, a diferença *dentro de cada fold* cancela essa componente comum.

A pergunta certa não é *"a diferença é maior que o desvio entre folds?"* e sim **"o sinal da diferença se mantém fold a fold?"**.

Contra a regressão logística, nos mesmos 15 folds:

| Modelo | Δ PR-AUC | Folds vencidos |
|---|---|---|
| Ensemble (stacking) | +0,0103 ±0,0055 | **14/15** |
| Ensemble (votação) | +0,0098 ±0,0063 | **15/15** |
| XGBoost | +0,0065 ±0,0087 | 11/15 |
| AdaBoost | −0,0011 ±0,0085 | 5/15 |
| LightGBM | −0,0044 ±0,0102 | 4/15 |

O sinal aparece nos dois sentidos: os ensembles **são** melhores, com consistência que a comparação de médias escondia; e LightGBM e AdaBoost **não** superam a logística — o que aquela tabela também escondia.

### 5.3. A regra de desempate, declarada em código

> Entre os modelos que empatam com o líder — o sinal da diferença pareada contra ele não se sustenta —, fica **o mais barato de treinar**.

Operacionalmente: um modelo empata com o líder quando `|dif_media − dif_media_do_líder| < dif_desvio`.

O líder por PR-AUC é o **stacking** (0,6715), a 8,84 s/ajuste. Empatam com ele **XGBoost** (0,24 s) e **votação** (1,53 s). O mais barato é o XGBoost — **37× mais rápido que o líder** por 0,0038 de PR-AUC.

**Por que essa regra e não "o melhor número vence":** complexidade que não paga em métrica é custo puro — mais tempo de treino, mais superfície de falha no deploy, mais dependências no artefato, e um modelo composto de quatro outros é muito mais difícil de defender numa entrevista do que um só. A regra fica em código, e não no julgamento, para que a escolha seja auditável e se refaça sozinha se o dado mudar.

**Campeão: XGBoost.**

### 5.4. O campeão não está escrito no código

`treinar.py` lê o nome de `models/comparacao.json`. A validação cruzada decide; o treino apenas persiste a decisão. Fixar o nome no código faria a escolha envelhecer calada na primeira vez que a comparação mudasse.

---

## 6. Calibração

### 6.1. `class_weight="balanced"` foi testado e rejeitado

Com 26,5% de positivos, é o reflexo comum. Foi medido em três variantes da mesma logística:

| Variante | ROC-AUC | Brier | Probabilidade média prevista |
|---|---|---|---|
| Sem balanceamento | 0,8424 | **0,1379** | **0,268** |
| `class_weight="balanced"` | 0,8418 | **0,1686** | **0,416** |
| Balanceada + isotônica | 0,8415 | **0,1388** | **0,267** |

A taxa base real é **0,265**. O modelo sem balanceamento prevê 0,268 em média — calibrado. O balanceado prevê **0,416**: acha que 42% da carteira vai cancelar quando 26,5% cancela, **superestimando o risco em mais de 50%**.

E o ROC-AUC não denuncia nada: 0,8424 contra 0,8418, diferença na terceira casa. Um projeto que reportasse só AUC entregaria à camada de decisão um modelo que infla o valor esperado de toda campanha, e o erro só apareceria quando o retorno real viesse abaixo do previsto.

**Decisão: modelo de produção sem `class_weight`.** A recalibração isotônica funciona (Brier volta a 0,1388), mas seria adicionar um estágio para consertar um problema que não precisava ser criado.

### 6.2. O AdaBoost como contraexemplo

Brier de **0,171** — pior que o da regra de uma linha (0,163) — com ROC-AUC competitivo (0,847). A curva de calibração mostra por quê: ele fica bem abaixo da diagonal em toda a faixa média, superestimando o risco de forma sistemática. É característica conhecida da família — o `predict_proba` do AdaBoost deriva da perda exponencial e comprime as probabilidades em torno de 0,5.

Ele **ordena bem e mente sobre o nível**. Para a camada de decisão seria o pior candidato da escada, apesar de parecer o quarto melhor numa tabela que só mostrasse AUC. Ficou no projeto exatamente por isso: prova, com dado próprio, que escolher modelo por AUC quando se precisa de probabilidade leva à escolha errada.

---

## 7. Ablação de `TotalCharges`

A EDA provou que a variável é reconstituível de `tenure × MonthlyCharges` (mediana do resíduo 0,00) e argumentou que é *pior que redundante*, por misturar dois sinais opostos. Argumento é argumento; a pergunta empírica é quanto custa remover.

**Decisão de desenho:** `usar_total_charges` é parâmetro de `construir_preprocessador`, não uma edição manual de código. O achado da EDA vira experimento reproduzível — `--sem-total-charges` na linha de comando.

**Resultado:** o maior deslocamento em qualquer modelo é de **0,0018 de ROC-AUC**, contra desvio entre folds de ~0,012 — menos de **0,15 desvio**. Removê-la não custa nada mensurável.

**Por que o modelo de produção ainda a mantém:** a ablação prova que *pode* sair, e a remoção definitiva é uma mudança para tomar junto com a Fase 4, quando as features também alimentarem o simulador do dashboard — tirar uma variável agora significaria mexer no formulário depois. É uma decisão adiada de propósito, não um esquecimento.

---

## 8. Resultado final, no conjunto de teste

Medição única, com o campeão treinado no treino inteiro:

| | Treino | Teste |
|---|---|---|
| ROC-AUC | 0,8840 | **0,8448** |
| PR-AUC | 0,7365 | **0,6639** |
| Brier | 0,1189 | **0,1356** |

**Sobre o gap treino–teste de 0,039:** é maior que o da logística (0,007), e isso é esperado — boosting tem muito mais capacidade de decorar. Não é sinal de problema: a estimativa que vale é a do teste, e ela bate com a validação cruzada (0,847 na CV contra 0,845 no teste), o que indica que a seleção não sobreajustou o processo de escolha. Mas é o número a vigiar se o modelo for retreinado com hiperparâmetros mais agressivos.

**Leitura de operação:** lift de **2,89** no primeiro decil. Contatando os dois primeiros decis — 282 clientes, 20% da base — a campanha alcança **51% de todos os cancelamentos**; a regra de contrato, no mesmo orçamento, alcança 30%.

---

## 9. Decisões de engenharia da fase

| Decisão | Razão |
|---|---|
| Artefato da ablação com nome próprio (`modelo_sem_total_charges.joblib`) | Antes, rodar a ablação substituía o modelo de produção pelo do experimento, e o dashboard passaria a servir a ablação sem nenhum sinal. |
| Relatório registra Python, sklearn e pandas | `joblib` grava pickle de objetos sklearn; carregar com outra versão pode desserializar e predizer diferente. Sem o registro, o desencontro é indiagnosticável. |
| Notebook 02 lê artefatos em vez de retreinar | Um notebook que retreina produz números um pouco diferentes a cada execução, e a narrativa passa a divergir da tabela sem ninguém perceber. |
| `graficos.py` não treina nem salva | Recebe dados prontos e devolve `Figure`, para os mesmos gráficos servirem notebook e dashboard sem retreinar. |
| `lift_por_decil` falha alto em recorte pequeno ou sem churns | Antes devolvia tabela com decis salteados ou `NaN`. Os dois casos são plausíveis no dashboard, onde a base é filtrada. |
| Figuras versionadas no Git | O README as exibe; um repositório que mostra gráfico quebrado até alguém rodar o pipeline não serve à leitura de 40 segundos. |

### Sobre os gráficos

A primeira versão usava barras e estava errada: com linha de base no zero, todos os modelos viravam a mesma barra de 0,85 e as barras de erro sumiam — apagando exatamente o que o gráfico precisava mostrar.

- **Ponto com barra de erro, não barra.** Ponto não promete linha de base no zero, então a escala pode focar a faixa onde a decisão acontece.
- **Ênfase, não sete cores.** O achado é que os modelos aprendidos empatam. Sete curvas quase sobrepostas pedem que o leitor distinga o indistinguível; destacar o campeão com o resto em cinza é a forma que corresponde à conclusão.
- **Baseline fora da escala, citado no subtítulo.** Ele está tão distante (PR-AUC 0,41 contra 0,67) que, mantido como ponto, comprimiria a comparação real num décimo do gráfico. São duas perguntas diferentes, e a nota responde a primeira sem sacrificar a segunda.
- **Ablação como haltere**, com o desvio da validação cruzada como régua cinza atrás: quando o haltere é mais curto que a régua, não houve efeito.

---

## 10. O que **não** foi feito, e por quê

| Não feito | Razão |
|---|---|
| **Busca de hiperparâmetros** | Candidatos que já se distinguem por menos que o ruído do split; a busca adicionaria variância e o vencedor seria em parte quem teve sorte. Entraria depois de o campeão estar decidido, para espremer o modelo escolhido — não para escolhê-lo. |
| **SMOTE e reamostragem** | Mesmo problema do `class_weight`, agravado: distorcem a distribuição e portanto a probabilidade, que é a saída de que a Fase 4 depende. Com 26,5% de positivos o desbalanceamento nem é severo. |
| **Feature engineering além da EDA** | A base tem 19 features para 7.043 linhas. Criar variáveis derivadas sem hipótese vinda dos dados é caminho curto para sobreajuste. |
| **Split temporal** | Impossível: não há coluna de data (§2.1). |
| **Seleção de features automática** | A única remoção candidata veio de um argumento da EDA e foi testada explicitamente (§7). Seleção automática num dado pequeno tende a escolher ruído. |
| **Threshold de decisão** | É a Fase 4, e por escolha: o limiar sai de valor esperado — custo da oferta contra receita preservada —, não de F1 ou do ponto de Youden. Escolher agora por métrica estatística seria decidir o problema errado. |

---

## 11. Limitações honestas

1. **A performance é otimista.** Sem split temporal, treino e teste compartilham a janela de tempo (§2.1).
2. **Uma base pequena e limpa.** 7.043 linhas, sem valores ausentes de verdade, sem deriva, sem histórico. Nada aqui exercita o que quebra em produção.
3. **O gap treino–teste do campeão é de 0,039.** Aceitável para boosting e coerente com a CV, mas é o número a vigiar num retreino mais agressivo.
4. **A calibração foi medida uma vez, no teste.** Não há garantia de que se mantenha sob deriva — em produção isso seria monitorado, e aqui não há como.
5. **Os ganhos sobre a logística são pequenos.** ~1,5% relativo de PR-AUC. São reais (15/15 folds), mas quem precisar de interpretabilidade total pode ficar com a logística pagando pouco.
