# Procedimentos e decisões da camada de decisão

Registro de **todas** as decisões tomadas na fase que traduz probabilidade de churn em lista de campanha, com a razão de cada uma. Documento irmão de [Procedimentos e decisões da fase de modelagem](procedimentos-e-decisoes-da-fase-de-modelagem.md), e mesma intenção: nesta fase quase nada é imposto pelos dados, quase tudo é escolha, e escolha sem razão registrada é opinião disfarçada de resultado.

Complementa, sem repetir: a narrativa está em [`03_decisao.ipynb`](../notebooks/03_decisao.ipynb), o resumo no [README](../README.md), e o código em [`src/telco_churn/decisao.py`](../src/telco_churn/decisao.py).

**Reprodução:** `uv run python -m telco_churn.decisao`
**Base de simulação:** 1.409 clientes do conjunto de teste · taxa base 26,54%

---

## 1. Por que o limiar não sai de uma métrica estatística

### 1.1. O que foi descartado, e o motivo

A escolha convencional de limiar tem três candidatos usuais. Nenhum serve aqui, e a razão é a mesma nos três:

| Critério | Por que foi descartado |
|---|---|
| **F1** | Maximiza a média harmônica de precisão e recall. Trata falso positivo e falso negativo como se custassem o mesmo, e neste problema eles diferem em ordem de grandeza: um falso negativo é um cliente perdido (margem × horizonte); um falso positivo é uma ligação mais um desconto. |
| **Ponto de Youden** (`max(TPR − FPR)`) | Mesma cegueira, com peso implícito ainda mais arbitrário, assumindo que uma unidade de sensibilidade vale exatamente uma de especificidade. |
| **Joelho da curva ROC** | É um critério *visual*. O ponto de maior curvatura não corresponde a nada no negócio. |

O problema comum aos três: **otimizam a forma da matriz de confusão, não o resultado da operação**. Nenhum deles pergunta quanto custa a oferta nem quanto vale o cliente e são exatamente esses dois números que determinam a resposta.

Há um sintoma que denuncia isso de imediato: os três produzem **um** limiar, igual para todo mundo. A seção 4 mostra que o limiar correto é diferente para cada cliente.

### 1.2. O que o valor esperado acrescenta

Duas coisas que nenhuma ordenação por probabilidade dá, por melhor que o modelo seja:

1. **Onde parar.** Uma lista ordenada por `p` não tem ponto de corte natural. Com valor esperado, o corte é onde o próximo cliente da fila deixa de se pagar.
2. **Se a campanha deveria existir.** A seção 3 mostra configurações de oferta em que a resposta correta é *não faça a campanha* — conclusão inalcançável por qualquer métrica de classificação.

---

## 2. O modelo econômico

### 2.1. A estrutura de custo escolhida

Três desenhos foram considerados:

| Desenho | Avaliação |
|---|---|
| Só custo fixo por contato | Simples, mas trata um cliente de R$ 20 e um de R$ 118 como se valessem o mesmo esforço. |
| Só desconto proporcional | Some com o fato de que abordar cliente custa dinheiro mesmo quando ele recusa. |
| **Contato fixo + desconto proporcional** ✅ | É como retenção funciona em telecom de verdade, e é o único que expõe a tensão central: o desconto vai também para quem não ia cancelar. |

### 2.2. A fórmula, e a linha que a versão usual omite

| Componente | Quando ocorre | Valor |
|---|---|---|
| Margem preservada | Ia cancelar (`p`) **e** aceitou (`a`) | `p · a · margem · mc · H` |
| Custo de contato | Sempre que se aborda | `custo_contato` |
| Custo do desconto | Aceitou (`a`), **tendo ou não** intenção de sair | `a · desconto · mc · H` |

```
VE = a · mc · H · (p · margem − desconto) − custo_contato
```

**A terceira linha é a correção mais importante desta fase.** A formulação que circula na maioria dos materiais é:

```
VE = p × a × margem × H − custo_da_oferta        ← errada
```

Ela trata o desconto como se fosse pago apenas por clientes retidos. Não é. A oferta é feita antes de saber quem ia sair, e **todo mundo que aceita recebe o desconto**, inclusive os ~73% que ficariam de qualquer forma. Esse é o custo real de qualquer campanha de retenção; omiti-lo infla o resultado e, pior, esconde o achado da seção 3.

### 2.3. `MonthlyCharges` como margem, não receita

`MonthlyCharges` é **receita**, não lucro. Contar a mensalidade inteira como valor salvo superestimaria o ganho em mais de 3x, e é o erro mais comum nesse tipo de conta.

A decisão foi aplicar um percentual de margem de contribuição, com **30% de padrão** — faixa típica de EBITDA em telecom. É premissa, não dado, e entra na análise de sensibilidade como todas as outras.

### 2.4. Os cinco parâmetros e de onde vêm

```python
@dataclass(frozen=True)
class ParametrosNegocio:
    margem: float = 0.30
    desconto: float = 0.10
    taxa_aceite: float = 0.40
    horizonte_meses: int = 12
    custo_contato: float = 15.0
```

**Nenhum dos cinco vem dos dados.** Essa é a diferença mais importante entre esta fase e a anterior: na modelagem, os números eram medidos; aqui, são arbitrados. Três decisões de engenharia seguem daí:

- **`dataclass` congelado**, e não constantes soltas ou argumentos espalhados. Reúne as premissas num objeto que se passa inteiro, se serializa no relatório e se varia na sensibilidade.
- **Registrados em `models/decisao.json`.** Resultado sem as premissas ao lado é número sem unidade.
- **Sensibilidade obrigatória** (§6), não opcional. Um resultado pontual, com cinco palpites embutidos, seria falsa precisão.

`taxa_aceite` é o mais frágil dos cinco — não tem nem uma referência de mercado para ancorar. É a primeira coisa que um piloto real mediria.

---

## 3. O achado principal: o piso do limiar

### 3.1. A álgebra

Isolando `p` na condição `VE > 0`:

```
p* = desconto/margem  +  custo_contato / (a · margem · mc · H)
     └── piso ────┘      └──── termo por cliente ────┘
```

O primeiro termo **não depende do cliente, do horizonte, da taxa de aceite nem da qualidade do modelo**. Só da razão entre desconto e margem.

### 3.2. O que isso significa, com os números

| Desconto | Piso do limiar | Contatáveis (de 1.409) | Lucro esperado |
|---|---|---|---|
| 2% | 0,067 | 597 | R$ 21.301 |
| 5% | 0,167 | 487 | R$ 15.044 |
| **10%** (padrão) | **0,333** | **313** | **R$ 7.223** |
| 15% | 0,500 | 154 | R$ 2.629 |
| 20% | 0,667 | 58 | R$ 491 |
| 25% | 0,833 | **0** | **R$ 0** |
| 30% | 1,000 | **0** | **R$ 0** |

Duas leituras:

**1. O desenho da oferta domina o modelo.** Entre 2% e 10% de desconto o lucro cai de R$ 21 mil para R$ 7 mil. Nenhum ganho de modelagem plausível na Fase 3 — onde os candidatos se distinguiam por ~0,010 de PR-AUC — chegaria perto desse efeito. 

**2. Com desconto igual à margem, a campanha é impossível.** O piso vai a 1,0 e nenhum cliente pode justificar contato, por melhor que seja o modelo. Um ROC-AUC de 0,99 não salvaria — o problema não está em prever, está no desenho da oferta. Cada real de desconto concedido precisa ser recuperado em margem de quem seria perdido; se o desconto consome a margem inteira, não sobra o que recuperar.

### 3.3. Uma descoberta da execução: ela morre antes do limite teórico

O plano previa que a campanha morreria em `desconto = margem` (30%). **Ela morre em 25%.**

O motivo: o limite em 30% é a garantia *teórica* — o ponto em que nem uma probabilidade de 1,0 basta. Mas o modelo não produz probabilidade 1,0. Basta o piso ultrapassar a **maior probabilidade que o modelo de fato gera** para não sobrar ninguém elegível, e a 25% o piso já é 0,833.

A consequência é que o limite real **depende do modelo** — um modelo mais confiante empurraria a morte um pouco para a direita. A figura marca as duas linhas: a laranja onde o lucro zera de fato, a cinza no limite teórico.

---

## 4. Não existe um limiar — existe um por cliente

O segundo termo da fórmula, `custo_contato / (a · margem · mc · H)`, cai com a mensalidade. Com os parâmetros padrão:

| Cliente | Limiar de contato |
|---|---|
| Mensalidade R$ 20 | **0,904** |
| Mediana da base | 0,482 |
| Mensalidade R$ 118 | **0,423** |

Cliente caro compensa ser abordado com **menos da metade** da certeza exigida do cliente barato: a margem em jogo é maior e o custo fixo de contato pesa proporcionalmente menos.

**Consequência operacional:** ordenar por valor esperado não é ordenar por probabilidade. Entre os 313 primeiros de cada lista, **286 coincidem e 27 trocam** — cerca de 9% da campanha muda de alvo. Os que entram são de mensalidade alta com risco moderado; os que saem, de mensalidade baixa com risco alto cuja margem não cobre o custo de abordagem.

Isto é o que os critérios da §1.1 não conseguem representar: eles produzem um escalar onde o problema pede uma função.

---

## 5. As quatro filas de contato

`simular_campanha` recebe a ordem da fila como parâmetro e aplica **a mesma conta de valor esperado**. Trocar só a ordem isola o que cada camada acrescenta:

| Fila | Lucro no orçamento ótimo | O que representa |
|---|---|---|
| Valor esperado | **R$ 7.223** | Modelo + camada de decisão |
| Probabilidade de churn | R$ 7.017 | O que o modelo entrega sozinho |
| Regra de contrato | **R$ −1.034** | O que dá para fazer sem modelo |
| Aleatória | R$ −5.453 | O piso: contatar sem informação |

**Por que a regra de contrato dá lucro negativo.** Ela acerta o segmento: mês a mês tem 42,7% de churn, mas **não ordena dentro dele**. Contatar todos os clientes mês a mês inclui gente demais que não ia cancelar, e o desconto pago a eles come o ganho. Uma regra de uma linha identifica risco; a campanha precisa de ordenação, e é isso que ela não tem. A implementação preserva essa limitação de propósito: a fila da regra usa `kind="stable"`, sem desempate por `p`, porque dar a ela um critério de ordenação que ela não possui seria inflar artificialmente o baseline.

**A honestidade desconfortável.** O ganho da camada de decisão sobre a ordenação por probabilidade pura é de **R$ 205, ~2,9%**. As duas curvas quase se sobrepõem no gráfico. O salto grande é da regra para o modelo; do modelo para o valor esperado é ajuste fino.

Isso está escrito no notebook, no README e aqui, em vez de omitido, porque o valor da camada não está na magnitude desse delta. Está nos dois itens da §1.2: saber **onde parar** e saber **se a campanha deveria existir**. Nenhum dos dois aparece numa comparação de lucro a orçamento fixo.

### 5.1. Por que o ponto ótimo não precisa de otimização

`campanha_otima` seleciona todo cliente com `VE > 0` e mais nenhum. Como a fila está ordenada por VE decrescente, esse conjunto é o máximo global e não há o que buscar. É por isso que a verificação compara o resultado com uma varredura em força bruta: para provar que a forma fechada não escondeu um erro.

---

## 6. Sensibilidade

### 6.1. Um parâmetro por vez, e por que isso basta aqui

A varredura é univariada: cada parâmetro varia com os outros quatro no padrão. Não é análise de interação, e isso é uma limitação real, mas o objetivo aqui é responder **quais premissas dominam**, e para isso a varredura univariada é suficiente e legível. Uma superfície de cinco dimensões seria mais correta e ilegível.

### 6.2. O que domina

| Parâmetro | Amplitude do lucro ótimo |
|---|---|
| Horizonte | R$ 32.534 |
| Margem | R$ 26.862 |
| Desconto | R$ 21.301 |
| Taxa de aceite | R$ 19.612 |
| Custo de contato | R$ 13.172 |

A leitura tem de separar o que se **controla** do que se **herda**:

- **Horizonte e margem** têm a maior amplitude, mas são características do negócio. Ninguém decide que a margem é 50%.
- **Desconto** é o mais decisivo entre os controláveis, e no sentido contrário à intuição: reduzir de 10% para 5% **dobra** o lucro, porque encolhe o subsídio pago a quem ia ficar.
- **Taxa de aceite** é a premissa mais frágil e escala o resultado inteiro.

### 6.3. Eixo compartilhado nos painéis

Os cinco painéis compartilham o eixo vertical. Com escalas independentes, todo parâmetro pareceria igualmente íngreme e a pergunta do gráfico é justamente qual deles domina. Escala livre aqui não seria neutra: seria enganosa.

---

## 7. Decisões de engenharia

| Decisão | Razão |
|---|---|
| Simulação sobre o **conjunto de teste**, nunca o treino | O modelo já viu o treino; simular campanha ali produziria lucro de fantasia, com `p` otimista para os clientes que ele memorizou. |
| `graficos.py` recebe dados prontos, não calcula | Mesmo padrão da Fase 3: permite reusar as figuras no dashboard sem refazer a simulação. |
| `models/decisao.json` como fonte única | README, notebook e o dashboard da Fase 5 leem daí. Número copiado à mão envelhece calado. |
| Notebook 03 **lê artefatos**, não recalcula | Recalcular produziria números ligeiramente diferentes a cada execução, e a narrativa divergiria da tabela sem ninguém notar. |
| CLI com os cinco parâmetros | `--desconto 0.05` testa um cenário sem editar código. As premissas são para ser questionadas. |
| `_reais()` formata só os números | A primeira versão aplicava `.replace(",", ".")` na frase inteira e comia a vírgula da própria sentença. |

### 7.1. Decisões de forma nas figuras

- **`limiar_por_cliente`** — dispersão com a curva do limiar por cima. É a única forma que mostra *de uma vez* que o corte não é uma linha horizontal; uma tabela de limiares por faixa diria o mesmo e não convenceria ninguém. A legenda saiu para fora do eixo porque, no canto superior direito, cobria justamente os clientes de maior risco.
- **`curva_orcamento`** — as quatro curvas convergem no extremo direito, como têm de convergir: contatando a base inteira, a ordem deixa de importar. Isso funciona como verificação visual embutida.
- **`piso_do_desconto`** — duas linhas verticais depois da descoberta da §3.3: laranja onde o lucro zera de fato, cinza no limite teórico.

---

## 8. O que **não** foi feito, e por quê

| Não feito | Razão |
|---|---|
| **Modelo de uplift** | É o que a fase deveria usar (§9.1), e é impossível: exige grupo de controle aleatorizado, que esta base não tem. |
| **Otimizar o desenho da oferta** | A §3 mostra que o desconto domina o resultado, o que sugere resolver para o desconto ótimo. Não foi feito porque a taxa de aceite quase certamente **depende** do desconto, e oferta maior é mais aceita, e essa relação não é observável aqui. Otimizar com `taxa_aceite` fixa daria a resposta absurda "ofereça o menor desconto possível". |
| **Ofertas diferenciadas por segmento** | Mesma razão: sem dado de aceite por segmento, seria inventar estrutura. |
| **Restrição de capacidade** | A curva de orçamento já responde a qualquer corte de budget; formalizar como problema de otimização com restrição não acrescentaria nada, já que a solução continua sendo "ordene por VE e corte". |
| **Valor de vida do cliente (LTV)** | Substituiria o horizonte fixo por um modelo de sobrevivência. Mais correto e fora de escopo e `tenure` truncado em 72 (censura à direita, ver EDA) complicaria a estimativa. |
| **Intervalo de confiança sobre o lucro** | O lucro herda incerteza de `p` **e** dos cinco parâmetros. A segunda domina de longe a primeira, e propagá-la daria um intervalo tão largo que a sensibilidade da §6 comunica melhor. |

---

## 9. Limitações

### 9.1. A ausência de uplift é o limite estrutural

O modelo prevê `P(churn)`. A campanha precisaria de `P(churn | não contatado) − P(churn | contatado)` — o efeito **incremental** de abordar.

Os dois só coincidem se a oferta funcionar igualmente bem em todo mundo, o que não há razão para supor. Em campanhas reais, quem tem risco altíssimo costuma já estar decidido e não ser recuperável; quem tem risco médio é mais sensível a uma oferta. Isso significa que a lista ordenada por `p` pode estar **sistematicamente mirando as pessoas erradas**, e nada nesta análise detectaria.

Medir incrementalidade exige grupo de controle aleatorizado. Esta base tem 7.043 clientes observados, sem tratamento, sem aleatorização e sem histórico de campanha.

### 9.2. Sleeping dogs

Parte dos clientes cancela **porque** foi lembrada de que podia. O efeito é contrário ao objetivo e invisível sem experimento — para esses, contatar é pior que não fazer nada. A conta atual assume, implicitamente, que ele não existe.

### 9.3. As demais

- **`taxa_aceite` é palpite** e escala toda a magnitude do resultado.
- **Retenção tratada como permanente dentro do horizonte.** Nada impede o cliente retido de cancelar no mês seguinte ao fim do desconto.
- **`p` é probabilidade na janela do rótulo**, não taxa de risco mensal. Tratá-la como válida ao longo de `H` meses é aproximação.
- **O otimismo do modelo é herdado.** Sem split temporal (ver o documento da modelagem, §2.1), `p` é melhor do que seria em produção — e o lucro calculado herda isso.
- **Aceitar ⇒ retido**, por simplificação. Na prática, parte de quem aceita a oferta cancela mesmo assim.

### 9.4. A conclusão que essas limitações impõem

Somadas, elas determinam o que este resultado **é** e o que ele **não é**.

Ele **não** é um número para levar ao comitê de orçamento. Ele **é** o dimensionamento de um piloto: quantos clientes, quais, com que oferta, e qual retorno esperar se as premissas se confirmarem. O passo seguinte correto é um teste com grupo de controle aleatorizado que meça aceite real e churn incremental nos dois braços e substitua três dos cinco palpites por medida.

---

## 10. Conclusões

**1. O desenho da oferta importa mais que o modelo.** O limiar tem um piso `desconto/margem` que nenhuma qualidade de modelo vence, e o lucro é muito mais sensível ao desconto do que a qualquer ganho de AUC plausível.

**2. A campanha morre antes do limite teórico, em 25% de desconto e não 30%  porque o limite real depende da maior probabilidade que o modelo produz.

**3. O limiar é uma função, não um número.** De 0,904 a 0,423 conforme a mensalidade, movendo ~9% da campanha em relação à ordenação por probabilidade.

**4. A regra de contrato dá lucro negativo** no orçamento ótimo: identifica o segmento, não ordena dentro dele.

**5. O ganho da camada sobre a probabilidade pura é modesto (+3%).** O que ela acrescenta é saber onde parar e se a campanha deveria existir — nenhum dos dois mensurável como delta de lucro a orçamento fixo.

**6. Sem uplift, isto dimensiona um piloto e não autoriza uma campanha.**
