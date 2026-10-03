# RoadEdge-BR — Resultados Experimentais

Este documento registra a evolução experimental do módulo de visão computacional do RoadEdge-BR, desde o protótipo inicial até os primeiros testes de implantação Edge em um smartphone Android.

> **Status atual:** o sistema ainda está em desenvolvimento. Os resultados apresentados aqui correspondem a experimentos realizados durante o TCC e não representam ainda a avaliação final do sistema completo.

---

## 1. Objetivo

O RoadEdge-BR tem como objetivo investigar a utilização de Inteligência Artificial em dispositivos Edge para identificação de patologias asfálticas.

Nesta etapa do projeto, o foco do módulo de visão computacional foi a detecção de **potholes (panelas/buracos)** utilizando YOLO.

A arquitetura planejada do projeto também prevê, em etapas posteriores:

- dados de acelerômetro;
- localização por GPS;
- classificação dos sinais inerciais;
- fusão entre evidências visuais e inerciais;
- avaliação em trajetos reais.

Esses componentes ainda não fazem parte dos resultados apresentados neste documento.

---

## 2. Protótipo inicial

A primeira versão do projeto foi desenvolvida em um notebook experimental utilizando YOLOv8n.

O treinamento inicial utilizou:

- YOLOv8n;
- transfer learning;
- imagens redimensionadas para 640 × 640;
- GPU NVIDIA T4 no Google Colab;
- dataset público de detecção de buracos.

O protótipo apresentou aproximadamente:

| Métrica | Resultado |
|---|---:|
| Precision | 0,805 |
| Recall | 0,711 |
| mAP@50 | 0,788 |
| mAP@50-95 | 0,467 |

É importante destacar que **0,788 representa mAP@50 e não acurácia de 78,8%**.

O notebook inicial serviu como prova de conceito, mas ainda não possuía uma metodologia adequada para avaliação final ou implantação Edge.

---

## 3. Auditoria do dataset

O dataset utilizado contém aproximadamente **9.240 imagens**.

A distribuição utilizada foi:

| Conjunto | Imagens |
|---|---:|
| Treino | 6.091 |
| Validação | 2.094 |
| Teste | 1.055 |
| **Total** | **9.240** |

Durante a auditoria foram identificados problemas de nomenclatura das classes.

O dataset continha rótulos como:

- `pothole`;
- `Pothole`;
- `potholes`;
- `Potholes`;
- `porthole`;
- `bache`;
- `0`;
- `-1`;
- `object`;
- `manhole`.

A inspeção mostrou que várias dessas classes representavam o mesmo objeto de interesse: buracos no pavimento.

Para os experimentos atuais, o problema foi simplificado para uma única classe:

```text
pothole
```

As classes `object` e `manhole` não foram consideradas automaticamente como potholes.

Após a preparação para o experimento de uma classe, a validação utilizada continha:

- 2.094 imagens;
- 5.214 bounding boxes de potholes.

O conjunto de teste permanece reservado para a avaliação final.

---

## 4. Baseline reproduzido

Foi realizado um novo treinamento do YOLOv8n por 30 épocas.

Durante o treinamento foi observada evolução consistente das métricas.

Por exemplo:

| Época | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| 1 | 0,438 | 0,343 | 0,324 | 0,132 |
| 30 | 0,797 | 0,675 | 0,769 | 0,439 |

Esse resultado confirmou que o modelo estava efetivamente aprendendo o problema de detecção.

---

## 5. Baseline FP32 — 640 × 640

O modelo treinado foi exportado para TensorFlow Lite em FP32.

Na validação de uma classe foram obtidos:

| Métrica | Resultado |
|---|---:|
| Precision | 0,7936 |
| Recall | 0,6830 |
| mAP@50 | 0,7626 |
| mAP@50-95 | 0,4299 |

O arquivo possui aproximadamente **11,69 MiB**.

Esse modelo passou a ser utilizado como referência de qualidade para os experimentos de implantação Edge.

---

## 6. Critérios de engenharia

Foram definidos critérios de engenharia para orientar a escolha do modelo Edge.

Entre os critérios avaliados estão:

- latência P95;
- tamanho do modelo;
- perda de qualidade após otimização;
- estabilidade da execução no dispositivo.

Os valores de referência utilizados nesta etapa são:

- P95 ≤ 100 ms;
- modelo ≤ 10 MB;
- perda de mAP@50-95 após quantização ≤ 3 pontos percentuais.

Esses valores são **critérios definidos para este estudo**, e não regras universais para aplicações de visão computacional.

Além disso, o critério original de latência foi definido considerando execução em CPU. Alguns dos experimentos posteriores utilizaram GPU no smartphone e, portanto, devem ser interpretados separadamente.

---

## 7. Quantização INT8

O primeiro experimento de quantização INT8 reduziu significativamente o tamanho do modelo.

Resultado:

| Característica | Resultado |
|---|---:|
| Tamanho | ~3,17 MB |
| Precision | 0,7266 |
| Recall | 0,6401 |
| mAP@50 | 0,6735 |
| mAP@50-95 | 0,3361 |

Comparando com o FP32 640:

```text
FP32:  mAP@50-95 = 0,4299
INT8:  mAP@50-95 = 0,3361
```

A perda foi de aproximadamente **9,38 pontos percentuais**.

Apesar da redução de tamanho, a perda de qualidade ultrapassou o limite estabelecido para o projeto.

---

## 8. INT8 com calibração controlada

Para verificar se a degradação estava relacionada à calibração, foi realizado um segundo experimento utilizando:

- 1.000 imagens de treino;
- seed 42;
- conjunto representativo controlado.

Durante essa etapa também foram encontrados labels em formatos diferentes, incluindo polígonos.

Foram convertidos **139 polígonos para bounding boxes**.

Após o tratamento, o conjunto de calibração apresentou:

- 2.530 bounding boxes;
- 28 labels vazios;
- 0 labels inválidos.

O novo INT8 apresentou:

| Métrica | Resultado |
|---|---:|
| Precision | 0,7100 |
| Recall | 0,6481 |
| mAP@50 | 0,6553 |
| mAP@50-95 | 0,3224 |

A perda em relação ao FP32 foi ainda maior, aproximadamente **10,75 pontos percentuais**.

Portanto, o INT8 não foi selecionado como candidato atual.

---

## 9. Quantização W8A16

Também foi testada a representação W8A16.

Após corrigir o conjunto de calibração, foram obtidos:

| Característica | Resultado |
|---|---:|
| Tamanho | ~3,22 MB |
| Precision | 0,7905 |
| Recall | 0,6808 |
| mAP@50 | 0,7587 |
| mAP@50-95 | 0,4286 |

A diferença de mAP@50-95 em relação ao FP32 640 foi de apenas:

```text
0,4299 - 0,4286 = 0,0013
```

ou aproximadamente **0,13 ponto percentual**.

Do ponto de vista de qualidade e tamanho, esse resultado foi muito bom.

Entretanto, a avaliação no smartphone revelou problemas de compatibilidade e desempenho.

---

## 10. Ambiente Edge

O dispositivo utilizado para os testes foi:

```text
Motorola moto g14
Android 14
arquitetura arm64
```

Foi criado um aplicativo Android utilizando:

- Flutter;
- Dart;
- TensorFlow Lite/LiteRT por meio da integração utilizada pelo plugin YOLO;
- dispositivo físico conectado via USB.

O desenvolvimento foi realizado sem necessidade de emulador Android.

---

## 11. W8A16 no moto g14

Ao tentar executar o W8A16 utilizando GPU, o runtime apresentou incompatibilidade relacionada a operações com `INT16`.

Por isso, o modelo foi testado em CPU.

Resultado aproximado em modo Profile:

| Métrica | Resultado |
|---|---:|
| Média | 3363,26 ms |
| Mediana | 3311,59 ms |
| P95 | 3758,57 ms |
| FPS equivalente | 0,30 |

Apesar de preservar a qualidade e reduzir o tamanho, o modelo W8A16 mostrou desempenho inadequado no dispositivo utilizado.

Esse resultado reforçou que **reduzir o tamanho do modelo não garante automaticamente menor latência**.

---

## 12. FP32 640 no smartphone

O FP32 640 também foi avaliado no moto g14.

### CPU

| Métrica | Resultado |
|---|---:|
| Média | 325,75 ms |
| Mediana | 318,18 ms |
| P95* | 445,28 ms |
| FPS equivalente | 3,07 |

### GPU

| Métrica | Resultado |
|---|---:|
| Média | 207,94 ms |
| Mediana | 194,78 ms |
| P95* | 259,27 ms |
| FPS equivalente | 4,81 |

\* Esses experimentos iniciais utilizaram uma quantidade menor de execuções e uma implementação anterior do cálculo de percentis.

A GPU melhorou significativamente o desempenho, mas a latência ainda ficou acima do objetivo estabelecido.

---

## 13. Redução da resolução

Como a quantização não produziu simultaneamente qualidade e desempenho adequados no dispositivo, foi investigada outra estratégia:

**reduzir a resolução de entrada mantendo os pesos FP32.**

Foram avaliadas as resoluções:

- 640 × 640;
- 416 × 416;
- 384 × 384;
- 320 × 320.

A alteração da resolução reduz a quantidade de operações realizadas durante a inferência, mas não reduz significativamente o número de parâmetros do modelo.

Por isso, os modelos FP32 continuaram com aproximadamente 11,6 MB.

---

## 14. Comparação de qualidade

Resultados na mesma validação:

| Resolução | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| 320 | 0,7347 | 0,5610 | 0,6353 | 0,3209 |
| 384 | 0,7763 | 0,6126 | 0,6969 | 0,3706 |
| 416 | 0,7791 | 0,6358 | 0,7130 | 0,3873 |
| 640 | 0,7936 | 0,6830 | 0,7626 | 0,4299 |

Como esperado, resoluções menores reduziram a qualidade de detecção.

Entretanto, também reduziram o custo computacional.

---

## 15. FP32 320 × 320

No moto g14, utilizando GPU e um benchmark corrigido com 100 inferências medidas:

| Métrica | Resultado |
|---|---:|
| Média | 70,71 ms |
| Mediana | 70,16 ms |
| P95 | 78,32 ms |
| P99 | 86,66 ms |
| Mínimo | 67,75 ms |
| Máximo | 88,94 ms |
| FPS equivalente | 14,14 |

O modelo apresentou excelente desempenho de inferência no dispositivo.

Porém, seu mAP@50-95 foi:

```text
0,3209
```

representando uma perda considerável de qualidade em relação ao modelo 640.

---

## 16. FP32 416 × 416

O modelo 416 recuperou parte da qualidade:

```text
mAP@50-95 = 0,3873
```

Entretanto, no benchmark inicial no moto g14:

| Métrica | Resultado |
|---|---:|
| Média | 107,08 ms |
| P95* | 135,51 ms |
| FPS equivalente | 9,34 |

\* Benchmark inicial com 20 execuções e cálculo anterior de percentil.

O resultado indicou que 416 estava próximo da região de interesse, porém acima do objetivo de latência nesse teste.

---

## 17. FP32 384 × 384

Foi então avaliada uma resolução intermediária de **384 × 384**.

Qualidade na validação:

| Métrica | Resultado |
|---|---:|
| Precision | 0,7763 |
| Recall | 0,6126 |
| mAP@50 | 0,6969 |
| mAP@50-95 | 0,3706 |

Benchmark corrigido no moto g14 utilizando:

- GPU;
- 5 warm-ups;
- 100 inferências medidas;
- percentis calculados por interpolação linear.

Resultado:

| Métrica | Resultado |
|---|---:|
| Média | 86,64 ms |
| Mediana | 85,50 ms |
| P95 | 95,50 ms |
| P99 | 103,94 ms |
| Mínimo | 81,71 ms |
| Máximo | 109,99 ms |
| FPS equivalente | 11,54 |

O modelo 384 apresentou um compromisso melhor entre qualidade e latência do que o modelo 320.

Por esse motivo, ele foi escolhido como **candidato provisório para os próximos experimentos Edge**.

Essa escolha não representa ainda a seleção definitiva do modelo final do TCC.

---

## 18. Correção do cálculo de percentis

Os benchmarks iniciais utilizavam uma abordagem baseada em arredondamento para selecionar o índice do percentil.

Com apenas 20 execuções, isso fazia com que P95 e P99 frequentemente coincidissem com o maior tempo observado.

A implementação foi posteriormente alterada para utilizar interpolação linear.

Além disso, os benchmarks principais passaram a utilizar uma quantidade maior de inferências.

Isso tornou as estimativas de P95 e P99 mais representativas.

---

## 19. Benchmark sustentado

Para verificar se o desempenho do modelo 384 permanecia estável durante uma execução mais longa, foi realizado um teste sustentado.

Configuração:

```text
Dispositivo: moto g14
Modelo: YOLOv8n FP32
Entrada: 384 × 384
Backend: GPU
Warm-ups: 5
Inferências medidas: 2.000
Blocos: 20
Inferências por bloco: 100
```

Resultado global:

| Métrica | Resultado |
|---|---:|
| Duração | 2,82 min |
| Média | 84,58 ms |
| Mediana | 84,44 ms |
| P95 | 88,53 ms |
| P99 | 95,11 ms |
| Mínimo | 80,19 ms |
| Máximo | 111,04 ms |
| FPS equivalente | 11,82 |

Durante os 20 blocos, as médias permaneceram aproximadamente na faixa de 84–86 ms.

Não foi observada degradação progressiva relevante de latência durante os 2,82 minutos do experimento.

Entretanto, **não foi realizada medição direta de temperatura ou estado térmico do processador**. Portanto, esse experimento não é suficiente para afirmar definitivamente que não ocorreu thermal throttling.

---

## 20. Interpretação do P95

P95 representa o valor abaixo do qual aproximadamente 95% das medições se encontram.

No benchmark sustentado:

```text
P95 = 88,53 ms
```

Isso significa que aproximadamente 95% das inferências medidas terminaram em até 88,53 ms.

A utilização de P95 evita avaliar o sistema apenas pela média e ajuda a observar a consistência da latência.

---

## 21. FPS equivalente

O FPS apresentado nos benchmarks é calculado aproximadamente por:

```text
FPS equivalente = 1000 / latência média em milissegundos
```

No benchmark sustentado:

```text
1000 / 84,58 ≈ 11,82
```

Esse número representa apenas a **taxa equivalente de inferência do modelo**.

Ele **não representa ainda o FPS real da aplicação utilizando câmera**, pois o pipeline completo também possui custos de:

- captura do frame;
- conversão de formato;
- pré-processamento;
- inferência;
- pós-processamento;
- renderização das bounding boxes.

---

## 22. Comparação resumida dos principais modelos

| Modelo | Tamanho aprox. | mAP@50-95 | Backend no celular | Latência / situação |
|---|---:|---:|---|---|
| FP32 640 | 11,69 MB | 0,4299 | CPU | P95 inicial 445,28 ms |
| FP32 640 | 11,69 MB | 0,4299 | GPU | P95 inicial 259,27 ms |
| INT8 | 3,17 MB | 0,3361 | — | perda elevada de qualidade |
| INT8 calib. 1000 | 3,17 MB | 0,3224 | — | perda elevada de qualidade |
| W8A16 640 | 3,22 MB | 0,4286 | CPU | P95 inicial 3758,57 ms |
| FP32 416 | ~11,63 MB | 0,3873 | GPU | P95 inicial 135,51 ms |
| FP32 384 | ~11,63 MB | 0,3706 | GPU | P95 88,53 ms no teste sustentado |
| FP32 320 | ~11,62 MB | 0,3209 | GPU | P95 78,32 ms |

Os benchmarks marcados como iniciais não possuem exatamente o mesmo protocolo dos experimentos posteriores e não devem ser interpretados como uma comparação experimental perfeitamente controlada.

---

## 23. Por que 384 × 384 é o candidato atual?

O modelo 320 apresentou a menor latência, porém com maior perda de qualidade.

O modelo 416 recuperou qualidade, mas ultrapassou o objetivo de latência no benchmark inicial.

O modelo 384 ficou entre os dois:

```text
mAP@50-95 = 0,3706
P95 sustentado = 88,53 ms
```

Por isso, atualmente ele representa o compromisso mais interessante encontrado entre qualidade e velocidade para o moto g14.

Entretanto, existem duas limitações importantes:

1. o benchmark utiliza GPU, enquanto o critério original de 100 ms foi definido para CPU;
2. o modelo possui aproximadamente 11,6 MB, ultrapassando o objetivo de 10 MB.

Portanto, nenhum resultado deve ser interpretado como atendimento completo de todos os critérios definidos para o projeto.

---

## 24. Integração inicial da câmera

Após os benchmarks com imagem estática, foi iniciada a integração com a câmera real do smartphone.

Foi adicionada ao aplicativo Flutter a dependência de câmera e a permissão correspondente no Android.

O aplicativo conseguiu:

- identificar as câmeras disponíveis;
- encontrar 3 câmeras no dispositivo;
- selecionar a câmera traseira;
- abrir a câmera;
- inicializar o preview ao vivo.

Log observado:

```text
===== ROADEDGE CÂMERA =====
Câmeras encontradas: 3
Câmera utilizada: 0
Direção: CameraLensDirection.back
Preview inicializado com sucesso.
===========================
```

Neste marco do desenvolvimento, **o YOLO ainda não está processando os frames da câmera**.

A etapa atual comprova apenas que o pipeline de captura da câmera foi inicializado corretamente.

---

## 25. Estado atual

O candidato provisório do módulo visual é:

```text
YOLOv8n
FP32
384 × 384
GPU
moto g14
```

Resultados principais:

```text
mAP@50-95: 0,3706

Benchmark sustentado:
2.000 inferências
média: 84,58 ms
mediana: 84,44 ms
P95: 88,53 ms
P99: 95,11 ms
```

Além disso, a câmera traseira do dispositivo já foi integrada ao aplicativo Flutter e o preview ao vivo foi validado.

---

## 26. Próxima etapa

A próxima etapa será conectar o fluxo da câmera ao modelo YOLO.

O pipeline pretendido é:

```text
Câmera
   ↓
Frame
   ↓
Pré-processamento
   ↓
YOLOv8n FP32 384 × 384
   ↓
Pós-processamento
   ↓
Bounding boxes
   ↓
Preview
```

Como a câmera pode produzir frames mais rapidamente do que o modelo consegue processá-los, será necessário impedir inferências concorrentes.

A estratégia prevista é:

```text
Novo frame
    ↓
Modelo está processando?
    ├── Sim → descartar o frame
    └── Não → executar inferência
```

Isso evita a formação de uma fila crescente de frames antigos.

---

## 27. Benchmark futuro do pipeline completo

Após a integração da câmera, serão medidas separadamente:

1. latência de inferência;
2. custo de preparação/conversão do frame;
3. pós-processamento;
4. latência do pipeline completo.

Isso é importante porque:

```text
latência da inferência ≠ latência total da aplicação
```

Por exemplo, um modelo com aproximadamente 85 ms de inferência pode apresentar uma latência total maior quando os demais componentes forem incluídos.

---

## 28. Avaliação em ambiente real

Após a integração câmera + YOLO, serão necessários testes exploratórios em cenários reais.

Entre os casos relevantes estão:

- pavimento sem buracos;
- buracos próximos;
- buracos distantes;
- sombras;
- remendos no asfalto;
- diferentes condições de iluminação;
- diferentes velocidades de deslocamento.

Esses testes serão utilizados inicialmente para identificar comportamentos e falhas do sistema.

Eles não substituem a avaliação quantitativa final.

---

## 29. Trabalho futuro

As principais etapas ainda pendentes incluem:

- integração YOLO + câmera em tempo real;
- benchmark end-to-end;
- avaliação térmica mais longa;
- avaliação de memória e consumo energético;
- definição final do threshold utilizando validação;
- avaliação única no conjunto de teste;
- experimentos com acelerômetro;
- extração de features inerciais;
- classificação com Random Forest;
- fusão visual + inercial;
- integração com GPS;
- avaliação em trajetos reais.

---

## 30. Conclusão parcial

Os experimentos demonstraram que a implantação Edge não depende apenas da qualidade obtida durante o treinamento.

Durante o desenvolvimento foram observados diferentes compromissos:

- INT8 reduziu significativamente o tamanho, mas apresentou perda elevada de qualidade;
- W8A16 preservou a qualidade e reduziu o tamanho, porém apresentou problemas de compatibilidade/desempenho no dispositivo;
- FP32 640 preservou maior qualidade, mas apresentou latência elevada;
- FP32 320 apresentou excelente latência, mas maior perda de qualidade;
- FP32 384 apresentou até o momento o compromisso mais interessante entre qualidade e velocidade no dispositivo avaliado.

O resultado mais promissor desta etapa foi o YOLOv8n FP32 384 × 384 executado em GPU no moto g14, que apresentou:

```text
mAP@50-95 = 0,3706
P95 sustentado = 88,53 ms
P99 sustentado = 95,11 ms
```

em um teste de 2.000 inferências.

A câmera traseira também foi inicializada com sucesso no aplicativo.

O próximo marco será avaliar o comportamento do modelo dentro do **pipeline real de câmera**, permitindo comparar o benchmark isolado de inferência com o desempenho efetivo da aplicação.