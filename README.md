# RoadEdge-BR

<p align="center">
  <b>Detecção de patologias asfálticas com Visão Computacional e Edge AI</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Em%20Desenvolvimento-orange">
  <img src="https://img.shields.io/badge/YOLO-YOLOv8n-yellow">
  <img src="https://img.shields.io/badge/Edge%20AI-TFLite-blue">
  <img src="https://img.shields.io/badge/Mobile-Flutter-02569B">
  <img src="https://img.shields.io/badge/Android-Testado-green">
</p>

> Trabalho de Conclusão de Curso (TCC) — Engenharia de Software | Universidade de Pernambuco (UPE)

## Sobre o projeto

O **RoadEdge-BR** investiga a utilização de Inteligência Artificial em dispositivos Edge para detecção e, futuramente, mapeamento de patologias asfálticas utilizando smartphones.

A proposta é executar modelos de visão computacional diretamente no dispositivo móvel, reduzindo a dependência de processamento em nuvem e permitindo investigar uma solução de baixo custo para monitoramento viário.

O projeto está sendo desenvolvido de forma incremental. A etapa atual concentra-se no módulo de **detecção visual de potholes (panelas/buracos)** e na implantação do modelo em um smartphone Android real.

A arquitetura planejada também prevê a utilização futura de:

- acelerômetro;
- GPS;
- classificação de sinais inerciais;
- fusão entre evidências visuais e inerciais;
- avaliação em trajetos reais.

Esses componentes ainda não representam funcionalidades finalizadas do sistema.

---

## Estado atual

Até o momento, o projeto já possui:

- auditoria e preparação do dataset;
- detector YOLOv8n treinado para potholes;
- pipeline de validação separado do conjunto de teste;
- experimentos com FP32, INT8 e W8A16;
- comparação de diferentes resoluções de entrada;
- aplicação Android desenvolvida em Flutter;
- execução do modelo em dispositivo físico;
- benchmarks em CPU e GPU;
- benchmark sustentado com 2.000 inferências;
- inicialização da câmera traseira e preview ao vivo.

A próxima etapa é integrar os frames da câmera diretamente ao modelo YOLO.

---

## Dataset

Os experimentos utilizam o dataset público **New Pothole Detection**, disponibilizado por meio do Roboflow Universe.

O conjunto utilizado possui:

| Conjunto | Imagens |
|---|---:|
| Treino | 6.091 |
| Validação | 2.094 |
| Teste | 1.055 |
| **Total** | **9.240** |

Durante a auditoria foram identificadas diferentes nomenclaturas relacionadas a potholes.

Para a etapa atual, o problema foi padronizado como detecção de uma única classe:

```text
pothole
```

A validação preparada para esse experimento contém:

```text
2.094 imagens
5.214 bounding boxes de potholes
```

O conjunto de teste permanece reservado para a avaliação final.

---

## Modelo de visão computacional

O detector utilizado atualmente é baseado em:

```text
YOLOv8n
Object Detection
1 classe: pothole
```

O modelo FP32 em resolução 640 × 640 foi utilizado como referência de qualidade.

### Baseline FP32 640

| Métrica | Resultado |
|---|---:|
| Precision | 0,7936 |
| Recall | 0,6830 |
| mAP@50 | 0,7626 |
| mAP@50-95 | 0,4299 |

---

## Experimentos de otimização Edge

A implantação no smartphone envolveu diferentes estratégias.

Foram avaliados:

- FP32;
- INT8;
- W8A16;
- diferentes resoluções de entrada;
- CPU;
- GPU.

Um dos principais resultados observados foi que **reduzir o tamanho do modelo não significa necessariamente reduzir a latência no hardware alvo**.

### INT8

A quantização INT8 reduziu o modelo para aproximadamente:

```text
3,17 MB
```

Porém, houve degradação relevante da qualidade.

```text
mAP@50-95 FP32 640: 0,4299
mAP@50-95 INT8:     0,3361
```

Uma segunda calibração controlada com 1.000 imagens resultou em:

```text
mAP@50-95: 0,3224
```

Por isso, o INT8 não foi selecionado como candidato atual.

### W8A16

O W8A16 preservou praticamente toda a qualidade:

```text
Tamanho: ~3,22 MB
mAP@50-95: 0,4286
```

Entretanto, a representação apresentou incompatibilidade com o backend GPU utilizado no dispositivo e desempenho inadequado em CPU.

Isso levou à investigação de uma alternativa: manter FP32 e reduzir a resolução de entrada.

---

## Comparação de resoluções

Todos os modelos abaixo derivam do mesmo detector treinado, alterando principalmente a resolução utilizada na exportação/inferência.

| Resolução | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| 320 × 320 | 0,7347 | 0,5610 | 0,6353 | 0,3209 |
| 384 × 384 | 0,7763 | 0,6126 | 0,6969 | 0,3706 |
| 416 × 416 | 0,7791 | 0,6358 | 0,7130 | 0,3873 |
| 640 × 640 | 0,7936 | 0,6830 | 0,7626 | 0,4299 |

A redução da resolução diminui o custo computacional, mas também reduz a qualidade de detecção.

---

## Candidato Edge atual

O melhor compromisso encontrado até o momento para o dispositivo avaliado foi:

```text
YOLOv8n
FP32
384 × 384
GPU
```

O dispositivo utilizado nos testes foi um:

```text
Motorola moto g14
Android 14
arm64
```

Na validação:

```text
Precision:   0,7763
Recall:      0,6126
mAP@50:      0,6969
mAP@50-95:   0,3706
```

A escolha de 384 × 384 é **provisória** e representa o melhor compromisso encontrado até esta etapa entre qualidade e latência.

---

## Benchmark sustentado no smartphone

Para avaliar a estabilidade do candidato 384 × 384, foi realizado um teste com:

```text
5 warm-ups
20 blocos
100 inferências por bloco
2.000 inferências medidas
GPU
```

Resultados globais:

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

Não foi observada degradação progressiva relevante de latência durante o período medido.

O FPS apresentado é apenas uma taxa equivalente calculada a partir da latência de inferência. Ele ainda **não representa o FPS final do pipeline de câmera**.

Também não foi realizada medição direta da temperatura do dispositivo, portanto o resultado não deve ser interpretado como uma avaliação térmica completa.

---

## Aplicação Android

Foi criada uma aplicação utilizando **Flutter** para executar os experimentos diretamente no smartphone.

O ambiente móvel atualmente permite:

```text
Flutter
   |
   +-- TensorFlow Lite / LiteRT
   |
   +-- YOLOv8n
   |
   +-- CPU / GPU
   |
   +-- Câmera Android
```

A aplicação já foi utilizada para:

- carregar modelos TFLite;
- executar inferências no dispositivo;
- visualizar bounding boxes;
- medir latência;
- comparar CPU e GPU;
- executar benchmarks sustentados.

---

## Integração da câmera

A câmera traseira do moto g14 já foi inicializada pelo aplicativo Flutter.

Durante o teste foram identificadas três câmeras e selecionada a câmera traseira:

```text
===== ROADEDGE CÂMERA =====
Câmeras encontradas: 3
Câmera utilizada: 0
Direção: CameraLensDirection.back
Preview inicializado com sucesso.
===========================
```

### Status atual do pipeline

Atualmente:

```text
Câmera Android
      |
      v
Preview Flutter
      |
      v
Funcionando
```

Próxima implementação:

```text
Câmera
   |
   v
Frame
   |
   v
Pré-processamento
   |
   v
YOLOv8n FP32 384 × 384
   |
   v
Detecções
   |
   v
Bounding boxes no preview
```

O modelo ainda não está executando diretamente sobre os frames da câmera neste marco do projeto.

---

## Critérios de engenharia

Durante o desenvolvimento foram adotados critérios de referência para orientar os experimentos Edge:

```text
P95 <= 100 ms
modelo <= 10 MB
perda após quantização <= 3 pontos percentuais de mAP@50-95
```

Esses valores são critérios definidos para este estudo e não devem ser interpretados como requisitos universais.

O candidato FP32 384 × 384 atende numericamente ao objetivo de P95 no benchmark GPU, porém:

- o critério original de latência foi definido para CPU;
- o modelo FP32 permanece com aproximadamente 11,6 MB.

Portanto, o candidato atual ainda não satisfaz simultaneamente todos os critérios estabelecidos.

---

## Estrutura do repositório

```text
Deteccao_de_Patologias_Asfalticas_com_IA/
|
+-- configs/
|   +-- dataset.yaml
|   +-- experiment.yaml
|   +-- calibration.yaml
|
+-- docs/
|   +-- RESULTADOS_EXPERIMENTAIS.md
|
+-- mobile/
|   +-- android/
|   +-- assets/
|   +-- lib/
|   +-- pubspec.yaml
|
+-- src/
|   +-- roadedge/
|       +-- common.py
|       +-- data_audit.py
|       +-- split_groups.py
|       +-- train_vision.py
|       +-- evaluate_vision.py
|       +-- select_threshold.py
|       +-- select_pareto.py
|       +-- export_edge.py
|       +-- benchmark_local.py
|       +-- sensor_features.py
|       +-- train_sensor.py
|       +-- forest_json.py
|       +-- fusion.py
|
+-- DetecçãoDeBuracos.ipynb
+-- pyproject.toml
+-- README.md
```

O notebook representa a fase inicial/prototipação do projeto. A implementação modular em `src/roadedge` representa a evolução da estrutura experimental.

---

## Como executar o aplicativo Android

### Requisitos

- Flutter instalado;
- Android SDK configurado;
- dispositivo Android com depuração USB ativada.

Entre na pasta:

```bash
cd mobile
```

Verifique os dispositivos:

```bash
flutter devices
```

Instale as dependências:

```bash
flutter pub get
```

Execute no dispositivo:

```bash
flutter run -d <DEVICE_ID>
```

Para benchmarks de desempenho, utilize o modo Profile:

```bash
flutter run --profile -d <DEVICE_ID>
```

> O arquivo do modelo TFLite pode não estar versionado no repositório. Nesse caso, ele deve ser disponibilizado separadamente no diretório de assets correspondente antes da execução das etapas que utilizam inferência.

---

## Resultados experimentais completos

A evolução dos experimentos, incluindo:

- preparação do dataset;
- baseline;
- INT8;
- W8A16;
- benchmarks;
- comparação de resoluções;
- benchmark sustentado;
- integração inicial da câmera;
- limitações;

está documentada em:

[Resultados Experimentais](docs/RESULTADOS_EXPERIMENTAIS.md)

---

## Próximas etapas

- [x] Treinamento do detector visual
- [x] Validação do modelo FP32
- [x] Experimentos de quantização
- [x] Implantação inicial no Android
- [x] Benchmark CPU/GPU
- [x] Comparação de resoluções
- [x] Benchmark sustentado
- [x] Inicialização da câmera
- [ ] Inferência YOLO sobre frames da câmera
- [ ] Bounding boxes no preview ao vivo
- [ ] Benchmark do pipeline completo
- [ ] Testes em vias reais
- [ ] Definição final do threshold
- [ ] Avaliação final no conjunto de teste
- [ ] Integração do acelerômetro
- [ ] Classificador de sinais inerciais
- [ ] Integração GPS
- [ ] Fusão visual + inercial

---

## Limitações atuais

O projeto ainda está em fase experimental.

Os resultados atuais não devem ser interpretados como evidência de um sistema final pronto para produção.

Entre as principais limitações atuais estão:

- avaliação móvel realizada em um único modelo de smartphone;
- pipeline câmera + inferência ainda não integrado;
- ausência de avaliação térmica direta;
- ausência de medição de consumo energético;
- conjunto de teste ainda reservado;
- módulo inercial ainda não integrado ao aplicativo;
- fusão sensorial ainda não avaliada em campo.

---

## Autor

**Mikhael Vinicius**

- [LinkedIn](https://linkedin.com/in/mikhaelvincius)
- [Portfólio](https://siteportifolio-gules.vercel.app/)