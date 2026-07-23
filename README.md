<h1 align="center">🛣️ Monitoramento Rodoviário Inteligente | Edge AI & Visão Computacional</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Em%20Desenvolvimento-orange">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg">
  <img src="https://img.shields.io/badge/YOLO-v8_Nano-yellow.svg">
  <img src="https://img.shields.io/badge/TensorFlow-Lite-FF6F00?logo=tensorflow">
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-brightgreen">
</p>

> **Trabalho de Conclusão de Curso (TCC)** - Engenharia de Software | Universidade de Pernambuco (UPE)

## 📋 Sobre o Projeto

A precariedade da malha rodoviária impacta diretamente a economia e a segurança viária. Segundo a **Lei de Viana**, o custo de reparo de patologias asfálticas cresce exponencialmente com o tempo. A detecção rápida e eficiente é um fator crítico.

Este projeto propõe o desenvolvimento de um **sistema inteligente híbrido e de baixo custo** para o monitoramento e mapeamento automatizado de vias. Utilizando dispositivos móveis comuns (smartphones), o sistema aplica o conceito de **Inteligência Artificial de Borda (Edge AI)** para processar dados 100% offline, transformando frotas veiculares em agentes ativos de zeladoria urbana.

## 🎯 Diferencial: Fusão Sensorial Híbrida

Sistemas puramente visuais sofrem com falsos positivos (ex: sombras projetadas interpretadas como buracos). Para solucionar este gargalo tecnológico, este projeto integra duas frentes:
1. **Visão Computacional (Câmera):** Identifica visualmente a patologia no asfalto.
2. **Análise Vibracional (Acelerômetro):** Valida fisicamente o impacto mecânico sofrido pelo veículo.

*Uma patologia só é registrada e georreferenciada no sistema caso ocorra a convergência simultânea da imagem (câmera) e do impacto físico (acelerômetro).*

## 🚀 Tecnologias Utilizadas

*   **Visão Computacional:** YOLOv8 Nano (Ultralytics)
*   **Otimização Edge AI:** Quantização INT8, formato TensorFlow Lite (TFLite)
*   **Análise de Dados Inerciais:** Random Forest (Algoritmo de Aprendizagem Supervisionada)
*   **Sincronização:** Timestamp e Georreferenciamento (GPS)
*   **Dataset:** "New Pothole Detection" (Roboflow Universe)

## 📊 Resultados Alcançados

- **Acurácia Visual:** O modelo YOLOv8n treinado alcançou **78,8% de mAP** na detecção de cavidades asfálticas.
- **Alta Compressão (Edge AI):** Através da quantização INT8, o tamanho original do modelo de 12.5 MB foi reduzido para apenas **3.2 MB**, viabilizando a execução fluida e offline diretamente na CPU de smartphones comuns.
- **Mitigação de Latência:** Transição do modelo de inferência em nuvem para processamento estritamente local, eliminando consumo de dados móveis.

## 🔮 Próximos Passos

- [ ] Acoplamento algorítmico da Fusão Sensorial (Timestamp + GPS).
- [ ] Implementação de alternância de estado de energia (ativar a câmera apenas após pico vibracional captado pelo Random Forest) para poupar bateria.
- [ ] Testes experimentais de tráfego em campo.

## 👨‍💻 Autor

**Mikhael Vinicius**
*   [LinkedIn](https://linkedin.com/in/mikhaelvincius)
*   [Portfólio](https://siteportifolio-gules.vercel.app/)
