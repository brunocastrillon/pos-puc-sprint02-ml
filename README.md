# Spotify Streams Classification MVP

Este repositório contém todas as partes programáveis do MVP de classificação de faixas do Spotify, desenvolvido na disciplina de Engenharia de Sistemas de Software Inteligentes (Pós-Graduação em Engenharia de Software).

## 🚀 Visão Geral

O projeto implementa um fluxo completo de **Machine Learning** e **entrega de software**:

1. **Notebook de EDA e Treinamento** (`ml_mvp_spotify_v2.ipynb`):

   * Exploração e visualização de dados
   * Pré-processamento e limpeza
   * Treinamento de múltiplos modelos (KNN, Decision Tree, Naive Bayes, SVM)
   * Otimização de hiperparâmetros com GridSearchCV
   * Avaliação de desempenho e exportação do artefato

2. **Script de Treinamento Local** (`model/train_model_v2.py`):

   * Mesma lógica do notebook, mas executável em linha de comando
   * Permite integração com IDEs (VS Code)

3. **API RESTful em Flask** (`backend/app.py`):

   * Carrega o artefato treinado (`.pkl`)
   * Endpoints para predição de conteúdo explícito
   * Endpoints para correlação de plataformas e métricas de artistas

4. **Frontend Estático** (`frontend/index.html` + `index.js`):

   * Interface Bootstrap minimalista
   * Formulário dinâmico para classificação de faixa (preenchimento manual ou via upload CSV/Excel)
   * Visualizações Plotly: heatmap de correlação e gráficos de bolhas

5. **Testes Automatizados PyTest** (`tests/test_model_performance.py`):

   * Validação de acurácia mínima e F1-score ponderado
   * Garante que o modelo atende requisitos antes de deploy

## 📁 Estrutura de Diretórios

```
pos-puc-sprint02-ml/
├── backend/                  # API Flask
│   └── app.py                # Endpoints de predição e métricas
├── frontend/                 # Arquivos estáticos (HTML, JS, CSS)
│   ├── index.html            # UI principal
│   └── index.js              # Lógica de chamada aos endpoints
├── model/                    # Scripts e artefatos de ML
│   ├── dataset/              # CSV original v2 do Kaggle
│   └── train_model_v2.py     # Script de treinamento local
├── tests/                    # Testes PyTest
│   └── test_model_performance.py
├── ml_mvp_spotify_v2.ipynb    # Notebook de EDA e treinamento
├── requirements.txt          # Dependências Python
└── README.md                 # Documentação (este arquivo)
```

## 🛠️ Pré-requisitos

* **Python 3.8+**
* **pip**
* Navegador moderno (para o Frontend estático)

> Opcional: criar um ambiente virtual (venv, conda)

## 📦 Instalação

1. Faça o clone:

   ```bash
   git clone https://github.com/brunocastrillon/pos-puc-sprint02-ml.git
   cd pos-puc-sprint02-ml
   ```
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

## 🧪 Executando o Notebook

* Abra o Colab:
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brunocastrillon/pos-puc-sprint02-ml/blob/master/ml_mvp_spotify_v2.ipynb)
* Execute todas as células para EDA, modelagem e exportação do artefato.

## ▶️ Treinamento Local

```bash
python model/train_model_v2.py
```

* Gera `artifact_spotify_mvp_v2.pkl` com:

  * Pipeline treinado
  * Matriz de correlação
  * Métricas de artista
  * Lista de features

## 🖥️ API Flask

1. Configure (opcional) um arquivo `.env` com:

   ```ini
   DATA_SOURCE=local
   DATA_FILE=model/dataset/most_streamed_spotify_songs_2024_v2.csv
   MODEL_PATH=artifact_spotify_mvp_v2.pkl
   ```
2. Inicie o servidor:

   ```bash
   cd backend
   python app.py
   ```
3. Endpoints:

   * `POST /predict-classified` → Classifica conteúdo explícito
   * `GET /platform-correlation` → Retorna matriz de correlação
   * `GET /artist-impact` → Retorna métricas de artista

## 🌐 Frontend

* Abra `frontend/index.html` diretamente no navegador.
* Preencha o formulário manualmente ou faça upload de CSV/Excel.
* Navegue pelas abas:

  * **Classificar**
  * **Platform Comparison**
  * **Artist Impact**

## ✅ Testes Automatizados

```bash
pytest --maxfail=1 -q
```

* Garante **acurácia ≥ 62%** e **F1-weighted ≥ 58%** antes de qualquer deploy.

## 🤝 Contribuição

Contribuições são bem-vindas! Abra issues ou pull requests para novas features ou correções.

## 📄 Licença

Este projeto está licenciado sob [MIT](LICENSE) — veja o arquivo de licença para detalhes.

---

*Desenvolvido por Bruno Castrillón*
