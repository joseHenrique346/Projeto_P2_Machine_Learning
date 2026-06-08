# Classificação da Qualidade de Vinhos Tintos

Projeto acadêmico de Machine Learning desenvolvido para classificar vinhos
tintos como **bons** ou **ruins/médios** utilizando suas propriedades
físico-químicas. O modelo treinado é disponibilizado por meio de uma aplicação
web desenvolvida com Streamlit.

## Integrantes

| Integrante | RA |
|---|---|
| Carlos Henrique Legutcke Filho | RA: 1988772 |
| José Henrique de Castro Fernandes | RA: 1994033 |
| Leonardo dos Santos da Silva | RA: 2034122 |

## Descrição do problema

A avaliação tradicional da qualidade de um vinho depende de análise sensorial,
que pode ser subjetiva e exigir profissionais especializados. Este projeto
investiga se propriedades químicas mensuráveis podem auxiliar na identificação
de vinhos tintos considerados bons.

A nota original do dataset varia de 0 a 10. Para este projeto, o problema foi
transformado em uma classificação binária:

- **Classe 1 - Vinho bom:** nota maior ou igual a 7.
- **Classe 0 - Vinho ruim/médio:** nota menor que 7.

## Objetivo

Desenvolver, comparar e avaliar modelos de classificação capazes de prever a
classe de qualidade de um vinho tinto. O projeto também tem como objetivo
disponibilizar o modelo final em uma aplicação Streamlit simples, organizada e
capaz de realizar previsões a partir de dados informados pelo usuário.

## Dataset utilizado

Foi utilizada a versão de vinhos tintos do dataset **Wine Quality**, disponível
no UCI Machine Learning Repository.

- Arquivo: `winequality-red.csv`
- Registros originais: 1.599
- Variáveis de entrada: 11
- Variável original de saída: `quality`
- Valores ausentes: 0
- Linhas duplicadas removidas: 240
- Registros utilizados após a limpeza: 1.359
- Vinhos bons após a limpeza: 184, correspondendo a 13,54% dos dados

As variáveis de entrada são:

| Variável | Descrição |
|---|---|
| `fixed acidity` | Acidez fixa |
| `volatile acidity` | Acidez volátil |
| `citric acid` | Ácido cítrico |
| `residual sugar` | Açúcar residual |
| `chlorides` | Cloretos |
| `free sulfur dioxide` | Dióxido de enxofre livre |
| `total sulfur dioxide` | Dióxido de enxofre total |
| `density` | Densidade |
| `pH` | Potencial hidrogeniônico |
| `sulphates` | Sulfatos |
| `alcohol` | Teor alcoólico |

Fonte oficial: [UCI Machine Learning Repository - Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

Referência: Cortez, P., Cerdeira, A., Almeida, F., Matos, T. e Reis, J.
(2009). *Wine Quality*. DOI:
[10.24432/C56S3T](https://doi.org/10.24432/C56S3T).

## Tipo de problema de Machine Learning

O projeto utiliza **aprendizado supervisionado para classificação binária**.
Os modelos aprendem a relação entre as 11 propriedades físico-químicas e a
classe-alvo derivada da nota de qualidade.

Existe desbalanceamento entre as classes, pois apenas 13,54% dos registros
tratados pertencem à classe de vinhos bons. Por esse motivo, a seleção não foi
baseada somente em acurácia.

## Metodologia

1. Carregamento e inspeção inicial com `head()`, `info()` e `describe()`.
2. Verificação de tipos, valores ausentes, valores únicos e duplicatas.
3. Remoção de 240 linhas exatamente duplicadas.
4. Criação da classe-alvo com `quality >= 7`.
5. Análise exploratória com distribuições, boxplots e mapa de correlação.
6. Identificação de possíveis outliers pelo método IQR.
7. Preservação dos outliers, pois eles podem representar medições válidas.
8. Divisão estratificada em 60% treino, 20% validação e 20% teste.
9. Uso de `Pipeline` para aplicar `StandardScaler` sem vazamento de dados.
10. Validação cruzada com `StratifiedKFold`, cinco folds e `random_state=42`.
11. Comparação dos modelos no conjunto de validação.
12. Escolha pelo maior F1-score, com ROC-AUC e recall como desempate.
13. Treinamento final com a união dos conjuntos de treino e validação.
14. Avaliação única no conjunto de teste.
15. Salvamento do pipeline, classificador, scaler e metadados.

O conjunto de teste não foi utilizado para escolher o modelo. Ele permaneceu
separado até a avaliação final, reduzindo o risco de resultados otimistas.

## Modelos treinados

Foram avaliados três classificadores:

| Modelo | Configuração principal |
|---|---|
| K-Nearest Neighbors | `n_neighbors=5` e `weights="distance"` |
| Support Vector Classifier | `class_weight="balanced"` e probabilidades habilitadas |
| Random Forest | 300 árvores e `class_weight="balanced"` |

Todos os modelos foram inseridos em um `Pipeline` com `StandardScaler`. Embora
árvores não dependam de padronização, a mesma estrutura foi mantida para
uniformizar a comparação e o uso dos artefatos.

## Métricas de avaliação

Foram utilizadas:

- **Acurácia:** proporção total de classificações corretas.
- **Precisão:** proporção de acertos entre os vinhos previstos como bons.
- **Recall:** proporção de vinhos realmente bons que foram identificados.
- **F1-score:** equilíbrio entre precisão e recall.
- **ROC-AUC:** capacidade de separação entre as classes em diferentes limiares.
- **Matriz de confusão:** quantidade de acertos e erros por classe.

### Validação cruzada no conjunto de treino

| Modelo | Acurácia | Precisão | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVC | 0,815 | 0,401 | 0,745 | 0,521 | 0,870 |
| Random Forest | 0,861 | 0,489 | 0,491 | 0,488 | 0,868 |
| KNN | 0,863 | 0,506 | 0,336 | 0,400 | 0,792 |

### Comparação no conjunto de validação

| Modelo | Acurácia | Precisão | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVC | 0,816 | 0,411 | 0,811 | 0,545 | 0,876 |
| Random Forest | 0,860 | 0,486 | 0,486 | 0,486 | 0,884 |
| KNN | 0,868 | 0,519 | 0,378 | 0,438 | 0,820 |

## Modelo final escolhido

O modelo final escolhido foi o **Support Vector Classifier (SVC)**.

O SVC obteve o maior F1-score no conjunto de validação (`0,545`) e também
apresentou o maior recall (`0,811`). Esse comportamento é importante porque a
classe positiva é minoritária e o projeto busca identificar vinhos bons sem
depender apenas da acurácia geral.

## Principais resultados

Após a escolha, o SVC foi treinado novamente com a união de treino e validação
e avaliado no conjunto de teste, composto por 272 registros inéditos.

| Métrica | Resultado no teste |
|---|---:|
| Acurácia | 0,805 |
| Precisão | 0,392 |
| Recall | 0,784 |
| F1-score | 0,523 |
| ROC-AUC | 0,884 |

Na matriz de confusão final, o modelo identificou corretamente 29 dos 37 vinhos
bons presentes no teste. O recall de `0,784` indica que aproximadamente 78,4%
dos vinhos realmente bons foram encontrados.

A precisão de `0,392` mostra que o modelo também produz falsos positivos. Essa
é uma consequência do equilíbrio adotado para aumentar a identificação da
classe minoritária e deve ser considerada na interpretação das previsões.

## Aplicação Streamlit

A aplicação permite informar as 11 propriedades químicas, executar a previsão,
visualizar a classe estimada e consultar a probabilidade calculada para um vinho
ser considerado bom.

**Link do app publicado:**  
https://classificador-vinhos-unimar.streamlit.app/

## Estrutura dos arquivos

```text
Projeto_P2_Machine_Learning/
├── app.py
├── README.md
├── requirements.txt
├── model/
│   ├── metadata_modelo.json
│   ├── modelo_vinhos_final.pkl
│   ├── pipeline_vinhos_final.pkl
│   └── scaler_vinhos.pkl
├── notebooks/
│   ├── DESAFIO_09_Qualidade_de_Vinhos.ipynb
│   └── winequality-red.csv
└── reports/
    └── Relatório_Machine_Learning_P2.pdf
```

## Tecnologias utilizadas

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

As versões utilizadas para gerar os artefatos estão fixadas em
`requirements.txt` e registradas em `model/metadata_modelo.json`.

## Como executar o notebook

### 1. Clonar o repositório

```bash
git clone https://github.com/joseHenrique346/Projeto_P2_Machine_Learning.git
cd Projeto_P2_Machine_Learning
```

Caso o projeto já esteja no computador, basta abrir o terminal na pasta raiz.

### 2. Criar e ativar um ambiente virtual

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
python -m pip install -r requirements.txt
python -m pip install notebook
```

### 4. Abrir o notebook

```bash
python -m notebook
```

Abra `notebooks/DESAFIO_09_Qualidade_de_Vinhos.ipynb` e utilize a opção
**Run All** para executar todas as células na ordem.

Executar novamente o notebook substitui os arquivos da pasta `model/` pelos
artefatos correspondentes ao novo treinamento.

## Como executar o app Streamlit

Com o ambiente virtual ativado e as dependências instaladas:

```bash
python -m streamlit run app.py
```

A aplicação ficará disponível normalmente em:

```text
http://localhost:8501
```

## Limitações

- A classe de vinhos bons representa apenas 13,54% dos registros tratados.
- A transformação em duas classes simplifica diferenças entre notas distintas.
- O dataset representa apenas vinhos tintos Vinho Verde de uma origem específica.
- As variáveis disponíveis são físico-químicas e não incluem marca, preço, tipo
  de uva ou outras informações comerciais.
- A qualidade sensorial não depende somente das propriedades utilizadas.
- A precisão da classe positiva é inferior ao recall, indicando falsos positivos.
- O modelo auxilia a análise, mas não substitui avaliação sensorial especializada.

## Conclusão

O projeto demonstrou que propriedades físico-químicas podem ser utilizadas para
auxiliar a classificação da qualidade de vinhos tintos. Entre os modelos
avaliados, o SVC apresentou o melhor equilíbrio para a classe minoritária,
alcançando recall de `0,784`, F1-score de `0,523` e ROC-AUC de `0,884` no teste.

A metodologia final utiliza validação estratificada, pipelines, tratamento do
desbalanceamento e um conjunto de teste reservado. O modelo foi integrado a uma
aplicação Streamlit funcional, permitindo a realização de previsões de maneira
simples e reproduzível.
