# Projeto Passos Mágicos - Previsão de Risco de Defasagem


Analisar os indicadores educacionais da base PEDE e construir um modelo preditivo para estimar a probabilidade de um aluno entrar em risco futuro de defasagem, queda de desempenho, queda de engajamento ou evasão.


- `data/raw`: base original
- `data/silver`: base tratada
- `data/gold`: bases analíticas e de modelagem
- `notebooks`: exploração, feature engineering, storytelling e modelagem
- `models`: modelo treinado
- `app`: aplicação Streamlit
- `reports`: apresentação final

Foi utilizado um modelo Random Forest com tratamento de valores ausentes por mediana. O modelo final apresentou acurácia de 76% e AUC de 0,73.

Instalar dependências:

```bash
pip install -r requirements.txt