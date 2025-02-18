import ConnectionDB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import skew, kurtosis, ttest_ind
from xgboost import XGBRegressor


class AnalysisMarketing:
    def __init__(self):
        """Inicializa a conexão com o banco de dados"""
        self.connection = ConnectionDB.ConnectionDB()
        self.connection.connect()
        self.connection.connect_sqlalchemy()

    def _dataframe_from_query(self, query):
        """Executa uma query e retorna um DataFrame sem valores nulos"""
        df = pd.read_sql_query(query, self.connection.engine)
        return df.dropna()

    def _describe_statistics(self, df):
        """Exibe estatísticas descritivas dos dados"""
        print(df.describe())

        for column in df.select_dtypes(include=['number']).columns:
            print(f"{column}:")
            print(f"   Assimetria: {skew(df[column]):.2f}")
            print(f"   Curtose: {kurtosis(df[column]):.2f}\n")

    def _heatmap(self, df):
        """Cria um heatmap de correlação entre as variáveis"""
        rename_columns = {
            "advertisement_spend": "Publicidade",
            "promotion_spend": "Promoção",
            "administration_spend": "Administração",
            "profit": "Lucro"
        }
        df = df.rename(columns=rename_columns)

        plt.figure(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlação entre Variáveis")
        plt.show()

    def _train_model(self, X_train, X_test, y_train, y_test):
        """Treina um modelo de regressão linear e retorna os resultados"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Exibir coeficientes do modelo
        print("Coeficientes do modelo:", model.coef_)
        print("Intercepto:", model.intercept_)

        # Calcular métricas de erro
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.2f}")

        return model, y_pred

    def _train_xgboost(self, X_train, X_test, y_train, y_test):
        """Treina um modelo XGBoost e retorna os resultados"""
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="r2")
        print(f"R² médio (XGBoost): {scores.mean():.2f}")

        return xgb_model, y_pred

    def _plot_predictions(self, y_test, y_pred):
        """Gera um gráfico comparando valores reais e previstos"""
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="royalblue", edgecolor="black")
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color="red", linewidth=2, label="Linha Ideal (y=x)")

        plt.xlabel("Lucro Real", fontsize=12)
        plt.ylabel("Lucro Previsto", fontsize=12)
        plt.title("Comparação entre Lucro Real e Lucro Previsto", fontsize=14, fontweight="bold")
        plt.legend()
        plt.show()

    def _profit_impact(self, model, X):
        """Gera um gráfico de impacto dos investimentos no lucro"""
        coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Impacto no Lucro'])
        print(coef_df)

        impactos = model.coef_
        variaveis = ["Publicidade", "Promoção", "Administração"]
        cores = ['green' if val > 0 else 'red' for val in impactos]

        plt.figure(figsize=(8, 5))
        plt.bar(variaveis, impactos, color=cores, alpha=0.7, edgecolor='black')

        for i, v in enumerate(impactos):
            plt.text(i, v + (0.05 if v > 0 else -0.08), f"{v:.3f}", ha='center', fontsize=12, fontweight='bold')

        plt.ylim(min(impactos) - 0.1, max(impactos) + 0.2)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.title("Impacto dos Investimentos no Lucro", fontsize=14, fontweight="bold")
        plt.xlabel("Tipo de Investimento", fontsize=12)
        plt.ylabel("Impacto no Lucro", fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.show()

    def _analyze_residuals(self, y_test, y_pred):
        """Analisa os resíduos do modelo"""
        residuos = y_test - y_pred

        # Histograma dos resíduos
        plt.figure(figsize=(8,5))
        sns.histplot(residuos, bins=20, kde=True)
        plt.title("Distribuição dos Resíduos")
        plt.xlabel("Erro (Lucro Real - Lucro Previsto)")
        plt.ylabel("Frequência")
        plt.show()

        # Q-Q plot para avaliar normalidade dos resíduos
        sm.qqplot(residuos, line="s")
        plt.title("Q-Q Plot dos Resíduos")
        plt.show()


# Criando objeto de análise e carregando os dados
analyse = AnalysisMarketing()
query = "SELECT advertisement_spend, promotion_spend, administration_spend, profit FROM \"Supermarket\".marketing_spend ORDER BY profit DESC"
df = analyse._dataframe_from_query(query)

# Exibir estatísticas descritivas dos dados
analyse._describe_statistics(df)

# Exibir o heatmap
analyse._heatmap(df)

# Definir variáveis X (features) e y (target)
X = df[['advertisement_spend', 'promotion_spend', 'administration_spend']]
y = df['profit']

# Criar nova variável ROI (Retorno sobre Investimento)
df['ROI'] = (df['profit'] - (df['advertisement_spend'] + df['promotion_spend'])) / (df['advertisement_spend'] + df['promotion_spend'])

# Dividir os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão linear
model, y_pred = analyse._train_model(X_train, X_test, y_train, y_test)

# Criar e treinar o modelo XGBoost
xgb_model, xgb_pred = analyse._train_xgboost(X_train, X_test, y_train, y_test)

# Analisar previsões
analyse._plot_predictions(y_test, y_pred)

# Analisar impacto
analyse._profit_impact(model, X)

# Analisar resíduos
analyse._analyze_residuals(y_test, y_pred)
