import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# Carregar dados
df = pd.read_csv('telco_churn.csv')

# Limpeza básica: TotalCharges deve ser numérico
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# 1. Distribuição da variável alvo (Churn)
plt.figure()
sns.countplot(data=df, x='Churn', hue='Churn', palette='viridis', legend=False)
plt.title('Distribuição de Cancelamentos (Churn)')
plt.savefig('churn_distribution.png')

# 2. Churn por tipo de contrato
plt.figure()
sns.countplot(data=df, x='Contract', hue='Churn', palette='magma')
plt.title('Cancelamento por Tipo de Contrato')
plt.savefig('churn_by_contract.png')

# 3. Distribuição de Mensalidade vs Churn
plt.figure()
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, palette='crest')
plt.title('Distribuição de Mensalidade por Status de Churn')
plt.savefig('churn_by_monthly_charges.png')

# 4. Tempo de contrato (Tenure) vs Churn
plt.figure()
sns.boxplot(data=df, x='Churn', y='tenure', hue='Churn', palette='pastel', legend=False)
plt.title('Tempo de Contrato (Meses) vs Churn')
plt.savefig('churn_by_tenure.png')

print("Análise exploratória concluída. Gráficos salvos.")
