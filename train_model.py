import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar e Limpar
df = pd.read_csv('telco_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df = df.drop('customerID', axis=1)

# 2. Pré-processamento
# Converter variáveis categóricas em números usando get_dummies (One-Hot Encoding)
# Isso é mais seguro que LabelEncoder para evitar erros de tipo e capturar relações
df_encoded = pd.get_dummies(df, drop_first=True)

# Garantir que todos os dados sejam numéricos (bool -> int)
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'bool':
        df_encoded[col] = df_encoded[col].astype(int)

# Separar Features (X) e Target (y)
# Após o get_dummies, o Churn virou Churn_Yes
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

# Dividir em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Treinar o Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Avaliar
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Acurácia do Modelo: {accuracy:.2%}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 5. Importância das Variáveis
importances = model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title('Top 15 Variáveis mais importantes para prever o Churn')
plt.tight_layout()
plt.savefig('feature_importance.png')

# 6. Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print("Modelo treinado e avaliado com sucesso.")
