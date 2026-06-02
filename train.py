import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Configuration des clés d'accès pour le stockage d'artéfacts MinIO (S3 local)
os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

# 2. Connexion au serveur de tracking MLflow (Interface Graphique Port 5000)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Prediction_Maladies_Foie")

print("--- Initialisation du Dataset Indian Liver Patient ---")

# 3. Création/Simulation du Dataset Hépatique (583 patients, caractéristiques biochimiques)
np.random.seed(42)
n_patients = 583

data = {
    'Age': np.random.randint(7, 90, size=n_patients),
    'Gender': np.random.choice([0, 1], size=n_patients), # 0: Femme, 1: Homme
    'Total_Bilirubin': np.random.uniform(0.4, 40.0, size=n_patients),
    'Direct_Bilirubin': np.random.uniform(0.1, 19.0, size=n_patients),
    'Alkaline_Phosphotase': np.random.randint(63, 2110, size=n_patients),
    'Alamine_Aminotransferase': np.random.randint(10, 2000, size=n_patients),
    'Aspartate_Aminotransferase': np.random.randint(10, 4929, size=n_patients),
    'Total_Protiens': np.random.uniform(2.7, 9.6, size=n_patients),
    'Albumin': np.random.uniform(0.9, 5.5, size=n_patients),
    'Albumin_and_Globulin_Ratio': np.random.uniform(0.3, 2.8, size=n_patients)
}

# La cible : 1 pour malade du foie, 0 pour sain
# Logique clinique simplifiée : de hauts taux de Bilirubine et d'Enzymes augmentent le risque
risk_score = (data['Total_Bilirubin'] * 2) + (data['Alamine_Aminotransferase'] / 100) + (data['Age'] / 20)
data['Dataset_Target'] = np.where(risk_score > 15, 1, np.random.choice([0, 1], size=n_patients, p=[0.7, 0.3]))

df = pd.DataFrame(data)

# 4. Séparation des fonctionnalités (X) et de la cible (y)
X = df.drop(columns=['Dataset_Target'])
y = df['Dataset_Target']

# Découpage Train (80%) et Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset préparé avec succès : {X_train.shape[0]} patients d'entraînement et {X_test.shape[0]} patients de test.")

# 5. Lancement de l'entraînement et du suivi automatisé avec MLflow
with mlflow.start_run() as run:
    print("Entraînement de l'algorithme Random Forest...")
    
    # Configuration des hyperparamètres
    n_estimators = 150
    max_depth = 8
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation des performances cliniques
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions) # Crucial en médecine (ne rater aucun malade)
    f1 = f1_score(y_test, predictions)
    
    print(f"\n--- Résultats de l'évaluation ---")
    print(f"Précision Globale (Accuracy) : {accuracy * 100:.2f}%")
    print(f"Rappel (Recall)              : {recall * 100:.2f}%")
    print(f"Score F1                     : {f1 * 100:.2f}%")
    
    # 6. Envoi des métadonnées vers l'interface Web de MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # 7. Sauvegarde physique du modèle dans le bucket MinIO et enregistrement dans le registre
    mlflow.sklearn.log_model(model, "model_foie", registered_model_name="LiverDiseaseModel")
    
    print("\n[SUCCÈS] Toutes les métriques et le modèle hépatique ont été poussés dans l'infrastructure MLOps !")