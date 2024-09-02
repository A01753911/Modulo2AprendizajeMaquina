import numpy as np
import pandas as pd
from .logistic_regression import predict

def evaluate_model(X, y, params, threshold=0.5):
    """
    Con esto se va a evaluar un modelo de regresión logistica utilizando las métricas de precision, recall, 
    f1 score y una matriz de confusión.
    
    lo que se piensa devolver para esta seccion es:
    - threshold: el umbral utilizado para la clasificacion.
    """
    y_pred_prob = predict(X, params)  # aqui se predeciran los datos usando el modelo entrenado
    y_pred = y_pred_prob >= threshold  # todo eso se clasificaran con en el umbral que di
    
    # Caalculo de metricas de evaluacion, esto es para determinar la evaluacion del modelo una vez corrida la aplicacion
    accuracy = np.mean(y_pred == y)
    precision = np.sum((y_pred == 1) & (y == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1) if np.sum(y == 1) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Generacion de la matriz de confusion
    cm = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    # Imprime las metricas
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nMatriz de confusión:")
    print(cm)
    
    return threshold
