import numpy as np
import pandas as pd

def prepare_data(filepath):
    """
    aqui la idea es preparar los datos para el entrenamiento del modelo, incluyendo manejo de valores 
    faltantes y normalizacion.
    
    dentro de lo que se hace menciono una lista de lo que devuelve esta funcion:
    - X: array con las caracteristicas.
    - y: array con las etiquetas (0 o 1).
    - means: medias de las caracteristicas.
    - stds: desviaciones estándar de las caracteristicas.
    """
    df = pd.read_csv(filepath)
    
    # con eso lo que se hace es reemplazar valores de 0 con NaN y luego llena con la mediana de la columna
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_replace:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    X = df.drop('Outcome', axis=1).values  # Caracteristicas
    y = df['Outcome'].values  # Etiquetas
    
    # Normalizacion de las caracteristicas
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    X = (X - means) / stds
    
    # Añadir un término de sesgo
    X = np.c_[np.ones(X.shape[0]), X]
    
    return X, y, means, stds

def split_data(X, y, train_size=0.7, validation_size=0.15):
    """
    Se va a dividir los datos en conjuntos de entrenamiento, validación y prueba como se nos especifica 
    realizar.
    

    Con esto lo esperado a devolver es:
    - X_train, X_validation, X_test: caracteristicas para entrenamiento, validacion y prueba.
    - y_train, y_validation, y_test: etiquetas para entrenamiento, validacion y prueba.
    """
    m = len(y)
    indices = np.arange(m)
    np.random.shuffle(indices)
    
    train_end = int(train_size * m)
    validation_end = int(validation_size * m) + train_end
    
    X_train = X[indices[:train_end]]
    y_train = y[indices[:train_end]]
    
    X_validation = X[indices[train_end:validation_end]]
    y_validation = y[indices[train_end:validation_end]]
    
    X_test = X[indices[validation_end:]]
    y_test = y[indices[validation_end:]]
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test
