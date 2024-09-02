import numpy as np

def sigmoid(z):
    """
    Esta funcion lo que hara es calcular la función sigmoidal, que mapea cualquier valor real en un rango 
    entre 0 y 1.
    
    la cual retorna:
    - La probabilidad entre 0 y 1.
    """
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_pred, params, reg_strength):
    """
    la funcion de perdida logaritmica (log loss) con regularizacion L2 para evitar el overfitting.
    
    Y aqui vuelve:
    - El valor de la función de perdida logaritmica regularizada.
    """
    epsilon = 1e-15  # aqui lo ajusto debido a que tenia problemas y evitar log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clipping de predicciones para evitar problemas numericos, debido a que tenia errores
    regularization = (reg_strength / 2) * np.sum(params[1:] ** 2)  # Aqui finaliza la regulacion 
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) + regularization

def predict(X, params):
    """
    se realizara lo que es la predicción utilizando la regresion logistica.
    
    esta funcion lo que devuelve:
    - Probabilidades de predicción.
    """
    return sigmoid(np.dot(X, params))

def gradient_descent(X, y, params, learning_rate, epochs, reg_strength):
    """
    Optimiza los parametros del modelo utilizando descenso de gradiente con regularizacion L2.
    
    El gradiente descendiente nos va a devolver:
    - params: que es la variable array numpy con los parámetros optimizados.
    """
    m = len(y)  # Numero de ejemplos a tomar
    for epoch in range(epochs):
        y_pred = predict(X, params)  # esto lo que hace es hacer las predicciones actuales
        error = y_pred - y  # y aqui el error es calculado
        gradient = (np.dot(X.T, error) / m) + (reg_strength * np.r_[0, params[1:]]) / m  # Gradiente con regularización
        params -= learning_rate * gradient  # finalemnte se actualizan los parametros
        
        if epoch % 100 == 0:
            loss = log_loss(y, y_pred, params, reg_strength)
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    return params
