from utils.data_preparation import prepare_data, split_data
from models.logistic_regression import gradient_descent, predict
from models.evaluation import evaluate_model
import numpy as np

def main():
    """
    Funcion principal que ejecuta la preparacion de datos y de los codigos implementados, 
    el entrenamiento y la evaluacion del modelo de regresion logistica.
    
    """
    # Preparacion de los datos, aqui se carga el archivo y se mandan los train, validation y test
    X, y, means, stds = prepare_data('data/diabetes.csv')
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)
    
    # se inicializan los parámetros del modelo
    params = np.zeros(X_train.shape[1])
    
    # aqui son los parametros del modelo, aqui practicamente fui jugando con los valores hasta terminar 
    # en cual es mejor para el modelo
    learning_rate = 0.02
    epochs = 1000
    reg_strength = 0.1
    
    # aqui ya se empezara a entrenar el modelo
    params = gradient_descent(X_train, y_train, params, learning_rate, epochs, reg_strength)
    
    # aqui el modelo es evaluado para la validacion del mismo
    print("Evaluación en el conjunto de validación:")
    threshold = evaluate_model(X_validation, y_validation, params)
    
    # y por ultimo se imprime lo que es la evaluacion de prueba 
    print("\nEvaluación en el conjunto de prueba:")
    evaluate_model(X_test, y_test, params)
    
    # Predicción para un nuevo paciente, esto para comprobar que tan bueno 
    # es el modelo, con el entrenamiento brindado
    print("\nIntroduce los valores del nuevo paciente para hacer una predicción:")
    pregnancies = float(input("Pregnancies: "))
    glucose = float(input("Glucose: "))
    blood_pressure = float(input("BloodPressure: "))
    skin_thickness = float(input("SkinThickness: "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    dpf = float(input("DiabetesPedigreeFunction: "))
    age = float(input("Age: "))
    
    # Creacion del array de caracteristicas del nuevo paciente
    new_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Normalizar los datos del nuevo paciente utilizando las medias y desviaciones estándar calculadas
    new_data = (new_data - means) / stds
    new_data = np.c_[np.ones(new_data.shape[0]), new_data]
    
    # Prediccion del modelo
    prediction = predict(new_data, params) >= threshold
    print("\nPredicción sobre el nuevo paciente:")
    if prediction[0]:
        print(f"El paciente probablemente tiene diabetes (Predicción: True) con un umbral de {threshold:.4f}.")
    else:
        print(f"El paciente probablemente NO tiene diabetes (Predicción: False) con un umbral de {threshold:.4f}.")

if __name__ == "__main__":
    main()
