from sklearn.datasets import load_iris # Para cargar el conjunto de datos Iris
from sklearn.model_selection import train_test_split # Para dividir el conjunto de datos en entrenamiento y prueba
from sklearn.ensemble import RandomForestClassifier # Para crear el modelo de Random Forest
from sklearn.metrics import accuracy_score # Calcula la precisión del modelo
from joblib import Parallel, delayed # Paralelizar la evaluación de los modelos

# Funcion para evaluar un modelo de RandomForestClassifier con un numero especifico de estimadores
# n_estimators: Numero de arboles en el bosque
# X_train: Conjunto de datos de entrenamiento
# X_test: Conjunto de datos de prueba.
# y_train: Etiquetas del conjunto de entrenamiento
# y_test: Etiquetas del conjunto de pruebas
def evaluate_model(n_estimators, X_train, X_test, y_train, y_test):
    # Creamos y entrenamos el modelo
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    # Predecimos las etiquetas para el conjunto de prueba
    y_pred = model.predict(X_test)
    # Calculamos la precision del modelo
    accuracy = accuracy_score(y_test, y_pred)
    return (n_estimators, accuracy) # Retornanomos el numero de estiadores y la precision del modelo

# Funcion pra realizar la evaluacion de varios modelos paralelos
def parallel_model_evaluation():
    # Cargamos el conjunto de datos Iris
    iris = load_iris()
    # Dividimos el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    # Lista de diferentes numeros de estimadores a evaluar
    n_estimators_list = [10, 50, 100, 200]
    # Evaluamos los modelos en paralelo
    results = Parallel(n_jobs=4)(delayed(evaluate_model)(n, X_train, X_test, y_train, y_test) for n in n_estimators_list)
    return results # Lista de tuplas con el numero de estimadores y la precision de cada modelo

# Ejecutamos la evaluacion de modelos en paralelo y obtenemos los resultados
results = parallel_model_evaluation()

# Imprimimos los resultados
print(results)
