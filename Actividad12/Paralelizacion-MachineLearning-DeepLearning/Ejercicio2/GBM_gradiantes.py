import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generacion de un dataset de ejemplo
# make_classification crea un conjunto de datos sinteticos para un problema de clasificacion binaria
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2)
# Dividimos el conjunto de datos en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creamos el dataset de LightGBM
# Convertimos los datos de entrenamiento y prueba en formatos que LightGBM puede utilizar
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Configuramos el modelo
# Definimos los parametros del modelo LightGBM
params = {
    'objective': 'binary',         # Tipo de problema: clasificacion binaria
    'metric': 'binary_logloss',    # Metrica de evaluacion: binary log loss
    'boosting_type': 'gbdt',       # Tipo de boosting: Gradient Boosting Decision Tree
    'num_leaves': 31,              # Numero maximo de hojas en cada arbol
    'learning_rate': 0.05,         # Tasa de aprendizaje
    'feature_fraction': 0.9,       # Fraccion de caracteristicas a utilizar en cada iteracion
    'n_jobs': -1                   # Usar multiples nucleos para la paralelizacion
}

# Entrenamiento del modelo
# Entrenamos el modelo con los datos de entrenamiento y validamos con los datos de prueba
bst = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

