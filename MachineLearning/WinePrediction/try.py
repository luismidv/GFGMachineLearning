import pandas as pd

# Datos de entrenamiento
list = [0,2,3,4]
X_train = pd.DataFrame(list)
print(X_train)
num = X_train.isnull().any()
print("Nulls",num)


