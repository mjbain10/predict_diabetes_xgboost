from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading data as numpy array
dataset = loadtxt('diabetes_data.csv', delimiter=',')

# split input and target featurtes
X = dataset[:, 0:8]
y = dataset[:, 8]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, y_train)

# print(model)

# make predictions
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(accuracy * 100.0)