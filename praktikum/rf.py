from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def train(train_data, test_data, n_estimators=100, max_depth=3):
    X_train = np.array([x for x, _ in train_data])
    y_train = np.array([np.argmax(y) for _, y in train_data])
    X_test = np.array([x for x, _ in test_data])
    y_test = np.array([np.argmax(y) for _, y in test_data])

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    # test
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)
    return test_mse, test_acc * 100
