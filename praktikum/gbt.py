from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def train(train_data, test_data, lr=1.0, n_estimators=100, max_depth=3):
    X_train = np.array([x for x, _ in train_data])
    y_train = np.array([np.argmax(y) for _, y in train_data])
    X_test = np.array([x for x, _ in test_data])
    y_test = np.array([np.argmax(y) for _, y in test_data])

    model = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth, random_state=0
    )
    model.fit(X_train, y_train)
    # test
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)
    return test_mse, test_acc * 100
