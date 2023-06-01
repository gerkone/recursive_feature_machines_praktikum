from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def train(
    train_data,
    test_data,
    eval_metric="mlogloss",
    n_estimators=100,
    booster="gbtree",
):
    X_train = np.array([x for x, _ in train_data])
    y_train = np.array([y for _, y in train_data])
    X_test = np.array([x for x, _ in test_data])
    y_test = np.array([y for _, y in test_data])

    model = XGBClassifier(
        eval_metric=eval_metric,
        n_estimators=n_estimators,
        booster=booster,
    )
    model.fit(X_train, y_train, verbose=False)
    # test
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)
    return test_mse, test_acc * 100
