import string

from sklearn import tree, model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


from sklearn.ensemble import RandomForestClassifier

import numpy as np

def run_dt(X, y, debug_flag):
    # 1. Split dataset to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # 2. scale values

    print("run_dt - X_train shape:")
    print(X_train.shape)
    print("run_dt - X_test shape:")
    print(X_test.shape)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Create a Gaussian Classifier
    # clf = RandomForestClassifier(n_estimators=100)
    clf = RandomForestClassifier(n_jobs=-1)

    if debug_flag:
        print("run_dt - param_grid - WITH DEBUG")
        param_grid = {
            "n_estimators": np.arange(100, 300, 100),
            "max_depth": np.arange(1, 3),
            "criterion": ["gini", "entropy"],
        }
    else:
        print("run_dt - param_grid - NO DEBUG")
        param_grid = {
            "n_estimators": np.arange(100, 1500, 100),
            "max_depth": np.arange(1, 20),
            "criterion": ["gini", "entropy"],
        }

    if debug_flag:
        print("run_dt - RandomizedSearchCV - WITH DEBUG")
        model = model_selection.RandomizedSearchCV(
            estimator=clf,
            param_distributions=param_grid,
            n_iter=3,
            scoring="accuracy",
            verbose=1,
            n_jobs=1,
            cv=3,
        )
    else:
        print("run_dt - RandomizedSearchCV - NO DEBUG")
        model = model_selection.RandomizedSearchCV(
            estimator=clf,
            param_distributions=param_grid,
            n_iter=7,
            scoring="accuracy",
            verbose=1,
            n_jobs=1,
            cv=4,
        )


    y_train = y_train.astype('int')
    model.fit(X_train, y_train.ravel())  # RandomForestClassifier (clf) encapsulated in RandomizedSearchCV

    y_pred = model.predict(X_test)
    label_dict = {0: "non-extinction", 1: "extinction"}
    y_test = y_test.astype('int').ravel()

    run_dt_res = classification_report(y_test, y_pred) # AI: add target names
    return run_dt_res

# def run_dt_test_only(X, y, target_names_in, model, sc):
def run_dt_test_only(X, y, model, sc):


    print("run_dt_test_only - START")

    X_test = sc.transform(X)

    y_pred = model.predict(X_test)


    y_pred_int = y_pred.astype('int')

    y_pred_ravel = y_pred_int.ravel()

    label_dict = {0: "non-extinction", 1: "extinction"}

    y_test_int = y.astype('int')


    y_test_ravel = y_test_int.ravel()

    run_dt_res = classification_report(y_test_ravel, y_pred_ravel)

    return run_dt_res

def  run_dt_train_only(X, y, debug_flag):
    print("run_dt_train_only - START")

    # 1. scale values
    sc = StandardScaler()
    X_train = sc.fit_transform(X)
    y_train = y

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_jobs=-1)
    if debug_flag:
        param_grid = {
            "n_estimators": np.arange(100, 300, 100),
            "max_depth": np.arange(1, 3),
            "criterion": ["gini", "entropy"],
        }
    else:
        param_grid = {
            "n_estimators": np.arange(100, 1500, 100),
            "max_depth": np.arange(1, 20),
            "criterion": ["gini", "entropy"],
        }

    if debug_flag:
        model = model_selection.RandomizedSearchCV(
            estimator=clf,
            param_distributions=param_grid,
            n_iter=3,
            scoring="accuracy",
            verbose=1,
            n_jobs=1,
            cv=3,
        )
    else:
        model = model_selection.RandomizedSearchCV(
            estimator=clf,
            param_distributions=param_grid,
            n_iter=4,
            scoring="accuracy",
            verbose=1,
            n_jobs=1,
            cv=4,
        )

    print(f"X_train.shape = {X_train.shape}")
    print(f"y_train.shape = {y_train.shape}")

    # Train the model using the training sets y_pred=clf.predict(X_test)
    print(f"y_train type {type(y_train[0])}")
    y_train = y_train.astype('int')

    model.fit(X_train, y_train.ravel())  # model = RandomForestClassifier (clf) encapsulated in RandomizedSearchCV

    print(f"model best_score:")
    print(model.best_score_)
    print(f"model best estimator get params:")
    print(model.best_estimator_.get_params())
    print(f"y_train.shape = {y_train.ravel().shape}")

    return model, sc
