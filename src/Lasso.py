import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

####LASSO
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

####Scaler
def scale(sample, len_train):
    scaler_rt = MinMaxScaler(feature_range=(-1, 1))
    scaler_rt = scaler_rt.fit(sample[0:len_train, 0].reshape(len_train, 1))
    rt_scaled = scaler_rt.transform(sample[:, 0].reshape(len(sample), 1))

    if sample.shape[1] > 1:
        scaler_oth = MinMaxScaler(feature_range=(-1, 1))
        scaler_oth = scaler_oth.fit(sample[0:len_train, 1:])
        oth_scaled = scaler_oth.transform(sample[:, 1:])
        result = np.hstack([rt_scaled, oth_scaled]).reshape(np.shape(sample))
    else:
        result = rt_scaled
    return scaler_rt, result


####RMSE
def RMSE(a, b):
    return np.sqrt(mean_squared_error(a, b))


####Rolling window
def rolling_window2D(a, n):
    # a: 2D Input array
    # n: Group/sliding window length
    return a[np.arange(a.shape[0] - n + 1)[:, None] + np.arange(n)]


###Cross validation
def FV1():
    for i in range(21, 126, 21):
        idx = np.array(range(0, i))
        idy = np.array(range(i, i + 21))
        yield idx, idy


# import data
data = pd.read_csv("../data/processed/SBER.csv", sep=";", index_col=0)[0:563]
data.index = pd.to_datetime(data.index)
data = data.replace(np.inf, np.nan)
data["LIX"][data["LIX"].isnull()] = max(data["LIX"])
data["F_LIX"][data["F_LIX"].isnull()] = max(data["F_LIX"])
data[data.isnull()] = 0

dataset_1 = data["Log_returns"].pipe(np.array).reshape(len(data), 1)
dataset_2 = data[["Log_returns", "Vol", "Amihud", "LHH", "LIX"]].pipe(np.array)
dataset_3 = data[["Log_returns", "Vol", "Amihud", "LHH", "LIX", "VAR", "VAR_21"]].pipe(
    np.array
)

dataset_1F = data[["Log_returns", "F_Log_returns"]].pipe(np.array)
dataset_2F = data[
    [
        "Log_returns",
        "F_Log_returns",
        "Vol",
        "Amihud",
        "LHH",
        "LIX",
        "F_Vol",
        "F_Amihud",
        "F_LHH",
        "F_LIX",
        "F_Open_pos",
        "F_Open_con",
    ]
].pipe(np.array)
dataset_3F = data[
    [
        "Log_returns",
        "F_Log_returns",
        "Vol",
        "Amihud",
        "LHH",
        "LIX",
        "F_Vol",
        "F_Amihud",
        "F_LHH",
        "F_LIX",
        "F_Open_pos",
        "F_Open_con",
        "VAR",
        "F_VAR",
        "VAR_21",
        "F_VAR_21",
    ]
].pipe(np.array)

all_datasets = [dataset_1, dataset_2, dataset_3, dataset_1F, dataset_2F, dataset_3F]

real = dataset_1[145:].flatten()

####LASSO
len_test = 1
LASSO_preds = []
for DS in all_datasets:
    alphas = []
    pred_LASSO = np.array([])
    for i in range(145, len(DS), len_test):
        scaler, data_scaled = scale(DS[i - 131 : i + len_test], 131)
        samples = rolling_window2D(data_scaled, 6)
        X, y = samples[:, 0:-1], samples[:, -1, 0]
        X, y = X.reshape(X.shape[0], X.shape[1] * X.shape[2]), y.reshape(y.shape[0])
        train_X, train_Y = X[:126], y[:126]
        test_X, test_Y = X[126:], y[126:]
        clf = linear_model.LassoLarsCV(
            max_iter=1000,
            cv=FV1(),
            verbose=False,
            normalize=False,
            fit_intercept=True,
            n_jobs=-1,
        ).fit(train_X, train_Y)
        alphas.append(clf.alpha_)
        result = scaler.inverse_transform([clf.predict(test_X)]).flatten()
        pred_LASSO = np.append(pred_LASSO, result)
        print(i, DS.shape)
    LASSO_preds.append(pred_LASSO)

# np.savetxt('../data/predictions/LASSO_preds.txt', LASSO_preds)
# LASSO_preds = np.loadtxt('LASSO_preds.txt', dtype=float)

# Full_sample_err0 = [RMSE(real, LASSO_preds[i]) for i in range(6)]
# plt.plot(Full_sample_err0)

# Pre_cov_err0 = [RMSE(real[:126], LASSO_preds[i][:126]) for i in range(6)]
# plt.plot(Pre_cov_err0)
