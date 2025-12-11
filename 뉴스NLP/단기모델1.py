net = pd.read_csv('../24.12_수급,거래 데이터.csv', dtype={'ticker': str})
momentum = pd.read_csv('../단기모멘텀.csv', dtype={'ticker': str})


df1 = net[['date', '기관_순매수_금액', '외국인_순매수_금액', '개인_순매수_금액', 'ticker']]
df2 = momentum[['date', 'ret_1d', 'ret_5d', 'ret_20d', '등락률', 'ticker']]
df = pd.merge(df1, df2, on=['date', 'ticker'], how='outer')

FEATURES = [
    'ret_5d',
    'ret_1d',
    '외국인_순매수_금액',
    '기관_순매수_금액',
    '개인_순매수_금액',
    '등락률'
]

TARGET = 'ret_20d'

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error


# 날짜 정렬 (매우 중요)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 필요한 컬럼만 + 결측 제거
df_model = df[['date'] + FEATURES + [TARGET]].dropna()

# 기간 분할
train_df = df_model[
    (df_model['date'] >= '2015-01-01') &
    (df_model['date'] <= '2019-12-31')
]

val_df = df_model[
    (df_model['date'] >= '2020-01-01') &
    (df_model['date'] <= '2021-12-31')
]

test_df = df_model[
    (df_model['date'] >= '2022-01-01') &
    (df_model['date'] <= '2024-12-31')
]

# X / y 분리
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val, y_val     = val_df[FEATURES], val_df[TARGET]
X_test, y_test   = test_df[FEATURES]

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

print(f"[Validation] RMSE: {val_rmse:.5f}")

X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

model.fit(X_train_full, y_train_full)

test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print(f"[TEST / OFFICIAL] RMSE: {test_rmse:.5f}")