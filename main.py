import datetime

import pandas as pd
import numpy as np
import xgboost
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

config = {
    'look_back': 10,
    'lead_time': 1,
    'feature_cols': ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)', 'QObs(mm/d)'],
    'target_cols': ['QObs(mm/d)']
}
df = pd.read_csv(r'data/02324400.csv')

df['date'] = pd.to_datetime(df['date'])
train_df = df.loc[(pd.to_datetime('19801001', format='%Y%m%d') <= df['date']) & (
        df['date'] <= pd.to_datetime('19950930', format='%Y%m%d')), :]
val_df = df.loc[(pd.to_datetime('19951001', format='%Y%m%d') - datetime.timedelta(days=config['look_back'])
                 <= df['date']) & (df['date'] <= pd.to_datetime('20000930', format='%Y%m%d')), :]
test_df = df.loc[(pd.to_datetime('20001001', format='%Y%m%d') - datetime.timedelta(days=config['look_back'])
                  <= df['date']) & (df['date'] <= pd.to_datetime('20100930', format='%Y%m%d')), :]


def slide_window(df, look_back, lead_time, feature_cols, target_cols):
    x_list = []
    for i in range(0, len(df) - look_back - lead_time + 1, 1):
        temp_x = df[feature_cols].values[i:i + look_back, :].reshape(1, -1)
        x_list.append(temp_x)
    x = np.concatenate(x_list, axis=0)
    y = df[target_cols].values[lead_time + look_back - 1:, :].reshape(-1, 1)
    return x, y


train_x, train_y = slide_window(train_df, config['look_back'], config['lead_time'], config['feature_cols'],
                                config['target_cols'])
val_x, val_y = slide_window(val_df, config['look_back'], config['lead_time'], config['feature_cols'],
                            config['target_cols'])
test_x, test_y = slide_window(test_df, config['look_back'], config['lead_time'], config['feature_cols'],
                              config['target_cols'])

model = xgboost.XGBRegressor(max_depth=15, learning_rate=0.1, n_estimators=200)
model.fit(train_x, train_y, eval_set=[(val_x, val_y)])
test_pred = model.predict(test_x)
print(f"test r2_score: {r2_score(test_y, test_pred)}")
plt.plot(test_y, label='real')
plt.plot(test_pred, label='pred')
plt.show()
