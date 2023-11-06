from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import pickle
import pandas as pd
import numpy as np

### Data preparation

df = pd.read_csv('dataset.csv')

categorical = list(df.dtypes[df.dtypes == 'object'].index)

for col in categorical:
  df[col] = df[col].str.lower().str.replace(' ', '_')

df.head()

categorical = [
	'airline',
	'flight',
	'source_city',
	'departure_time',
	'stops',
	'arrival_time',
	'destination_city',
	'class'
]

numerical = [
	'duration',
	'days_left'
]

### Split the dataset

df['price'] = np.log1p(df['price'])

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.price.values
y_test = df_test.price.values

del df_full_train['price']
del df_test['price']

full_train_dict = df_full_train[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(full_train_dict)

X_full_train = dv.transform(full_train_dict)

test_dict = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(test_dict)

### Train Model

rf = RandomForestRegressor(random_state=42,
                          n_jobs=-1,
                          n_estimators=80,
                          min_samples_leaf=1,
                          max_depth=60)

rf.fit(X_full_train, y_full_train)
y_pred = rf.predict(X_test)

score =  mean_squared_error(y_test, y_pred, squared=False)
print('score ', round(score, 3))

### Save the model

with open('model_rf.bin', 'wb') as f_out:
  pickle.dump((rf), f_out)

with open('dv.bin', 'wb') as f_out:
  pickle.dump((dv), f_out)

