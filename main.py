import pandas as pd
import xgboost as xgb
import sklearn as sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

bat = pd.read_csv('Battery_RUL.csv')

# Remove "Cycle_Index" and "RUL" from X
X = bat.iloc[:, 1:-1]
y = bat["RUL"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4333)

xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=4333)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

best_xgb_regressor = grid_search.best_estimator_

y_pred = best_xgb_regressor.predict(X_test)

print("Best parameters found: ", grid_search.best_params_)
print("Mean Squared Error: ", metrics.mean_squared_error(y_test, y_pred))
print("R-squared: ", metrics.r2_score(y_test, y_pred))
