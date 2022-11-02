import catboost
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

def create_pool(x, y, cat_features, text_features):
    return catboost.Pool(
        data=x, label=y, cat_features=cat_features, text_features=text_features
    )


def grid_search(params, dataset, f):
    res = pd.DataFrame(columns=['iterations', 'lr', 'depth', 'rmse', 'mae'])
    ind = 0
    for i in params['iterations'].split(' - '):
        for lr in params['lr'].split(' - '):
            for d in params['depth'].split(' - '):
                p = {
                    'iterations': int(i),
                    'learning_rate': float(lr),
                    'depth': int(d) 
                }
                print(ind)
                rmse, mae = cross_validation(dataset, f, p)
                res.loc[ind] = [i, lr, d, rmse, mae]
                ind += 1
    return res

    

def cross_validation(dataset, f, params):
    n = len(dataset)
    mse_score = 0
    mae_score = 0
    for train_x, test_x, train_y, test_y in dataset:
        train = create_pool(train_x, train_y, f['cat'].split(' - '), f['text'].split(' - '))
        test = create_pool(test_x, test_y, f['cat'].split(' - '), f['text'].split(' - '))
        model = catboost.CatBoostRegressor(learning_rate=params['learning_rate'], depth=params['depth'], iterations=params['iterations'], loss_function='RMSE')
        model.fit(train)
        pred = model.predict(test)
        mse_score += mean_squared_error(test_y, pred, squared=False)
        mae_score += mean_absolute_error(test_y, pred)
    mse_score /= n
    mae_score /= n
    return mse_score, mae_score

