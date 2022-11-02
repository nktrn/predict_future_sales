from symbol import parameters
import uuid
from itertools import product

import catboost
from matplotlib.pyplot import get
import neptune.new as neptune

import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


def create_pool(x, y, cat_features):
    return catboost.Pool(
        data=x, label=y, cat_features=cat_features
    )


def grid_search(dataset, config):
    search_params = config['gs_params']
    params_combination = [dict(
            zip(tuple(search_params.keys()), (k))) 
            for k in product(*search_params.values()
    )]
    model_configuration = config['params']
    save_pth = config['model_save']
    run = neptune.init(
            project=config['neptune']['project'],
            api_token=config['neptune']['api_token']
    )
    
    res = []
    for i, parameters in enumerate(params_combination):
        parameters.update(model_configuration)
        run[f'global/{i}/parameters'] = parameters
        
        model = catboost.CatBoostRegressor(**parameters)
        categorical_features = config['features']['categorical']
        rmse, mae, model_name = cross_validation(dataset, model, categorical_features, run, save_pth, i)
        _ = {}
        _['parameters'] = parameters
        _['rmse'] = rmse
        _['mae'] = mae
        _['model_name'] = model_name
        res.append(_)
    run.stop()
    return res


def cross_validation(dataset, model, categorical_features, run, save_pth, i):
    mean_rmse, mean_mae = 0, 0
    n = len(dataset)
    for (x_train, x_test, y_train, y_test) in dataset:
        noise = np.random.normal(0, 0.1, len(y_train))
        y_train += noise
        train_pool = create_pool(x_train, y_train, categorical_features)
        test_pool = create_pool(x_test, y_test, categorical_features)

        model.fit(train_pool)
        forecast = model.predict(test_pool)

        fold_rmse = mean_squared_error(y_test, forecast, squared=False)
        fold_mae = mean_absolute_error(y_test, forecast)

        mean_rmse += fold_rmse
        mean_mae += fold_mae
        run[f"global/{i}/metrics/rmse"].log(fold_rmse)
        run[f"global/{i}/metrics/mae"].log(fold_mae)

    mean_rmse, mean_mae = mean_rmse/n, mean_mae/n

    run[f'global/metrics/mean_rmse'].log(mean_rmse)
    run[f'global/metrics/mean_mae'].log(mean_mae)

    name = f'{save_pth}{str(uuid.uuid4())}.cbm'
    model.save_model(name)
    run[f'global/{i}/model_name'] = name
    return mean_rmse, mean_mae, name


def track_best_model(res, config):
    model_version = neptune.init_model_version(
        model="PFS-CTR1",
        project=config['neptune']['project'],
        api_token=config['neptune']['api_token']
    )
    best_model = get_best_model(res, 'mae')
    model_version['info/parameters'] = best_model['parameters']
    model_version['info/rmse'] = best_model['rmse']
    model_version['info/mae'] = best_model['mae']
    model_version['info/name'] = best_model['model_name']
    model_version['model'].upload(best_model['model_name'])
    model_version.stop()


def get_best_model(res, type):
    m = 1000
    ret = None
    for ind, i in enumerate(res):
        if i[type] <= m:
            ret = i
            m = i[type]
    return ret
