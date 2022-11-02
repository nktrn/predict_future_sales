import uuid
from itertools import product

import catboost
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
    for i, parameters in enumerate(params_combination):
        parameters.update(model_configuration)
        run[f'global/{i}/parameters'] = parameters
        
        model = catboost.CatBoostRegressor(**parameters)
        categorical_features = config['features']['categorical']
        cross_validation(dataset, model, categorical_features, run, save_pth, i)


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

    name = save_pth + str(uuid.uuid4())
    model.save_model(name)
    run[f'global/{i}/model_name'] = name
