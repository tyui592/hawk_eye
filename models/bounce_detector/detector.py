import catboost as ctb


def get_bounce_detector(args):
    return ctb.CatBoostRegressor(loss_function='RMSE')