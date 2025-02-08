import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_features(path_dataset, num_frames):
    games = os.listdir(path_dataset)
    df = pd.DataFrame()
    for game in games:
        if 'game' not in game:
            continue          
          
        clips = os.listdir(os.path.join(path_dataset, game))
        for clip in clips:
            if 'Clip' not in clip:
                continue

            labels = pd.read_csv(os.path.join(path_dataset, game, clip, 'Label.csv'))

            eps = 1e-15
            for i in range(1, num_frames):
                labels['x_lag_{}'.format(i)] = labels['x-coordinate'].shift(i)
                labels['x_lag_inv_{}'.format(i)] = labels['x-coordinate'].shift(-i)
                labels['y_lag_{}'.format(i)] = labels['y-coordinate'].shift(i)
                labels['y_lag_inv_{}'.format(i)] = labels['y-coordinate'].shift(-i) 
                labels['x_diff_{}'.format(i)] = abs(labels['x_lag_{}'.format(i)] - labels['x-coordinate'])
                labels['y_diff_{}'.format(i)] = labels['y_lag_{}'.format(i)] - labels['y-coordinate']
                labels['x_diff_inv_{}'.format(i)] = abs(labels['x_lag_inv_{}'.format(i)] - labels['x-coordinate'])
                labels['y_diff_inv_{}'.format(i)] = labels['y_lag_inv_{}'.format(i)] - labels['y-coordinate']
                labels['x_div_{}'.format(i)] = abs(labels['x_diff_{}'.format(i)]/(labels['x_diff_inv_{}'.format(i)] + eps))
                labels['y_div_{}'.format(i)] = labels['y_diff_{}'.format(i)]/(labels['y_diff_inv_{}'.format(i)] + eps)

            labels['target'] = (labels['status'] == 2).astype(int)         
            for i in range(1, num_frames):    
                labels = labels[labels['x_lag_{}'.format(i)].notna()]
                labels = labels[labels['x_lag_inv_{}'.format(i)].notna()]
            labels = labels[labels['x-coordinate'].notna()]  

            labels['status'] = labels['status'].astype(int)
            # return df, labels
            df = pd.concat([df, labels], axis=0, ignore_index=True)
    return df


def create_train_test(df, num_frames):
    colnames_x = ['x_diff_{}'.format(i) for i in range(1, num_frames)] + \
                 ['x_diff_inv_{}'.format(i) for i in range(1, num_frames)] + \
                 ['x_div_{}'.format(i) for i in range(1, num_frames)]
    colnames_y = ['y_diff_{}'.format(i) for i in range(1, num_frames)] + \
                 ['y_diff_inv_{}'.format(i) for i in range(1, num_frames)] + \
                 ['y_div_{}'.format(i) for i in range(1, num_frames)]
    colnames = colnames_x + colnames_y 
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=5)
    X_train = df_train[colnames]
    X_test = df_test[colnames]
    y_train = df_train['target']
    y_test = df_test['target']
    return X_train, y_train, X_test, y_test


def get_bounce_datasets(args):
    df_features = create_features(args.data_path, args.num_seq)

    X_train, y_train, X_test, y_test = create_train_test(df_features, args.num_seq)
    
    return (X_train, y_train), (X_test, y_test)