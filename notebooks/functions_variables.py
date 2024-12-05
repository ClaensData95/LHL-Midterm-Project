import json
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def encode_tags(df, top_n=10):

    # Count the frequency of each tag
    counts = df['tags'].explode().value_counts()
    # Keep top_n tags
    keep = counts.index[:top_n].tolist()
    # Filter tags column
    df['tags'] = df['tags'].apply(
        lambda x: [t for t in x if t in keep] if isinstance(x, list) else x
    )
    exploded_df = df['tags'].explode('tags')
    dummies = pd.get_dummies(exploded_df, columns=['tags'], prefix='tag')
    df = pd.concat([dummies, df], axis=1)
    return df


def json_to_df(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Convert JSON data to a DataFrame
    return pd.json_normalize(data['data']['results'], sep='_')


def threshold_column(df, column, threshold=1):
    # default frequency threshold = 1
    if type(df) == pd.Series: df = df.to_frame()
    # print(df.shape, column, threshold)
    # Count the frequency of each tag
    counts = df[column].explode().value_counts()
    # Filter out tags below the threshold
    keep = counts[counts >= threshold].index
    # Return the filtered column
    return df[column].apply(
        lambda x: [t for t in x if t in keep] if isinstance(x, list) else x
    )


def ohe_column(df, column):
    # Explode the column to separate rows for each item
    exploded_df = df.explode(column)

    # One-hot encode the exploded items
    return pd.get_dummies(exploded_df, columns=[column])


def reorder_columns_train_test(train_df, test_df):
  
    # Define the desired column order
    column_order = [
        'year_listed', 'year_sold', 'days_on_market', 'city_frequency', 'state_frequency',
        'description_year_built', 'description_stories', 'description_baths',
        'description_beds', 'description_garage', 'description_sqft', 'central_air',
        'community_outdoor_space', 'basement', 'fireplace', 'hardwood_floors',
        'recreation_facilities', 'community_security_features', 'view', 'central_heat',
        'city_view', 'fenced_yard'
    ]
    
    # Reorder the DataFrame columns for both train and test sets
    train_df = train_df.reindex(columns=column_order)
    test_df = test_df.reindex(columns=column_order)
    
    return train_df, test_df

def year_sold_listed(df):
    df['year_sold'] = df['description_sold_date'].dt.year
    df['year_listed'] = df['list_date'].dt.year
    return df


def fill_na_with_distribution_train_test(train_df, test_df, column):
  
    # Get non-NaN values from the training set
    non_nan_values = train_df[column].dropna()

    # Fill NaN values in the training set by sampling from its own distribution
    train_fill_values = np.random.choice(non_nan_values, size=train_df[column].isna().sum(), replace=True)
    train_df.loc[train_df[column].isna(), column] = train_fill_values

    # Fill NaN values in the testing set using the training distribution
    test_fill_values = np.random.choice(non_nan_values, size=test_df[column].isna().sum(), replace=True)
    test_df.loc[test_df[column].isna(), column] = test_fill_values

    return train_df, test_df

def fill_na_with_median_train_test(train_df, test_df, column):

    # Calculate the median from the training set
    median_value = train_df[column].median()

    # Fill NaN values in the column for both training and testing sets
    train_df[column] = train_df[column].fillna(median_value)
    test_df[column] = test_df[column].fillna(median_value)

    return train_df, test_df


def scale_columns_train_test(train_df, test_df, columns_to_scale):
    
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the specified columns
    train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
    test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])

    return train_df, test_df, scaler
    


#def scale_target_train_test(y_train, y_test):

    # Convert y_train and y_test to DataFrame if they are Series
 #   if isinstance(y_train, pd.Series):
  #      y_train = y_train.to_frame(name='target')
   # if isinstance(y_test, pd.Series):
    #    y_test = y_test.to_frame(name='target')

    # Check for empty datasets
    #if y_train.empty or y_test.empty:
 #      raise ValueError("y_train or y_test is empty. Check your data splitting or preprocessing steps.")

  #  # Initialize the scaler
   # scaler = StandardScaler()

    # Fit the scaler on y_train and transform both y_train and y_test
    #y_train_scaled = pd.DataFrame(scaler.fit_transform(y_train), columns=['target'], index=y_train.index)
    #y_test_scaled = pd.DataFrame(scaler.transform(y_test), columns=['target'], index=y_test.index)

    #return y_train_scaled, y_test_scaled, scaler






