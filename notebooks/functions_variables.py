import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def encode_tags(df, top_n=10):
    # Count the frequency of each tag
    counts = df['tags'].explode().value_counts()
    # Keep top_n tags
    keep = counts.index[:top_n].tolist()
    # Filter tags column
    df['tags'] = df['tags'].apply(
        lambda x: [t for t in x if t in keep] if isinstance(x, list) else x
    )
    # One-hot encode the tags
    exploded_df = df['tags'].explode('tags')
    # One-hot encode the exploded items
    dummies = pd.get_dummies(exploded_df, columns=['tags'], prefix='tag')
    # Concatenate the one-hot encoded tags
    df = pd.concat([dummies, df], axis=1)
    # Return the DataFrame
    return df


def json_to_df(file_path: str):
    # Load JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Convert JSON data to a DataFrame
    return pd.json_normalize(data['data']['results'], sep='_')


def threshold_column(df, column, threshold=1):
    # Convert Series to DataFrame
    if type(df) == pd.Series: df = df.to_frame()
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
    # Return the reordered DataFrames
    return train_df, test_df


def year_sold_listed(df):
    # Convert the date columns to datetime
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
    # Return the filled DataFrames
    return train_df, test_df


def fill_na_with_median_train_test(train_df, test_df, column):
    # Calculate the median from the training set
    median_value = train_df[column].median()
    # Fill NaN values in the column for both training and testing sets
    train_df[column] = train_df[column].fillna(median_value)
    test_df[column] = test_df[column].fillna(median_value)
    # Return the filled DataFrames
    return train_df, test_df


def evaluate_model(model, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_target):

    # Train the model
    model.fit(X_train_scaled, y_train_scaled)

    # Predict (scaled)
    y_train_pred_scaled = model.predict(X_train_scaled).reshape(-1, 1)
    y_test_pred_scaled = model.predict(X_test_scaled).reshape(-1, 1)

    # Inverse transform predictions and targets
    y_train_pred = scaler_target.inverse_transform(y_train_pred_scaled).flatten()
    y_test_pred = scaler_target.inverse_transform(y_test_pred_scaled).flatten()
    y_train_original = scaler_target.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
    y_test_original = scaler_target.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    # Calculate metrics on original scale
    train_rmse = root_mean_squared_error(y_train_original, y_train_pred)
    test_rmse = root_mean_squared_error(y_test_original, y_test_pred)
    train_mae = mean_absolute_error(y_train_original, y_train_pred)
    test_mae = mean_absolute_error(y_test_original, y_test_pred)
    train_r2 = r2_score(y_train_original, y_train_pred)
    test_r2 = r2_score(y_test_original, y_test_pred)

    # Print metrics
    print(f"{model.__class__.__name__}:")
    print(f"  Train RMSE: ${train_rmse:.2f}, Test RMSE: ${test_rmse:.2f}")
    print(f"  Train MAE: ${train_mae:.2f}, Test MAE: ${test_mae:.2f}")
    print(f"  Train R^2: {train_r2:.2f}, Test R^2: {test_r2:.2f}")

    # Return metrics
    return {
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train R^2": train_r2,
        "Test R^2": test_r2
    }

# def scale_target_train_test(y_train, y_test):

# Convert y_train and y_test to DataFrame if they are Series
#   if isinstance(y_train, pd.Series):
#      y_train = y_train.to_frame(name='target')
# if isinstance(y_test, pd.Series):
#    y_test = y_test.to_frame(name='target')

# Check for empty datasets
# if y_train.empty or y_test.empty:
#      raise ValueError('y_train or y_test is empty. Check your data splitting or preprocessing steps.')

#  # Initialize the scaler
# scaler = StandardScaler()

# Fit the scaler on y_train and transform both y_train and y_test
# y_train_scaled = pd.DataFrame(scaler.fit_transform(y_train), columns=['target'], index=y_train.index)
# y_test_scaled = pd.DataFrame(scaler.transform(y_test), columns=['target'], index=y_test.index)

# return y_train_scaled, y_test_scaled, scaler