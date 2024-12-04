import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def encode_tags(df, top_n=10):
    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low
    counts of tags to keep cardinality to a minimum.

    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
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
    """
    Reorder columns in both training and testing DataFrames according to a predefined order.
    
    Parameters:
    - train_df: pandas.DataFrame, the training DataFrame.
    - test_df: pandas.DataFrame, the testing DataFrame.
    
    Returns:
    - train_df, test_df: DataFrames with columns reordered.
    """
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
    """
    Fills NaN values in a column by sampling from the non-NaN values in the training set.
    
    Parameters:
    - train_df: pandas.DataFrame, the training DataFrame.
    - test_df: pandas.DataFrame, the testing DataFrame.
    - column: str, the name of the column to fill.
    
    Returns:
    - train_df, test_df: DataFrames with NaN values filled in the specified column.
    """
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
    """
    Fills NaN values in a column using the median calculated from the training set.
    
    Parameters:
    - train_df: pandas.DataFrame, the training DataFrame.
    - test_df: pandas.DataFrame, the testing DataFrame.
    - column: str, the name of the column to fill.
    
    Returns:
    - train_df, test_df: DataFrames with NaN values filled in the specified column.
    """
    # Calculate the median from the training set
    median_value = train_df[column].median()

    # Fill NaN values in the column for both training and testing sets
    train_df[column] = train_df[column].fillna(median_value)
    test_df[column] = test_df[column].fillna(median_value)

    return train_df, test_df


def scale_columns_train_test(train_df, test_df, columns):
    """
    Scales specified columns in the training and testing DataFrames using StandardScaler.
    
    Parameters:
    - train_df: pandas.DataFrame, the training DataFrame.
    - test_df: pandas.DataFrame, the testing DataFrame.
    - columns: list of str, the column names to scale.
    
    Returns:
    - train_df, test_df: DataFrames with the specified columns scaled.
    """
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler on the training data and transform both train and test sets
    train_df[columns] = scaler.fit_transform(train_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])
    
    return train_df, test_df