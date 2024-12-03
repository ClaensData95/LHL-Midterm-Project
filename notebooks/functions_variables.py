import json
import pandas as pd


def encode_tags(df):
    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low
    counts of tags to keep cardinality to a minimum.

    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    tags = df["tags"].tolist()
    # create a unique list of tags and then create a new column for each tag

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


def reorder_columns(df):
    """
    Reorder columns in a DataFrame according to a predefined order.
    
    Parameters:
    - df: pandas.DataFrame, the input DataFrame.
    
    Returns:
    - pandas.DataFrame: A new DataFrame with columns reordered.
    """
    # Define the desired column order
    column_order = [
        "property_id", "location_address_state", "state_frequency", "location_address_city", "city_frequency",
        "year_listed", "year_sold", "description_sold_price", "price_reduced_amount",
        "days_on_market", "description_year_built", "description_stories", "description_sqft",
        "description_baths", "description_beds", "description_garage", "description_lot_sqft",
        "central_air", "community_outdoor_space", "basement", "fireplace", "hardwood_floors",
        "recreation_facilities", "community_security_features", "view", "central_heat",
        "city_view", "fenced_yard"
    ]
    
    # Ensure all columns in the desired order are present
    missing_cols = set(column_order) - set(df.columns)
    for col in missing_cols:
        df[col] = None  # Add missing columns with NaN values

    # Reorder the DataFrame columns
    reordered_df = df.reindex(columns=column_order)
    
    return reordered_df
