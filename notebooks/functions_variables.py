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
    return pd.json_normalize(data['data']['results'])


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
