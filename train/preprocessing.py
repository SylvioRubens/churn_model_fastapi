from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_linear_preprocessor(numeric_features, castegorical_features):
    """Function to create a linear preprocessor for numeric and categorical features.

    Args:
        numeric_features (list): list of numeric feature names
        categorical_features (list): list of categorical feature names
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handler_unknown='ignore'), castegorical_features)
        ]
    )
    
def make_tree_preprocessor(numeric_features, castegorical_features):
    """Function to create a tree preprocessor for numeric and categorical features.

    Args:
        numeric_features (list): list of numeric feature names
        categorical_features (list): list of categorical feature names
    """
    return ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), castegorical_features)
        ]
    )