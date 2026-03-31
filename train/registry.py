from mlflow.tracking import MlflowClient

    
def get_alias_version(model_name: str, alias: str = "champion") -> str | None:
    """Function to get the version number of the model currently associated with a specific alias.

    Args:
        model_name (str): Name of the registered model to query
        alias (str, optional): Alias to check for. Defaults to "champion".

    Returns:
        str | None: The version number of the model associated with the alias, or None if no model is associated
    """
    client = MlflowClient()
    try:
        alias_info = client.get_model_version_by_alias(model_name, alias)
        return alias_info.version
    except client.exceptions.RestException as e:
        if e.status_code == 404:
            return None
        else:
            raise 

def get_latest_registered_model_version(model_name: str) -> str:
    """ Function to get the latest registered model version for a given model name.

    Args:
        model_name (str): Name of the registered model to query

    Raises:
        ValueError: If no registered versions are found for the specified model name

    Returns:
        str: The version number of the latest registered model
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered versions found for model '{model_name}'")
    latest_version = max(versions, key=lambda v: int(v.version))
    return latest_version.version

def promote_model_alias(model_name: str, version: str, alias: str = "champion") -> None:
    """Method for promoting a model to a specific alias

    Args:
        model_name (str): Name of the model
        version (str): Version of the model to promote
        alias (str, optional): Alias to promote the model to. Defaults to "champion".
    """
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)