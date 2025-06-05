def config(path: str) -> SegmentConfig, RecognitionConfig:
    """
    Read the dataset description from a YAML file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        SegmentConfig, RecognitionConfig: Loaded configurations.
    """