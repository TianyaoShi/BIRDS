from .materialize import materialize_from_config, prepare

materialize_dataset_from_config = materialize_from_config
prepare_dataset = prepare

__all__ = [
    "materialize_dataset_from_config",
    "materialize_from_config",
    "prepare",
    "prepare_dataset",
]
