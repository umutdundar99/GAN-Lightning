registered_datasets = {}


def register_dataset(dataset_name):
    def register(dataset_cls):
        if dataset_name in registered_datasets["datasets"]:
            raise ValueError(f"Model {dataset_name} is already registered")
        registered_datasets["datasets"][dataset_name] = dataset_cls
        return dataset_cls

    return register
