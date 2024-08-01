registered_models = {}

def model_registration(model_name):
    def register(dataset_cls):
        if model_name in registered_models:
            raise ValueError(f"Model {model_name} is already registered")
        registered_models[model_name] = dataset_cls
        return dataset_cls

    return register
