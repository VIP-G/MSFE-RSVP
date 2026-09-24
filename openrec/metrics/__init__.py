import copy

__all__ = ['build_metric']

support_dict = ['RecMetric']


def build_metric(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'metric only support {}'.format(support_dict))

    # Lazy import
    if module_name == 'RecMetric':
        from .rec_metric import RecMetric
        module_class = RecMetric(**config)
    return module_class
