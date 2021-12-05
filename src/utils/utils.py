SEED = 15


def filter_by_key(key_filter, object):
    if isinstance(key_filter, str):
        if key_filter == 'all':
            filtered_object = object
        else:
            filtered_object = {n: m for n,
                               m in object.items() if n == key_filter}
    elif isinstance(key_filter, list):
        filtered_object = {n: m for n, m in object.items() if n in key_filter}
    else:
        raise TypeError(
            f'filter must be a str or list, but received {type(key_filter)}')

    if not filtered_object:
        raise ValueError(
            f"{key_filter} is an invalid value. Select one of the following: all, {', '.join(object.keys())}")
    return filtered_object
