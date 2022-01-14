def cache(f):
    cache_dict = {}
    
    def wrapper(*args, **kwargs):
        ar = list(args)
        kw = list(zip(kwargs.keys(), kwargs.values()))
        if not cache_dict.get(tuple(ar + kw)):
            value = f(*args, **kwargs)
            cache_dict[tuple(ar + kw)] = value
            return value
        return cache_dict[tuple(ar + kw)]
    
    return wrapper
