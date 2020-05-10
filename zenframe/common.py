def delegate(method_name, attribute):
    def to_attribute(self, *args, **kwargs):
        bound_method = getattr(getattr(self, attribute), method_name)
        return bound_method(*args, **kwargs)

    return to_attribute
