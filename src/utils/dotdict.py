class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == '__main__':

    d = DotDict({'foo': 42})
    print(d.foo)  # 42
