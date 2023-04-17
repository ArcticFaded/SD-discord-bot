class FixSizedDict(dict):
    def __init__(self, *args, maxlen=0, **kwargs):
        self._maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        if self._maxlen > 0:
            if len(self) > self._maxlen:
                self.pop(next(iter(self)))


cache = FixSizedDict(maxlen=100000)

def put_prompt(id_: str, pnginfo):
    cache[id_] = pnginfo

def get_prompt(id_: str):
    return cache[id_]
 
