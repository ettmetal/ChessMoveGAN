# Generate all sequential pairs in an iterable, i.e.  [[i[0], i[1]], ...,
# [i[n-1], i[n]]]
def pairs_in(iterable):
        for i in range(0, len(iterable) - 1):
            yield iterable[i:i + 2]


# Yield the result of a function until that function throws an exception
def iter_until_except(function, exception):
        try:
            while True:
                yield function()
        except exception:
            pass
