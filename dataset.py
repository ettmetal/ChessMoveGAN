from helpers.conversion import *
import msgpack


# Get a generator that will return x,y pairs from files at x_path and y_path
# files should be ordered identically, so that the matching x and y are at the same index
# Parameter binary signifies that the messagepack file was saved in binary mode
def paired_messagepack_file_dataset_generator(x_path, y_path, binary=True):
    x_file = open(x_path, 'r%s' % ('b' if binary else ''))
    y_file = open(y_path, 'r%s' % ('b' if binary else ''))

    # The generator to return
    def __next():
        nonlocal x_file # Capture closed vriables
        nonlocal y_file

        # Inner generator to provide x,y pairs with file reset when required
        def unpack():
            def get_reset_unpackers(): # resets files' seek positions and returns reinitialised unpackers
                x_file.seek(0)
                y_file.seek(0)
                return msgpack.Unpacker(x_file), msgpack.Unpacker(y_file)
            x_unpacker, y_unpacker = get_reset_unpackers()
            while True: # If unpackers have data, yield it else reset them & yield
                try:
                    next_x, next_y = x_unpacker.unpack(), y_unpacker.unpack()
                    yield next_x, next_y
                except msgpack.OutOfData:
                    x_unpacker, y_unpacker = get_reset_unpackers()
                    yield x_unpacker.unpack(), y_unpacker.unpack()

        for x, y in unpack():
            y = az_index_to_az(y) # Inflate y (move index) to one-hot
            yield x, y

    return __next
