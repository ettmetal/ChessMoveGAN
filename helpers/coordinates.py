# Coordinate transformations

def flatten_3d_coords(coords, dims):
    return coords[0] + (coords[1] * dims[0]) + (coords[2] * dims[0] * dims[1])


def flatten_2d_coords(coords, dims):  # Row-major
    return coords[1] * dims[0] + coords[0]


def inflate_1d_coords_3d(index, dims):
    return (index % dims[0], (index // dims[0]) % dims[1], index // (dims[0] * dims[1]))


def inflate_1d_coords_2d(index, dims):  # Row-major
    return (index % dims[0], index // dims[0])