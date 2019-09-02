import os


# Creates a directiory if it does not exist
def ensure_dir_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
