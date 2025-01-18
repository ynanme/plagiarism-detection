import os
from charset_normalizer import from_path


def read_file(file_path):
    """
    Read the file with the detected encoding.
    """
    #TODO: fix CharsetMatch problem
    result = from_path(file_path).best() #detects best encoding for the file
    if result:
        return str(result)  
    else:
        raise ValueError(f"Could not detect encoding for file: {file_path}")


def read_files (directory, targeted_extension='txt'):
    texts = []
    for file_name in os.listdir(directory):
        if file_name.endswith(targeted_extension):
            texts.append(read_file(os.path.join(directory, file_name)))
    return texts




if __name__ == "__main__":

    file_path = "../data/1/g0pA_taska.txt"
    directory = "../data/1/"

    print(read_file(file_path))
    print(read_files(directory))
    