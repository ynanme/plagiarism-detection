import os
import random
import math
import matplotlib.pyplot as plt
from charset_normalizer import from_path


"""
python3 -c "import numpy; print(numpy.__version__)"
pip install --upgrade pip
pip install numpy==1.24.4
pip install --upgrade scipy
pip install --upgrade scipy
pip install --upgrade scikit-learn
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
"""


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


def read_files (directory, targeted_extension='txt', files_names_required=False):
    texts = []
    files_names = []
    for file_name in os.listdir(directory):
        if file_name.endswith(targeted_extension):
            texts.append(read_file(os.path.join(directory, file_name)))
            files_names.append(file_name)
    if files_names_required:
        return texts, files_names
    return texts


def partition (elements, weights, partition_on_indexes=False):
    if math.isclose(sum(weights), 1):
        all_count = len(elements)
        elements_ = elements.copy() if not partition_on_indexes else [index for index in range(all_count)]
        parts = []
        for weight in weights:
            count = round(weight * all_count)
            part = []
            for _ in range(count):
                if len(elements_) > 0:
                    element = random.choice(elements_)
                    part.append(element)
                    elements_.remove(element)
            if len(part) > 0:
                parts.append(part)
        return parts
    return partition(elements, weights + [1-sum(weights)])


def draw_plot (
    X, Ys,
    plot_labels = [], axis_labels = ["X", "Y"], title = "Plots",
    show_figure = True, saving_figure_path = None
):
    plt.figure()
    if plot_labels == []:
        plot_labels = ["Plot " + str(i) for i in range(1,len(Ys)+1)]
    for Y, label in zip(Ys, plot_labels):
        plt.plot(X, Y, label=label)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.legend()
    if saving_figure_path is not None:
        plt.savefig(saving_figure_path, bbox_inches='tight')
    if show_figure:
        plt.show()
    

def file_name (description, unwanted=",;!?", linker="_"):
    description_clean = "".join(char for char in description if char not in unwanted)
    description_lower = description_clean.strip().lower()
    words = description_lower.split()
    return linker.join(words)



if __name__ == "__main__":

    """
    file_path = "../data/1/g0pA_taska.txt"
    directory = "../data/1/"

    print(read_file(file_path))
    print(read_files(directory))
    
    n, m, a, b = 3, 10, -4, 15
    draw_plot(
        [[random.randint(a,b) for _ in range(m)] for _ in range(n)],
        [[random.randint(a,b) for _ in range(m)] for _ in range(n)],
        ["le renard", "le loup", "le lièvre"], ["la vie", "la mort"], "Les animaux",
        True, "test"
    )

    print(file_name("Evaluating on untrained data, training data is 80.0%"))
    """

    words = ["running", "better", "feet", "playing", "was", "studies"]
    print(lemmatize(words))