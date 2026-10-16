from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

from experimentations.experiments_utils import *
from model.preprocessors.tokenizer import Tokenizer


class Cloud:
    
    @staticmethod
    def transform_zeros(val):
        if val == 0:
            return 255
        else:
            return val

    def get_word_frequencies(self, documents):
        featurizer = CountVectorizer()
        matrix = featurizer.fit_transform(documents)
        word_occurence = np.sum(matrix, axis=0)
        vocabulary = featurizer.vocabulary_
        occurence = {token: word_occurence[0, i] for token, i in vocabulary.items()}
        return occurence

    def wordcloud(self, documents, documents_to_exclude=[], frequency_diff=500, title="WordCloud", saving_results_path=None):
        frequencies = self.get_word_frequencies(documents)
        filtered_frequencies = frequencies
        if len(documents_to_exclude) > 0:
            frequencies_to_exclude = self.get_word_frequencies(documents_to_exclude)
            filtered_frequencies = {}
            for word, freq in frequencies.items():
                if word in frequencies_to_exclude:
                    if abs(freq - frequencies_to_exclude[word]) > frequency_diff:
                        filtered_frequencies[word] = freq
                else:
                    filtered_frequencies[word] = freq

        wordcloud = WordCloud(width=3000, height=2000, background_color="white", colormap="coolwarm").generate_from_frequencies(filtered_frequencies)
        
        plt.figure(figsize=(40, 30))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title, fontsize=75)
        if saving_results_path is not None:
            plt.savefig(saving_results_path)
        plt.show()


if __name__ == "__main__":

    data_directory = "data/1"
    results_path = "experiments_results/local/"
    texts, categories = load_data(data_directory)

    plagiarism_texts = [texts[i] for i in range(len(texts)) if categories[i] == 1]
    non_plagiarism_texts = [texts[i] for i in range(len(texts)) if categories[i] == 0]

    tokenizer = Tokenizer()

    texts_tokens = [tokenizer.tokenize(text) for text in texts]
    plagiarism_tokens = [tokenizer.tokenize(text) for text in plagiarism_texts]
    plagiarism_tokens = [tokenizer.tokenize(text) for text in plagiarism_texts]
    non_plagiarism_tokens = [tokenizer.tokenize(text) for text in non_plagiarism_texts]

    texts_tokenized = [token for tokens in texts_tokens for token in tokens]
    plagiarism_texts_tokenized = [token for tokens in plagiarism_tokens for token in tokens]
    non_plagiarism_texts_tokenized = [token for tokens in non_plagiarism_tokens for token in tokens]

    cloud = Cloud()
    cloud.wordcloud(texts_tokenized, title="All texts", saving_results_path=f"{results_path}all_texts.png")
    cloud.wordcloud(plagiarism_texts_tokenized, non_plagiarism_texts_tokenized, title="Plagiarism texts", saving_results_path=f"{results_path}non_plagiarism.png")
    cloud.wordcloud(non_plagiarism_texts_tokenized, plagiarism_texts_tokenized, title=" Non Plagiarism texts", saving_results_path=f"{results_path}plagiarism.png")
