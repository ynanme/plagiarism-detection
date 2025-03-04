import os
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split

from model.tokenizers.tokenizer import Tokenizer

from model.featurizers.CountVectorFeaturizer import CountVectorFeaturizer
from model.featurizers.TfidfFeaturizer import TfidfFeaturizer

from model.classifiers.NaiveBayesClassifier import NaiveBayesClassifier
from model.classifiers.LogisticRegressionClassifier import LogisticRegressionClassifier 
from model.classifiers.PerceptronClassifier import PerceptronClassifier

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image, ImageDraw, ImageFont


def map_value(value: str) -> str:
    return 'plagiat' if value in ['heavy', 'cut', 'light'] else 'non_plagiat'

def load_labels(folder_path: str, file_name: str) -> Dict[str, str]:
    expected_results = pd.read_csv(os.path.join(folder_path, file_name), sep=",")
    labels = dict(zip(expected_results["File"], expected_results['Category']))
    return {k: map_value(v) for k, v in sorted(labels.items())}

def transform_zeros(val):
    if val == 0:
        return 255
    else:
        return val

def Cloud(documents):

    #mask = create_text_mask(word="PLAGIAT")
    mask = np.array(Image.open("assets/plagiat1.png"))
    print("mask", mask)
    maskable_image = np.ndarray((mask.shape[0],mask.shape[1]), np.int32)
    maskable_image = mask
    #for i in range(len(mask)):
    #    maskable_image[i] = list(map(transform_zeros, mask[i]))

    #print(maskable_image)

    featurizer = CountVectorFeaturizer()
    featurizer.fit(documents)
    matrix = featurizer.transform(documents)

    word_occurence = np.sum(matrix, axis=0)
    print(word_occurence)
    vocabulary = featurizer.vocabulary
    occurence = {token: word_occurence[i] for token, i in vocabulary.items()}
    wordcloud = WordCloud(width=3000, height=2000, background_color="white", colormap="coolwarm", mask=maskable_image).generate_from_frequencies(occurence)
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud", fontsize=15)
    plt.show()

def create_text_mask(word, width=800, height=400, font_size=150):
    """Creates a mask image with the given word in white on a black background."""
    img = Image.new("L", (width, height), 0)  # Black background
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Choose a font
    except IOError:
        font = ImageFont.load_default()  # Fallback if the font is missing

    bbox = draw.textbbox((0, 0), word, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2
    draw.text((text_x, text_y), word, fill=255, font=font)  # Draw the word in white

    return np.array(img)




if __name__ == "__main__":
    folder_path = "../data-plagiarism"

    filtered_labels = load_labels(folder_path, "file_information.csv")
    labels_df = pd.DataFrame(list(filtered_labels.items()), columns=['File', 'label'])

    plagiat_df = labels_df[labels_df['label'] == 'plagiat']
    non_plagiat_df = labels_df[labels_df['label'] == 'non_plagiat']

    test_size = 0.25
    plagiat_test_size = int(len(plagiat_df) * test_size)
    non_plagiat_test_size = int(len(non_plagiat_df) * test_size)

    plagiat_train, plagiat_test = train_test_split(
        plagiat_df, test_size=plagiat_test_size, random_state=42
    )

    non_plagiat_train, non_plagiat_test = train_test_split(
        non_plagiat_df, test_size=non_plagiat_test_size, random_state=42
    )

    train_df = pd.concat([plagiat_train, non_plagiat_train])
    test_df = pd.concat([plagiat_test, non_plagiat_test])

    train_labels = dict(zip(train_df['File'], train_df['label']))
    test_labels = dict(zip(test_df['File'], test_df['label']))
    # test_labels = load_labels(folder_path, "test_info.csv")
    # train_labels = {file: label for file, label in filtered_labels.items() if file not in test_labels}
    #print(len(train_labels), )
    #cloud_labels = {file: label for file, label in filtered_labels.items() if label == "plagiat"}
    #test = {file: label for file, label in test_labels.items() if label == "plagiat"}
    #train = {file: label for file, label in train_labels.items() if label == "plagiat"}
    #print('total plagiat',len(cloud_labels))
    # print('total non plagiat',len(filtered_labels) - len(cloud_labels))
    print("Test plagiat count:", len(plagiat_test))
    print("Test non-plagiat count:", len(non_plagiat_test))
    print("Train plagiat count:", len(plagiat_train))
    print("Train non-plagiat count:", len(non_plagiat_train))
    #print(len(train_labels))
    documents = [Tokenizer.read_file(os.path.join(folder_path, file)) for file in train_labels.keys()]  # Read all documents

    #cloud_documents = [Tokenizer.read_file(os.path.join(folder_path, file)) for file in cloud_labels.keys()]
    #Cloud(cloud_documents)
    
    for featurizer_class in [CountVectorFeaturizer]:
        featurizer = featurizer_class()
        featurizer.fit(documents)
        matrix = featurizer.transform(documents)
        #print(matrix)
        for classifier_class in [NaiveBayesClassifier, LogisticRegressionClassifier, PerceptronClassifier]:
            classifier = classifier_class()

            binary_labels = (
                np.array([1 if label == "plagiat" else 0 for label in train_labels.values()])
                if classifier_class in [LogisticRegressionClassifier, NaiveBayesClassifier]
                else [1 if label == "plagiat" else -1 for label in train_labels.values()]
            )
            if classifier_class in [PerceptronClassifier, LogisticRegressionClassifier]:
                classifier.fit(matrix, binary_labels)
            else:
                classifier.fit(binary_labels, featurizer, matrix)

            results = []
            for file_name, feature_vector in zip(test_labels.keys(), matrix):
                if classifier_class in [LogisticRegressionClassifier, PerceptronClassifier]:

                    prediction = classifier.predict([feature_vector])[0] # or can use .reshape(1, -1)
                    predicted_label = "plagiat" if prediction == 1 else "non_plagiat"
                else:
                    content = Tokenizer.read_file(os.path.join(folder_path, file_name))
                    tokens = featurizer.tokenizer.tokenize(content)
                    predicted_label = classifier.predict(tokens)

                results.append({
                    "File Name": file_name,
                    "Expected Label": filtered_labels[file_name],
                    "Predicted Label": predicted_label
                })

            df_results = pd.DataFrame(results)
            df_results["Mismatch"] = df_results["Expected Label"] != df_results["Predicted Label"]

            print(f"\nResults for {featurizer_class.__name__} with {classifier_class.__name__}:")
            print(df_results.to_string(index=False))

            mismatches = df_results[df_results["Mismatch"]]
            print(f"Mismatches for {featurizer_class.__name__} with {classifier_class.__name__}: {len(mismatches)}\n")
            #print(mismatches.to_string(index=False))
