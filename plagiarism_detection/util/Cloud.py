import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image, ImageDraw, ImageFont
from model.featurizers.CountVectorFeaturizer import CountVectorFeaturizer
from util.util import read_files

class Cloud:
    def transform_zeros(val):
        if val == 0:
            return 255
        else:
            return val

    def Cloud(self, documents):
        mask = np.array(Image.open("assets/plagiat1.png"))
        maskable_image = np.ndarray((mask.shape[0],mask.shape[1]), np.int32)
        maskable_image = mask

        featurizer = CountVectorFeaturizer()
        featurizer.fit(documents)
        matrix = featurizer.transform(documents)

        word_occurence = np.sum(matrix, axis=0)
        vocabulary = featurizer.vocabulary
        occurence = {token: word_occurence[i] for token, i in vocabulary.items()}
        wordcloud = WordCloud(width=3000, height=2000, background_color="white", colormap="coolwarm", mask=maskable_image).generate_from_frequencies(occurence)
        plt.figure(figsize=(40, 30))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud", fontsize=15)
        plt.show()

    def create_text_mask(word, width=800, height=400, font_size=150):
        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), word, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 2
        draw.text((text_x, text_y), word, fill=255, font=font)

        return np.array(img)


if __name__ == "__main__":

    c = Cloud()
    c.Cloud(read_files("data/1/"))