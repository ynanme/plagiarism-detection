from nltk.corpus import stopwords
import re
from typing import List
from charset_normalizer import from_path


# In[2]:
class Tokenizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english')) #stop words from nltk
        self.pattern = re.compile(r"\w+(?:[-'’]\w+)?", re.VERBOSE) #compile the regular expressions

    def tokenize(self, text: str) -> List[str]:
        text = text.lower() #transforme all the text to lowercase
        tokens = self.pattern.findall(text)  #extract all tokens from the input text that matches the pattern
        
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
         
        #print("tokens",tokens)
        #print("\n")
        #print("tokens without stopwords",filtered_sentence)
        #print("and" in filtered_sentence)
        return filtered_tokens
    
    @staticmethod
    def read_file(file_path):
        #TODO: fix CharsetMatch problem
        result = from_path(file_path).best() #detects best encoding for the file
        if result:
            return str(result)  
        else:
            raise ValueError(f"Could not detect encoding for file: {file_path}")


# In[3]:


"""class CountVectorFeaturizer:
    def __init__(self):
        self.vocabulary = {}  
        self.tokenizer = Tokenizer()
        self.tokenized_docs = []

    def fit(self, documents: List[str]) -> None:
        token_counter = Counter()  # Counter to keep track of token frequencies
        self.tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]
        
        # Update the counter with tokens from each document
        for tokens in self.tokenized_docs:
            token_counter.update(tokens)
        
        # Map each token to a unique index based on frequency
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(token_counter.most_common())}

    def transform(self, documents: List[str]) -> np.ndarray:

        # Initialize a matrix for the count vectors
        matrix = np.zeros((len(documents), len(self.vocabulary)), dtype=int)

        # Fill the matrix with token frequencies based on the vocabulary
        for i, tokens in enumerate(self.tokenized_docs):
            for token in tokens:
                idx = self.vocabulary.get(token)
                if idx is not None:
                    matrix[i, idx] += 1
        
        return matrix"""


# In[4]:


"""folder_path = "../data-plagiarism"
documents = []

featurizer = CountVectorFeaturizer()

for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        try:
            content = Tokenizer.read_file(file_path)  # Use the Tokenizer class's file reading method
            documents.append(content)
        except ValueError as e:
            print(f"Error reading {file_name}: {e}")

# Fit the featurizer and transform the documents into a count matrix
featurizer.fit(documents)
count_matrix = featurizer.transform(documents)

# Print the results
#print("Vocabulary:", featurizer.vocabulary)
print("Count Matrix:\n", count_matrix)"""


# In[5]:


"""class CountVectorFeaturizer:
    def __init__(self):
        self.vocabulary = {}  

    def fit(self, documents: List[str]) -> None:
        
        token_counter = Counter() #using a counter to keep track of frequencies 
        self.tokenized_docs = []
        
        # Tokenize all docs
        self.tokenized_docs = [tokenize(doc) for doc in documents]
        for tokens in self.tokenized_docs:
            token_counter.update(tokens) #extract unique tokens
        
        # Map each token to a unique index based on frequency
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(token_counter.most_common())}

    def transform(self, documents: List[str]) -> np.ndarray:
        
        # Initialize a matrix for count vectors Featurizer
        matrix = np.zeros((len(documents), len(self.vocabulary)), dtype=int)

        #counting freq of each token in the vocab
        for i, tokens in enumerate(self.tokenized_docs):
            for token in tokens:
                idx = self.vocabulary.get(token)
                if idx is not None:
                    matrix[i, idx] += 1
        
        return matrix
"""


# In[6]:


"""folder_path = "../data-plagiarism"
documents = []

featurizer = CountVectorFeaturizer()
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"): 
        file_path = os.path.join(folder_path, file_name)
        content = read_file(file_path)
        documents.append(content)

featurizer.fit(documents)
count_matrix = featurizer.transform(documents)

#print("Vocabulary:", featurizer.vocabulary)
#print("Count Matrix:\n", count_matrix)
#print("\n") """


# In[7]:


"""expected_results = pd.read_csv(os.path.join(folder_path, "file_information.csv"), sep=",")
#print(expected_results)

#extracting labels that we'll be using in our classif
labels = dict(zip(expected_results["File"], expected_results['Category']))

#combine heavy and cut as plagiat, light orig as non_plagiat
def map_value(value):
    return 'plagiat' if value in ['heavy', 'cut'] else 'non_plagiat'

filtered_labels = { k: map_value(v) for k, v in sorted(labels.items()) } #filters the files

total_files = len(filtered_labels)
print("total_files", filtered_labels)

plagiat_files = sum(1 for v in filtered_labels.values() if v == "plagiat")
#print("plagiat_files", plagiat_files)
probability_plagiat = plagiat_files/total_files
#print("probability_plagiat", probability_plagiat)
#print("probability_plagiat", 1-probability_plagiat)
#print("filtered_labels", filtered_labels)

totals = np.zeros((len(featurizer.vocabulary), 2))  # shape: (vocab_size, 2)

# Iterate over the vocabulary
for word, index in featurizer.vocabulary.items(): #for each word in my vocab
    for file_name, row in zip(filtered_labels.keys(), count_matrix):  #for each file
        # Add the count of the current word to the total
        totals[index, 0] += row[index]

        # Check if the label for the current file is "plagiat"
        if filtered_labels[file_name] == 'plagiat':
            totals[index, 1] += row[index]

#print(word_totals)
# Convert results to a dictionary for readability
word_totals = {word: {"total": totals[index, 0], "total_plagiat": totals[index, 1]} 
               for word, index in featurizer.vocabulary.items()}

# Debugging or printing results
total_plagiat_words = 0
total_non_plagiat_words = 0

for file_name, row in zip(filtered_labels.keys(), count_matrix):
    if filtered_labels[file_name] == 'plagiat':
        total_plagiat_words += row.sum()
    else:
        total_non_plagiat_words += row.sum()

#print("total_plagiat_words", total_plagiat_words)
#print("total_non_plagiat_words", total_non_plagiat_words)"""


# In[8]:


"""data = []
for word, stats in word_totals.items():
    total = stats['total']
    total_plagiat = stats['total_plagiat']
    plagiat_ratio = (total_plagiat + 1) / total_plagiat_words  if total > 0 else 0  
    non_plagiat_ratio = (total - total_plagiat + 1) / total_non_plagiat_words 
    data.append([word, total, total_plagiat, plagiat_ratio, non_plagiat_ratio])

columns = ['Word', 'Total', 'Total Plagiat', 'Plagiat Ratio', 'Non Plagiat Ratio']

df = pd.DataFrame(data, columns=columns)

print(df.to_string(index=False))  # Removes the index for a cleaner look
"""


# In[9]:


"""class NaiveBayesClassifier:
    def __init__(self, class_priors, word_probabilities):
        self.class_priors = class_priors
        self.word_probabilities = word_probabilities

    def predict(self, tokens):
        scores = {cls: np.log(self.class_priors[cls]) for cls in self.class_priors}

        for token in tokens:
            if token in self.word_probabilities:
                for cls in scores:
                    scores[cls] += np.log(self.word_probabilities[token][cls])

        return max(scores, key=scores.get)
"""


# In[10]:


"""# Class priors
class_priors = {
    "plagiat": probability_plagiat,
    "non_plagiat": 1 - probability_plagiat
}

# Word probabilities
word_probabilities = {
    row["Word"]: {
        "plagiat": row["Plagiat Ratio"],
        "non_plagiat": row["Non Plagiat Ratio"]
    }
    for _, row in df.iterrows()
}

# Initialize classifier
classifier = NaiveBayesClassifier(class_priors, word_probabilities)

# Tokenize a test document (replace with your tokenizer)
test_document = featurizer.tokenizer.tokenize(Tokenizer.read_file("../data-plagiarism/g4pD_taska.txt"))

# Predict class
prediction = classifier.predict(test_document)
print(f"Predicted class: {prediction}")

results = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"): 
        file_path = os.path.join(folder_path, file_name)
        
        # Read and tokenize the document
        content = Tokenizer.read_file(file_path)
        test_document = featurizer.tokenizer.tokenize(content)  # Replace with your tokenizer
        
        # Predict the class using the Naive Bayes classifier
        prediction = classifier.predict(test_document)
        
        # Get the previous label
        previous_label = filtered_labels[file_name]
        
        # Append the result to the list
        results.append({
            "File Name": file_name,
            "Previous Label": previous_label,
            "Predicted Label": prediction
        })"""


# In[11]:


"""# Create a DataFrame from the results
print(results)
df_results = pd.DataFrame(results)
# Add a comparison column to check mismatches
df_results["Mismatch"] = df_results["Previous Label"] != df_results["Predicted Label"]
df_differences = df_results[df_results["Mismatch"]]
# Display the table
print(df_differences.to_string(index=False))  # Print without the index for clarity"""


# In[ ]:





# In[ ]:




