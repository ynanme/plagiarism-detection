class CountVectorFeaturizer:
    
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