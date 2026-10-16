# Plagiarism Detection

A machine learning and NLP project for detecting plagiarism in text, developed as a two-person academic project at **Université de Caen Normandie**.

The project implements a complete text-classification pipeline, from preprocessing and feature extraction to classification, evaluation and experimentation.

Most of the classifiers and text representations used in the project were implemented directly rather than relying exclusively on machine learning libraries.

## Features

### Text preprocessing

The preprocessing pipeline supports:

- tokenization;
- stop-word removal;
- lemmatization;
- n-gram generation.

### Feature extraction

Two main text representations are implemented:

- Count Vectorization
- TF-IDF

### Classifiers

The project includes implementations of several classification algorithms:

- Naive Bayes
- Logistic Regression
- Perceptron
- Decision Tree
- AdaBoost
- Bagging

Some experiments also compare the custom implementations with equivalents from **scikit-learn**.

### Evaluation

The experimentation tools support several evaluation metrics and analyses, including:

- accuracy;
- precision;
- recall;
- F1 score;
- balanced accuracy;
- confusion matrices;
- classifier ranking;
- learning curves;
- preprocessing and feature-extraction comparisons.

## Installation

### System dependency

The graphical interface uses Tkinter.

On Ubuntu/Debian:

```bash
sudo apt install python3-tk
```

### Python environment

Create and activate a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The project also requires the NLTK stop-word corpus and the spaCy English model:

```bash
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
```

## Graphical Interface

A Tkinter interface makes it possible to configure plagiarism detectors and visualize experiment results.

From the `plagiarism_detection/` directory:

```bash
python -m gui.app
```

## Experiments

The project contains scripts used for the experimental study presented in the report.

Run the commands below from the `plagiarism_detection/` directory.

### Corpus analysis

Generate word clouds for the corpus:

```bash
python -m experimentations.pre_analyze
```

The script generates word clouds for:

- the complete corpus;
- plagiarized texts;
- non-plagiarized texts.

Newly generated outputs are stored under:

```text
experiments_results/local/
```

Reference word clouds produced during the original project are kept in:

```text
experiments_results/wordclouds/
```

### Classifier experiments

Run the default classifier comparison:

```bash
python -m experimentations.experiments
```

The default experiment evaluates and ranks the implemented plagiarism detectors over repeated train/test splits.

Additional experiment functions are available in `experimentations/experiments.py` for:

- Count Vector vs TF-IDF comparison;
- n-gram analysis;
- stop-word removal;
- lemmatization;
- comparison with scikit-learn implementations;
- learning curves;
- confusion matrices.

Reference figures produced during the original experimental study are available in:

```text
experiments_results/reference/
```

New experiment outputs are stored locally and are not version-controlled in:

```text
experiments_results/local/
```
and their run might take a few minutes.

## Project Structure

```text
plagiarism_detection/
├── data/                  # Corpus and metadata
├── experimentations/      # Evaluation and experimental scripts
├── experiments_results/   # Reference and locally generated figures
├── gui/                   # Tkinter graphical interface
├── model/
│   ├── classifiers/       # Classification algorithms
│   ├── featurizers/       # Count Vector and TF-IDF
│   ├── pipeline/          # Plagiarism detection pipeline
│   └── preprocessors/     # Tokenization and lemmatization
├── tests/
└── util/
```

## Dataset

The project uses **A Corpus of Plagiarised Short Answers**, created at the
University of Sheffield by Paul Clough and Mark Stevenson.

The corpus contains 100 documents covering several levels of simulated
plagiarism, including near copy, light revision, heavy revision and
non-plagiarised answers.

Reference:

P. Clough and M. Stevenson, *Developing a corpus of plagiarised short
answers*, Language Resources and Evaluation, 2011.

## Documentation

The original academic report contains the detailed methodology, implementation choices and experimental analysis:

[`doc/rapport.pdf`](doc/rapport.pdf)

## Authors

Developed by:

- [Yendoumban NANMEDIGOU](https://github.com/ynanme)
- [Hadjer CHEDJARI EL MEUR](https://github.com/hadjer20039)

as an academic group project at **Université de Caen Normandie**.
