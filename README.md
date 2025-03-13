# HKU-COMP7607-A1: Exploring Word Embeddings with Word2Vec

**Course:** HKU COMP-7607 NLP (2024-2025)
**Assignment:** 1 - Exploring Word Embeddings with Word2Vec

**Important:** This codebase is exclusively for HKU COMP 7607 (2024-2025). Please do not upload your solutions or this codebase to any public platforms. All rights reserved.

## 1. Introduction: Unveiling Semantic Relationships with Word Embeddings

### 1.1. What are Word Embeddings?

Imagine you want to teach a computer to understand the meaning of words. Simply representing words as unique identifiers (like one-hot encoding) doesn't capture the rich semantic relationships between them. This is where **word embeddings** come in. They are dense vector representations of words, where semantically similar words are mapped to nearby points in a vector space.

**Why are Word Embeddings Important?**

Word embeddings are a cornerstone of many modern NLP applications, including:

*   **Machine Translation:**  Helping systems understand the meaning of words in different languages.
*   **Sentiment Analysis:**  Determining the emotional tone of text by analyzing the sentiment associated with words.
*   **Text Classification:**  Categorizing documents based on the semantic content of their words.
*   **Information Retrieval:**  Finding relevant documents based on the meaning of query words.
*   **Question Answering:**  Helping systems understand the semantic relationships between words in questions and answers.

### 1.2. Word2Vec: A Powerful Technique for Learning Word Embeddings

**Word2Vec** is a popular and efficient technique for learning high-quality word embeddings from large amounts of text data. It leverages the idea that words appearing in similar contexts tend to have similar meanings. Word2Vec offers two main architectures:

*   **Continuous Bag-of-Words (CBOW):** Predicts a target word based on its surrounding context words.
*   **Skip-gram:** Predicts the surrounding context words given a target word.

**How does Word2Vec Work?**

Both CBOW and Skip-gram are neural network models trained to perform a specific task (predicting a word or its context). The magic happens in the hidden layer of these networks. The weights of this hidden layer, after training, become the word embeddings.

### 1.3. Your Mission in This Assignment

In this assignment, you will dive into the world of word embeddings using Word2Vec. You will:

*   **Master the Basics:** Gain a solid understanding of how Word2Vec works, including the CBOW and Skip-gram architectures.
*   **Get Hands-On:** Learn to preprocess text data for training Word2Vec models.
*   **Build and Train:** Implement and train your own Word2Vec (Skip-gram) models from scratch using `torch`.
*   **Evaluate and Analyze:** Assess the quality of your learned embeddings using intrinsic evaluation methods (e.g., word analogy tasks) and visualize them using dimensionality reduction techniques like t-SNE.
*   **Become a Word Embedding Expert:** Develop a deeper understanding of the strengths and limitations of Word2Vec and its applications in NLP.

## 2. Setting Up Your Word Embedding Lab

### 2.1. HKU GPU Farm: Your High-Performance Playground (Recommended)

The HKU GPU Farm provides the computational power you need for efficient training of Word2Vec models, especially with larger datasets. Follow the provided [quickstart guide](https://www.cs.hku.hk/gpu-farm/quickstart) to set up your environment. This is the recommended approach for this assignment.

### 2.2. Local Setup: For the Resourceful

If you have a powerful local machine and have experience setting up deep learning environments, you can work locally. Ensure you have the necessary software installed and configured correctly.

### 2.3. Environment Setup: Your Toolkit

**Python:** This code is tested with Python 3.11.10.

**Virtual Environment (Recommended):** Use Anaconda to manage your project's dependencies and avoid conflicts:

```bash
conda create -n nlp_env python=3.11.10
conda activate nlp_env
```

**Install Packages:**

```bash
pip install -r requirements.txt
```

## 3. Embarking on Your Word Embedding Journey

### 3.1. Dataset: The Text8 Dataset

You will be working with the **Text8 dataset**, a cleaned subset of the English Wikipedia dump. It contains a sequence of words, providing a good starting point for learning word embeddings.

### 3.2. Preprocessing: Preparing Your Data for Word2Vec

Preprocessing is a crucial step to ensure your data is in the correct format for training Word2Vec.

**Steps:**

1.  **Load the Dataset:** Load the Text8 dataset into a suitable data structure.
2.  **Lowercase and Tokenization:** Convert all text to lowercase and split the text into a list of individual words (tokens).
3.  **Vocabulary Creation:** Create a vocabulary of unique words and assign each word a unique index.
5.  **Context Window:** Define a context window size (e.g., 5 words to the left and 5 words to the right of the target word).
6.  **Data Preparation:** Create training examples based on Skip-gram architecture and the context window.
    *   **Skip-gram:** Input: target word, Output: context words.

### 3.3. Task Description: Your Word Embedding Adventure

Your mission is divided into four key tasks:

1.  **Preprocessing and Data Preparation (dataset.py):** Implement the data loading, preprocessing, and training example generation logic.
2.  **Model Implementation (model.py):** Define the Word2Vec model (Skip-gram) using `torch`.
3.  **Evaluation (evaluation.py):** Implement code to evaluate the learned word embeddings using word analogy tasks and visualization.

### 3.4. Code Structure: Your Project's Blueprint

```text
project/
|-- data/            # Data files (e.g., Text8 dataset)
|-- test/            # Test cases for evaluation
├── constants.py     # Constants and configurations
├── model.py         # Word2Vec model (Skip-gram)
├── dataset.py       # Data loading, preprocessing, and data preparation
├── trainer.py       # Training loop and logic
├── evaluation.py    # Evaluation of word embeddings
├── main.py          # Main script to run training and evaluation
├── utils.py         # Utility functions
├── requirements.txt # Python dependencies
├── FAQ.md           # Frequently asked questions
└── README.md        # Your project's documentation
```

**Running the Code:**

```bash
python main.py
```

### 3.5. Assignment Tasks: Your Path to Word Embedding Mastery

#### Task 1: Preprocessing and Data Preparation (dataset.py)

*   **Objective:** Correctly load, preprocess, and prepare the Text8 dataset for training Word2Vec.
*   **File:** `dataset.py`
*   **Instructions:** Complete all sections marked with "`Write Your Code Here`".
*   **Hints:**
    *   Use efficient data structures to store the vocabulary and word indices.
    *   Carefully generate training examples based Skip-gram architecture and context window.
    *   Consider adding a function to visualize a few examples of prepared data to verify your implementation.
    *   After you finished this task, please make sure that you pass the test case in `test/test_dataset.py` by running the following command:

        ```bash
        pytest test/test_dataset.py
        ```

        If you pass the test case, you can move on to the next task.

#### Task 2: Model Implementation (model.py)

*   **Objective:** Define the Word2Vec model (Skip-gram) using `torch`.
*   **File:** `model.py`
*   **Instructions:** Complete the sections marked with "`Write Your Code Here`".
*   **Hints:**
    *   Use `torch.nn.Embedding` to represent the input and output word embeddings.
    *   For Skip-gram, you'll predict each context word separately.
    *   Use `torch.nn.Linear` for the output layer.
    *   After you finished this task, please make sure that you pass the test case in `test/test_model.py` by running the following command:

        ```bash
        pytest test/test_model.py
        ```

        If you pass the test case, you can move on to the next task.

#### Task 3: Evaluation (evaluation.py)

*   **Objective:** Evaluate the learned word embeddings using intrinsic evaluation methods and visualization.
*   **File:** `evaluation.py`
*   **Instructions:** Complete the sections marked with "`Write Your Code Here`".
*   **Hints:**
    *   **Word Analogy Task:** Implement a function to solve analogies like "king - man + woman = queen". You can use cosine similarity to find the closest word vector.
    *   **Visualization:** Use t-SNE (from `sklearn.manifold` or other libraries) to reduce the dimensionality of the word embeddings to 2D and visualize them. Plot the embeddings of a subset of words to observe semantic relationships.
    *   After you finished this task, please make sure that you pass the test case in `test/test_evaluation.py` by running the following command:

        ```bash
        pytest test/test_evaluation.py
        ```

        If you pass the test case, you can move on to the next task.

### 3.6. Submission: Packaging Your Word Embedding Masterpiece

**If your student ID is 30300xxxxx, organize your submission as follows:**

```text
30300xxxxx.zip
|-- data/            # Data files (e.g., Text8 dataset)
|-- test/            # Test cases for evaluation
├── constants.py     # Constants and configurations
├── model.py         # Word2Vec model (Skip-gram)
├── dataset.py       # Data loading, preprocessing, and data preparation
├── trainer.py       # Training loop and logic
├── evaluation.py    # Evaluation of word embeddings
├── main.py          # Main script to run training and evaluation
├── utils.py         # Utility functions
├── requirements.txt # Python dependencies
├── FAQ.md           # Frequently asked questions
└── README.md        # Your project's documentation
```

* **Code Files:** All your modified code files.
* **Submission Format:** Zip archive with your student ID as the filename.

### 3.7. Submission Deadline

**Deadline:** Mar 20th (23:59)

**Late Submission Policy:**

* 10% penalty within 1 day late.
* 20% penalty within 2 days late.
* 50% penalty within 7 days late.
* 100% penalty after 7 days late.

## 4. Grading: Your Path to Recognition

We will re-run your `main.py` script.

**Important Considerations:**

1. **Submission Format:** Follow the submission guidelines to ensure your work is correctly evaluated.
2. **Error-Free Execution:** Your code must run without any errors.
4. **Quality of Word Embeddings:** The quality of your word embeddings will be assessed based on their performance in word analogy tasks and visualization.
5. **Code Quality:** Your code should be well-structured, readable, and well-documented.

**Grading Breakdown:**

* **fail to pass all test cases:** 0
* **pass the one test cases:** 40%
* **pass the two test cases:** 60%
* **pass the three test cases:** 80%
* **pass the three test cases and achieve good performance in word analogy tasks and visualization:** 100%

## 5.  Need Help?

Please check [frequency asked questions](FAQ.md) (FAQ) first. If you have any questions or need clarification, feel free to reach out to the course instructor or teaching assistants.
