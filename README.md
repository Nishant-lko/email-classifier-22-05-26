# Email Spam Classifier

A modern, interactive Deep Learning application built using **TensorFlow/Keras** and **Streamlit** to classify emails as **Spam** or **Ham (Not Spam)**.

This repository contains the front-end application, the training notebooks, and the dataset preprocessing pipeline, all consolidated under the `app/` folder.

---

## ⚠️ Important Note on Excluded Files

Due to **GitHub's file size limitations (100 MB max)** and repository size best practices, the following generated files are excluded from this repository:

1. **`spam_classifer_NN.keras`** (~1.1 GB) – The fully trained Keras Neural Network model weights.
2. **`X.csv`** (~745 MB) – The extracted TF-IDF features matrix.
3. **`tfidf_body.pkl`** & **`tfidf_subject.pkl`** – The TF-IDF vectorizers generated during preprocessing.

> [!IMPORTANT]
> If you are cloning this repository to run it locally, you must first generate these files by running the training notebooks in the `app/` directory or obtain them from the original local system directory.

---

## 📂 Project Structure

```text
email-classifier/
├── app/
│   ├── app.py                  # Streamlit web application
│   ├── messages.csv            # Raw dataset containing subject and body
│   ├── model.ipynb             # Jupyter Notebook for training the Keras Neural Network
│   └── preprocessing.ipynb     # Jupyter Notebook for text preprocessing & lemmatization
│   # Excluded: spam_classifer_NN.keras (~1.1 GB)
│   # Excluded: X.csv (~745 MB)
│   # Excluded: tfidf_body.pkl (~1.2 MB)
│   # Excluded: tfidf_subject.pkl (~66 KB)
├── .gitignore                  # Git exclusion rules
└── README.md                   # Project documentation (this file)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Nishant-lko/email-classifier.git
cd email-classifier
```

### 2. Set Up Virtual Environment & Dependencies
Ensure you have Python 3.8+ installed. Set up a virtual environment and install the required libraries:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (Powershell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install required packages
pip install tensorflow streamlit scikit-learn nltk scipy pandas 
```

### 3. Generate the Large Model & Feature Files
Before running the Streamlit application, you need the trained model weights. 
1. Open the Jupyter Notebook `app/preprocessing.ipynb` and run it to prepare the preprocessed data 
2. Open the Jupyter Notebook `app/model.ipynb` and run it. This will process the dataset and output:
   - `spam_classifer_NN.keras` (neural network model weights)
   - `X.csv` (processed features)
3. Ensure `spam_classifer_NN.keras`, `tfidf_body.pkl`, and `tfidf_subject.pkl` are located in the `app/` folder so the application can load them.

---

## 💻 Running the Streamlit App

Once the dependencies are installed and the model weights and TF-IDF vectorizer pickles are in the `app/` directory, you can run the Streamlit server:

```bash
cd app
streamlit run app.py
```

This will launch a local development server. Open your web browser and navigate to `http://localhost:8501` to use the interactive classifier!

---

##   How It Works

1. **Preprocessing**: Text is cleaned, lowercased, and lemmatized (using NLTK's `WordNetLemmatizer`) to standardize the vocabulary.
2. **Feature Extraction**: Standard TF-IDF vectorization is applied separately to the **Email Subject** and the **Email Body**.
3. **Feature Fusion**: The Subject and Body sparse TF-IDF matrices are concatenated (`scipy.sparse.hstack`) into a single feature array.
4. **Classification**: A deep Neural Network (trained using Keras/TensorFlow) processes the combined feature array to predict the probability of the email being spam.
5. **UI Presentation**: Streamlit displays the output status ("Spam" or "Ham") with clear color-coded messages, probability metrics, and a dynamic progress bar.
