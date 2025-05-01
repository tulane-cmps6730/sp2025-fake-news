# ================================
#   Disable TensorFlow inside Transformers
# ================================
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # must be set before importing transformers

# ================================
# Hugging Face Login  (optional — remove if not pushing models)
# ================================
# from huggingface_hub import login
# login(token="YOUR_TOKEN")

# ================================
# Import Libraries
# ================================

# Import necessary libraries for the task
import torch  # For tensor operations
import pandas as pd  # For loading and handling data
import numpy as np  # For numerical operations
from sklearn.linear_model import LogisticRegression  # For the logistic regression model
from sklearn.decomposition import PCA  # For dimensionality reduction (PCA)
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, classification_report  # For model evaluation
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from transformers import pipeline  # For using Hugging Face pipelines

# ======================================================
# Load Features (train_features.pt and test_features.pt)
# ======================================================

# Load the pre-extracted features (BERT embeddings) for training and testing data
train_features = torch.load("data/preprocessed/train_features.pt", map_location="cpu").to(torch.float32)  # Load training features
test_features = torch.load("data/preprocessed/test_features.pt", map_location="cpu").to(torch.float32)  # Load testing features

# ======================================================
# Load CSV files (train.csv and test.csv) for Labels
# ======================================================

# Load the train and test CSV files to extract labels for evaluation
train_df = pd.read_csv("data/preprocessed/train.csv")  # Load the training data CSV
test_df = pd.read_csv("data/preprocessed/test.csv")  # Load the testing data CSV

# ================================
# Extract NER Features from Text
# ================================

# Load a smaller Named Entity Recognition (NER) model from Hugging Face for text extraction
ner_pipe = pipeline(
    "ner",  # NER task
    model="dbmdz/bert-base-cased-finetuned-conll03-english",  # A smaller, efficient model
    tokenizer="dbmdz/bert-base-cased-finetuned-conll03-english",  # Use the same tokenizer
    aggregation_strategy="simple",
    framework="pt"        # <-- Forces PyTorch backend
)

# Define a function to extract entity features from text
def extract_entity_features(texts):
    features = []  # List to store NER features for each text
    for text in texts:
        if not isinstance(text, str) or not text.strip():  # If text is empty or not a string
            features.append([0, 0, 0, 0])  # No entities detected, append zeros
            continue
        ents = ner_pipe(text[:512])  # Get entities for the first 512 tokens (due to BERT token limit)
        counts = {"PER": 0, "ORG": 0, "LOC": 0, "MISC": 0}  # Initialize counts for each entity type
        for ent in ents:
            label = ent["entity_group"]  # Extract the entity type (PERSON, ORGANIZATION, LOCATION, MISC)
            if label in counts:  # If the label is valid, increment the corresponding count
                counts[label] += 1
        features.append([counts["PER"], counts["ORG"], counts["LOC"], counts["MISC"]])  # Append the counts to features
    return np.array(features, dtype=np.float32)  # Convert list to numpy array for efficiency

# Extract NER features for both train and test datasets
train_ner = extract_entity_features(train_df["processed_text"].tolist())  # Extract NER features from train set
test_ner = extract_entity_features(test_df["processed_text"].tolist())    # Extract NER features from test set

# ===========================================
# Combine NER Features with BERT Features
# ===========================================

# Combine the BERT features (from `train_features.pt` and `test_features.pt`) with the NER features
X_train_combo = np.concatenate([train_features.numpy(), train_ner], axis=1)  # Concatenate train BERT features and NER features
X_test_combo = np.concatenate([test_features.numpy(), test_ner], axis=1)    # Concatenate test BERT features and NER features

# ===========================================
# Apply PCA for Dimensionality Reduction
# ===========================================

# Apply Principal Component Analysis (PCA) to reduce dimensionality for better performance and efficiency
pca = PCA(n_components=100)  # Set the number of components (100 for reduction)
X_train_pca = pca.fit_transform(X_train_combo)  # Apply PCA on training data
X_test_pca = pca.transform(X_test_combo)  # Apply PCA on test data (use fitted PCA)

# ================================
# Prepare Labels for Training
# ================================

# Ensure that there are no missing labels and map them correctly to 0 (real) and 1 (fake)
train_df['label'] = train_df['label'].fillna(0)  # Replace NaN with 0 for real news
test_df['label'] = test_df['label'].fillna(0)    # Replace NaN with 0 for real news

# Map labels to binary format: 0 for real and 1 for fake
y_train = train_df['label'].values  # Get the labels for training data
y_test = test_df['label'].values    # Get the labels for testing data

# ======================================================
# Train Logistic Regression with BERT + NER Features
# ======================================================

# Initialize and train the Logistic Regression model using the combined BERT + NER features
clf = LogisticRegression(max_iter=1000)  # Logistic Regression with a maximum of 1000 iterations for convergence
clf.fit(X_train_pca, y_train)  # Train the model using PCA-reduced features
y_pred = clf.predict(X_test_pca)  # Predict the labels for the test set

# ===========================================
# Model Evaluation (Logistic Regression)
# ===========================================

# Evaluate the model's performance using various metrics (Accuracy, Precision, Recall, F1-score)
print("Evaluation: Logistic Regression (BERT + Features)")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

# ============================================================
# ROC-AUC Curve and Score for Logistic Regression (BERT + NER)
# ============================================================

# Get the predicted probabilities for the test data
y_prob = clf.predict_proba(X_test_pca)[:, 1]  # Probability for the 'fake' class (class 1)

# Compute ROC-AUC score for Logistic Regression (BERT + NER)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC Score for Logistic Regression (BERT + NER): {roc_auc:.4f}")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve for Logistic Regression (BERT + NER)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random classification
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression (BERT + NER)')
plt.legend(loc='lower right')
plt.show()

# ===========================================
# Baseline (BERT Only) for Comparison
# ===========================================

print("Step 1: Converting train_features to NumPy")
train_np = train_features.numpy()
test_np = test_features.numpy()

print("Step 2: Applying PCA to train set")
pca_base = PCA(n_components=100)
X_train_base = pca_base.fit_transform(train_np)

print("Step 3: Applying PCA to test set")
X_test_base = pca_base.transform(test_np)

print("Step 4: Fitting Logistic Regression")
clf_base = LogisticRegression(max_iter=1000)
clf_base.fit(X_train_base, y_train)

print("Step 5: Predicting")
y_pred_base = clf_base.predict(X_test_base)

print("Step 6: Scoring and Evaluation")

# ================================
# Evaluation for BERT Only Model
# ================================

# Evaluate the baseline model (Logistic Regression on BERT-only features)
print("Evaluation: BERT Only (Baseline)")
print(f"Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_base):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_base):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_base):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_base, target_names=["REAL", "FAKE"]))

# =====================================================
# ROC-AUC Curve and Score for BERT-only Baseline Model
# =====================================================

# Get the predicted probabilities for the baseline model (BERT-only)
y_prob_base = clf_base.predict_proba(X_test_base)[:, 1]  # Probability for the 'fake' class (class 1)

# Compute ROC-AUC score for BERT-only model
roc_auc_base = roc_auc_score(y_test, y_prob_base)
print(f"\nROC-AUC Score for BERT Only (Baseline): {roc_auc_base:.4f}")

# Calculate ROC curve for BERT-only model
fpr_base, tpr_base, thresholds_base = roc_curve(y_test, y_prob_base)

# Plot ROC curve for BERT-only model
plt.figure(figsize=(8, 6))
plt.plot(fpr_base, tpr_base, color='green', label=f'ROC curve (area = {roc_auc_base:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random classification
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for BERT Only')
plt.legend(loc='lower right')
plt.show()