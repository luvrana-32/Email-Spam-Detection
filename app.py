import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Email Spam Detection", page_icon="📧", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "email_messages.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model():
    data = load_data().dropna(subset=["message", "label"]).copy()
    data["message"] = data["message"].astype(str)
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    X = data["message"]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="spam", zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label="spam", zero_division=0),
        "f1": f1_score(y_test, predictions, pos_label="spam", zero_division=0),
        "matrix": confusion_matrix(y_test, predictions, labels=["ham", "spam"]),
        "test_size": len(y_test),
        "train_size": len(y_train),
    }
    return model, metrics, data

model, metrics, data = train_model()

st.title("📧 Email Spam Detection")
st.caption("College Mini Project | Machine Learning | Text Classification")
st.info("Dataset notice: This project starts with a small, synthetic educational sample dataset. It is for demonstrating the ML workflow, not for filtering real inboxes.")

tab1, tab2, tab3 = st.tabs(["Check an Email", "Model Performance", "Dataset"])

with tab1:
    st.subheader("Paste an email or SMS message")
    message = st.text_area("Message text", height=170, placeholder="Example: Congratulations! You have won a free prize...")
    if st.button("Detect Spam", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            classes = model.named_steps["classifier"].classes_
            spam_probability = float(probabilities[list(classes).index("spam")]) if "spam" in classes else 0.0
            if prediction == "spam":
                st.error("Result: Likely SPAM")
            else:
                st.success("Result: Likely NOT SPAM (Ham)")
            st.write(f"Model's estimated spam probability: **{spam_probability:.1%}**")
            st.caption("This is a model score, not a guarantee. Do not use it as the only basis for security decisions.")

with tab2:
    st.subheader("Evaluation on a held-out test split")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Precision (spam)", f"{metrics['precision']:.1%}")
    c3.metric("Recall (spam)", f"{metrics['recall']:.1%}")
    c4.metric("F1-score", f"{metrics['f1']:.1%}")
    st.write(f"Training examples: **{metrics['train_size']}** | Test examples: **{metrics['test_size']}**")
    st.write("Confusion matrix (rows = actual, columns = predicted; order: ham, spam)")
    st.dataframe(pd.DataFrame(metrics["matrix"], index=["Actual ham", "Actual spam"], columns=["Predicted ham", "Predicted spam"]))
    st.caption("Because the sample dataset is small and synthetic, these scores only demonstrate evaluation methods. They do not represent real-world performance.")

with tab3:
    st.subheader("Sample dataset preview")
    st.write(f"Rows: **{len(data)}** | Ham: **{(data.label == 'ham').sum()}** | Spam: **{(data.label == 'spam').sum()}**")
    st.dataframe(data.head(20), use_container_width=True)
    st.download_button("Download dataset CSV", data=DATA_PATH.read_bytes(), file_name="email_messages.csv", mime="text/csv")

with st.sidebar:
    st.header("About this project")
    st.write("**Algorithm:** Multinomial Naive Bayes")
    st.write("**Text features:** TF-IDF")
    st.write("**Task:** Binary text classification")
    st.write("**Labels:** ham = legitimate, spam = unwanted")
    st.divider()
    st.write("Built with Python, Pandas, Scikit-learn and Streamlit.")
