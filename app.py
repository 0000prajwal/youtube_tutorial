import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎭",
    layout="wide",
)

# ── NLTK downloads ────────────────────────────────────────────────────────────
@st.cache_resource
def load_nltk():
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words", quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_nltk()

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    records = {}
    for _, row in df.iterrows():
        scores = sia.polarity_scores(str(row["Text"]))
        records[row["Id"]] = {f"vader_{k}": v for k, v in scores.items()}
    vader_df = pd.DataFrame(records).T.reset_index().rename(columns={"index": "Id"})
    return vader_df.merge(df, how="left")


def load_roberta():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.special import softmax
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model, softmax


def roberta_score(text, tokenizer, model, softmax):
    from scipy.special import softmax as sf
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    out = model(**enc)
    scores = sf(out[0][0].detach().numpy())
    return {"roberta_neg": float(scores[0]),
            "roberta_neu": float(scores[1]),
            "roberta_pos": float(scores[2])}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
st.sidebar.title("⚙️ Settings")

uploaded = st.sidebar.file_uploader(
    "Upload Reviews CSV (needs `Id`, `Text`, `Score` columns)",
    type=["csv"],
)
max_rows = st.sidebar.slider("Max rows to analyse", 50, 2000, 500, step=50)
use_roberta = st.sidebar.toggle("Enable RoBERTa model (slow, needs HuggingFace)", value=False)
use_pipeline = st.sidebar.toggle("Enable 🤗 Pipeline predictor", value=False)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(file_bytes, n):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.head(n)

st.title("🎭 Sentiment Analysis Dashboard")
st.markdown(
    "Analyse Amazon reviews with **VADER**, **RoBERTa**, and the **🤗 Transformers Pipeline**."
    " Upload your CSV in the sidebar or explore the demo below."
)

if uploaded:
    df = load_csv(uploaded.read(), max_rows)
else:
    st.info("No file uploaded — showing a synthetic demo dataset.")
    rng = np.random.default_rng(42)
    texts = [
        "I absolutely love this product! Best purchase ever.",
        "Terrible quality, broke after one day. Waste of money.",
        "It's okay, nothing special. Works as described.",
        "Amazing! Exceeded all my expectations.",
        "Would not recommend. Very disappointed.",
        "Great value for the price. Would buy again.",
        "Not what I expected. Misleading description.",
        "Five stars! My whole family loves it.",
        "Mediocre at best. Could be better.",
        "Horrible experience. Never buying again.",
    ] * 50
    scores = rng.integers(1, 6, len(texts))
    df = pd.DataFrame({"Id": range(1, len(texts)+1), "Text": texts[:len(texts)], "Score": scores})
    df = df.head(max_rows)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA", "😐 VADER", "🤖 RoBERTa", "🔀 Compare", "✍️ Try It"
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 – EDA
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(df))
    col2.metric("Avg Star Rating", f"{df['Score'].mean():.2f} ⭐")
    col3.metric("Unique Scores", df["Score"].nunique())

    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Score Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    df["Score"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color="#4C72B0", edgecolor="white"
    )
    ax.set_title("Count of Reviews by Stars")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Review Text Length")
    df["text_len"] = df["Text"].astype(str).str.len()
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    sns.histplot(df["text_len"], bins=40, kde=True, ax=ax2, color="#DD8452")
    ax2.set_xlabel("Character Count")
    ax2.set_title("Distribution of Review Length")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 – VADER
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("VADER Sentiment Scoring")
    st.markdown(
        "VADER uses a **bag-of-words** approach: stop-words are removed, "
        "each word is scored, and scores are combined into a compound value."
    )

    with st.spinner("Running VADER on dataset…"):
        vaders = run_vader(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Compound Score by Star Rating")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=vaders, x="Score", y="vader_compound", ax=ax, palette="coolwarm")
        ax.set_title("Compound Score vs Amazon Stars")
        ax.set_xlabel("Star Rating")
        ax.set_ylabel("Compound Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Pos / Neu / Neg by Star Rating")
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        for ax, col, title, color in zip(
            axs,
            ["vader_pos", "vader_neu", "vader_neg"],
            ["Positive", "Neutral", "Negative"],
            ["#2ca02c", "#7f7f7f", "#d62728"],
        ):
            sns.barplot(data=vaders, x="Score", y=col, ax=ax, color=color)
            ax.set_title(title)
            ax.set_xlabel("Stars")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Interesting Examples")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Highest positive score in 1-star reviews:**")
        top = vaders.query("Score == 1").sort_values("vader_pos", ascending=False)
        if not top.empty:
            st.info(str(top["Text"].values[0])[:500])
    with c2:
        st.markdown("**Highest negative score in 5-star reviews:**")
        top = vaders.query("Score == 5").sort_values("vader_neg", ascending=False)
        if not top.empty:
            st.warning(str(top["Text"].values[0])[:500])

    st.subheader("Full VADER Results")
    st.dataframe(
        vaders[["Id", "Score", "vader_neg", "vader_neu", "vader_pos", "vader_compound", "Text"]]
        .head(50),
        use_container_width=True,
    )

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 – RoBERTa
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("RoBERTa Pretrained Model")
    st.markdown(
        "RoBERTa is a transformer model trained on large corpora. "
        "Unlike VADER it accounts for **context** between words."
    )

    if not use_roberta:
        st.warning(
            "RoBERTa is disabled. Enable it in the sidebar (requires `transformers` + internet)."
        )
    else:
        try:
            with st.spinner("Loading RoBERTa model (first run may take a minute)…"):
                tokenizer, model, softmax_fn = load_roberta()

            sample_size = min(100, len(df))
            sample_df = df.head(sample_size).copy()
            records = {}
            prog = st.progress(0, text="Scoring with RoBERTa…")
            for i, (_, row) in enumerate(sample_df.iterrows()):
                try:
                    records[row["Id"]] = roberta_score(str(row["Text"]), tokenizer, model, softmax_fn)
                except Exception:
                    pass
                prog.progress((i + 1) / sample_size)
            prog.empty()

            rob_df = pd.DataFrame(records).T.reset_index().rename(columns={"index": "Id"})
            rob_df = rob_df.merge(sample_df, how="left")

            st.subheader("RoBERTa Scores by Star Rating")
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            for ax, col, title, color in zip(
                axs,
                ["roberta_pos", "roberta_neu", "roberta_neg"],
                ["Positive", "Neutral", "Negative"],
                ["#2ca02c", "#7f7f7f", "#d62728"],
            ):
                sns.barplot(data=rob_df, x="Score", y=col, ax=ax, color=color)
                ax.set_title(f"RoBERTa {title}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.subheader("Full RoBERTa Results")
            st.dataframe(
                rob_df[["Id", "Score", "roberta_neg", "roberta_neu", "roberta_pos", "Text"]].head(50),
                use_container_width=True,
            )

            st.session_state["rob_df"] = rob_df

        except ImportError:
            st.error("Install `transformers` and `torch` to use RoBERTa.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 – Compare
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Model Comparison (VADER vs RoBERTa)")

    if not use_roberta or "rob_df" not in st.session_state:
        st.warning("Enable RoBERTa in the sidebar and visit the RoBERTa tab first.")
    else:
        rob_df = st.session_state["rob_df"]
        vaders_cmp = run_vader(rob_df[["Id", "Text", "Score"]])
        merged = vaders_cmp.merge(
            rob_df[["Id", "roberta_neg", "roberta_neu", "roberta_pos"]],
            on="Id",
        )

        st.subheader("Pairplot: VADER vs RoBERTa")
        with st.spinner("Drawing pairplot…"):
            fig = sns.pairplot(
                data=merged,
                vars=["vader_neg", "vader_neu", "vader_pos",
                      "roberta_neg", "roberta_neu", "roberta_pos"],
                hue="Score",
                palette="tab10",
                plot_kws={"alpha": 0.5, "s": 20},
            )
            st.pyplot(fig)
            plt.close()

# ────────────────────────────────────────────────────────────────────────────
# TAB 5 – Try It
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("✍️ Analyse Your Own Text")
    user_text = st.text_area(
        "Enter any text to analyse:",
        value="I absolutely love this product! Best purchase ever.",
        height=120,
    )

    if st.button("🔍 Analyse", use_container_width=True):
        # VADER
        vader_scores = sia.polarity_scores(user_text)
        st.subheader("VADER Scores")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positive", f"{vader_scores['pos']:.3f}")
        c2.metric("Neutral", f"{vader_scores['neu']:.3f}")
        c3.metric("Negative", f"{vader_scores['neg']:.3f}")
        c4.metric("Compound", f"{vader_scores['compound']:.3f}")

        compound = vader_scores["compound"]
        if compound >= 0.05:
            st.success("Overall VADER Sentiment: **Positive** 😊")
        elif compound <= -0.05:
            st.error("Overall VADER Sentiment: **Negative** 😞")
        else:
            st.info("Overall VADER Sentiment: **Neutral** 😐")

        # RoBERTa (if enabled)
        if use_roberta:
            try:
                with st.spinner("Running RoBERTa…"):
                    if "roberta_tokenizer" not in st.session_state:
                        tok, mdl, sfm = load_roberta()
                        st.session_state["roberta_tokenizer"] = tok
                        st.session_state["roberta_model"] = mdl
                        st.session_state["roberta_softmax"] = sfm
                    rob = roberta_score(
                        user_text,
                        st.session_state["roberta_tokenizer"],
                        st.session_state["roberta_model"],
                        st.session_state["roberta_softmax"],
                    )
                st.subheader("RoBERTa Scores")
                d1, d2, d3 = st.columns(3)
                d1.metric("Positive", f"{rob['roberta_pos']:.3f}")
                d2.metric("Neutral", f"{rob['roberta_neu']:.3f}")
                d3.metric("Negative", f"{rob['roberta_neg']:.3f}")
            except ImportError:
                st.error("Install `transformers` and `torch` for RoBERTa.")

        # Pipeline (if enabled)
        if use_pipeline:
            try:
                from transformers import pipeline as hf_pipeline
                with st.spinner("Running 🤗 Pipeline…"):
                    pipe = hf_pipeline("sentiment-analysis")
                    result = pipe(user_text[:512])[0]
                st.subheader("🤗 Pipeline Result")
                label = result["label"]
                score = result["score"]
                if label == "POSITIVE":
                    st.success(f"Label: {label} — Confidence: {score:.2%}")
                else:
                    st.error(f"Label: {label} — Confidence: {score:.2%}")
            except ImportError:
                st.error("Install `transformers` for the pipeline predictor.")

        # NLTK tokens
        with st.expander("🔬 NLTK Tokenisation & POS Tags"):
            tokens = nltk.word_tokenize(user_text)
            tagged = nltk.pos_tag(tokens)
            st.write("**Tokens:**", tokens[:20])
            st.write("**POS Tags:**", tagged[:20])

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Streamlit · VADER (NLTK) · RoBERTa (HuggingFace) · "
    "Based on the Sentiment Analysis Python Tutorial."
)
