import streamlit as st
import numpy as np
import pandas as pd
import json
import ollama
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score

#Project title
st.title("Segment Studio")

def prepare(df):
    """Keep the numeric columns, fill gaps with the median, and scale to mean 0 and std 1."""
    numeric = df.select_dtypes("number")
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return prep.fit_transform(numeric)

def compute_scores(X, metric, min_k, max_k):
    """Fit K-Means for every k in the range and return a table of k and the chosen score (WCSS or silhouette)."""
    k_values = list(range(min_k, max_k + 1))
    scores = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(X)
        if metric == "WCSS":
            scores.append(model.inertia_)
        else:
            scores.append(silhouette_score(X, model.labels_))
    return pd.DataFrame({"k": k_values, "score": scores})

def pick_k_from_wcss(table):
    """Ask Llama for the elbow k from a WCSS table; fall back to the biggest bend if the answer is unusable."""
    if len(table) < 3:
        return int(table["k"].iloc[0]), "range too small"
    prompt = (
        "You are choosing the number of clusters for K-Means using the elbow method. "
        "WCSS always falls as k grows. The best k is where the drop in WCSS suddenly gets much smaller, the bend in the curve.\n"
        f"Here is the table:\n{table.to_string(index=False)}\n\n"
        'Answer only with JSON: {"k": one whole number from the k column}'
    )
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        k = int(json.loads(response["message"]["content"])["k"])
        if k in table["k"].values:
            return k, "Llama"
    except (ConnectionError, ollama.ResponseError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass
    drops = -np.diff(table["score"].values)
    bend = drops[:-1] - drops[1:]
    return int(table["k"].values[bend.argmax() + 1]), "bend rule"

#Upload file (Only works with .csv)
uploaded_file = st.file_uploader("Choose a CSV file:", type="csv")

#Making the CSV into a DF if the upload worked
if uploaded_file is not None:
    file_key = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("file_key") != file_key:
        for key in ["score_table", "cluster_table", "labels", "named_table", "suggested_k", "k_source", "keep"]:
            st.session_state.pop(key, None)
    st.session_state["file_key"] = file_key
    df = pd.read_csv(uploaded_file)
    st.subheader("The Table")
    st.dataframe(df)
    st.caption(f"{df.shape[0]} rows by {df.shape[1]} columns")

    #Stop the process if the file doesn't have number for clusters and define 'X' for the next steps if it's valid
    if df.select_dtypes("number").shape[1] == 0:
        st.error("The file has no numeric columns to cluster on.")
        st.stop()
    X = prepare(df)

    #Making the max/min K slider
    col1, col2 = st.columns(2)
    with col1:
        min_k = st.slider("Min K", 2, 20, 2)
    with col2:
        max_k = st.slider("Max K", 2, 20, 10)

    #Making a run button for either WCSS or Silhouette
    metric = st.radio("Metric", ["WCSS", "Silhouette"], horizontal=True)
    run_button = st.button(f"Run {metric}")
    

    if run_button:
        if min_k >= max_k:
            st.error("Min K must be smaller than max K.")
        elif metric == "Silhouette" and max_k >= len(X):
            st.error("Max K must be smaller than the number of rows.")
        else:
            new_table = compute_scores(X, metric, min_k, max_k)
            st.session_state["score_table"] = new_table
            st.session_state["metric"] = metric
            st.session_state.pop("suggested_k", None)
            st.session_state.pop("k_source", None)

    #Plotting the Elbow Plot
    if "score_table" in st.session_state:
        metric_used = st.session_state["metric"]
        st.subheader(metric_used)
        st.dataframe(st.session_state["score_table"])
        table = st.session_state["score_table"]
        fig, ax = plt.subplots()
        ax.plot(table["k"], table["score"], marker= "o")
        ax.set_xticks(list(table["k"]))
        ax.set_xlabel("K")
        ax.set_ylabel("WCSS (inertia)" if metric_used == "WCSS" else "Silhouette score")
        ax.set_title("Elbow Plot" if metric_used == "WCSS" else "Silhouette Plot")
        st.pyplot(fig)
        if st.button("Pick K automatically"):
            if metric_used == "Silhouette":
                st.session_state["suggested_k"] = int(table.loc[table["score"].idxmax(), "k"])
                st.session_state["k_source"] = "highest silhouette"
            else:
                with st.spinner("Asking the model..."):
                    k, source = pick_k_from_wcss(table)
                st.session_state["suggested_k"] = k
                st.session_state["k_source"] = source
        if "suggested_k" in st.session_state:
            st.success(f"Suggested K: {st.session_state['suggested_k']} ({st.session_state['k_source']})")

    #Choose K and create clusters
    clus_k = st.slider("Select K", 2, 20, st.session_state.get("suggested_k", 5))
    remove_outliers = st.checkbox("Remove outliers before clustering")
    outlier_std = st.slider("Outlier threshold (standard deviations)", 2.0, 4.0, 3.0, 0.5, disabled=not remove_outliers)
    run_clusters = st.button("Create Clusters")
    if run_clusters:
        model = KMeans(n_clusters=clus_k, n_init=10, random_state=42).fit(X)
        keep = np.ones(len(X), dtype=bool)
        if remove_outliers:
            dist = model.transform(X)[np.arange(len(X)), model.labels_]
            keep = dist <= dist.mean() + outlier_std * dist.std()
            model = KMeans(n_clusters=clus_k, n_init=10, random_state=42).fit(X[keep])
        labels = model.labels_
        clus_count = pd.Series(labels).value_counts().sort_index()
        st.session_state["cluster_table"] = pd.DataFrame({"Cluster ID": clus_count.index, "Count": clus_count}).set_index("Cluster ID")
        st.session_state["labels"] = labels
        st.session_state["keep"] = keep
        st.session_state.pop("named_table", None)
    if "cluster_table" in st.session_state:
        st.subheader("Choose K and create clusters")
        st.dataframe(st.session_state["cluster_table"])
        st.caption(f"{int((~st.session_state['keep']).sum())} rows removed as outliers")

    #Prepearing a DF and parameters for our LLM to read and describe ONLY after Create Clusters was used.
    if "labels" in st.session_state:
        if st.button("Generate names with LLM"):
            df_to_llm = df[st.session_state["keep"]].copy()
            labels = st.session_state["labels"]
            num_cols = df.select_dtypes("number").columns.tolist()
            text_cols = df.select_dtypes(exclude="number").columns.tolist()
            df_to_llm["cluster"] = labels
            summary = df_to_llm.groupby("cluster")[num_cols].mean().round(2)
            counts = df_to_llm.groupby("cluster").size()
            modes = df_to_llm.groupby("cluster")[text_cols].agg(lambda s: s.mode().get(0, "unknown"))
            profile = summary.join(modes)
            profile["count"] = counts
            overview = profile.to_string()
            names = []
            descriptions = []
            failed = False
            with st.spinner("Generating names... this can take a few minutes."):
                try:
                    for cid in profile.index:
                        prompt = (
                            "You are naming groups found by K-Means clustering in a table of data. "
                            "Use ONLY the numbers in the table. Do not use outside knowledge and do not mention species or real-world categories.\n"
                            f"Here are the averages and most common values of every group:\n{overview}\n\n"
                            f"Give group {cid} a name that clearly differs from the other groups, "
                            "and describe it in one sentence.\n"
                            'Answer only with JSON: {"name": "2 to 4 words", "description": "one sentence"}'
                            )
                        response = ollama.chat(
                        model="llama3.2",
                        messages=[{"role": "user", "content": prompt}],
                        format="json",
                        options={"temperature": 0},
                        )
                        try:
                            reply = json.loads(response["message"]["content"])
                            names.append(reply["name"])
                            descriptions.append(reply["description"])
                        except (json.JSONDecodeError, KeyError):
                            names.append(f"Cluster {cid}")
                            descriptions.append("No description generated.")
                    original = names.copy()
                    names = [f"{n} (group {c})" if original.count(n) > 1 else n for n, c in zip(original, profile.index)]
                    named_table = pd.DataFrame({
                    "Cluster ID": profile.index,
                    "Count": profile["count"].values,
                    "Name": names,
                    "Description": descriptions,
                    }).set_index("Cluster ID")
                except (ConnectionError, ollama.ResponseError):
                    failed = True
            if failed:
                st.error("Couldn't reach the model. Make sure Ollama is running and `llama3.2` is pulled.")
                st.stop()
    if "named_table" in st.session_state:
        st.subheader("Clusters defined by LLM (LLaMa)")
        st.session_state["named_table"] = named_table
        st.dataframe(st.session_state["named_table"])
        names_by_id = st.session_state["named_table"]["Name"].to_dict()
        export_df = df.copy()
        export_df["cluster_name"] = "outlier"
        export_df.loc[st.session_state["keep"], "cluster_name"] = pd.Series(st.session_state["labels"]).map(names_by_id).values
        file_name = uploaded_file.name.rsplit(".", 1)[0] + "_clustered.csv"
        st.download_button("Download clustered CSV", data=export_df.to_csv(index=False), file_name=file_name, mime="text/csv")
