import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
import plotly.express as px

def load_df():
    data = load_iris()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df["Species"] = data.target
    df["Species_Name"] = df["Species"]
    for index, row in df.iterrows():
        species = int(row["Species"])
        df.loc[index, "Species_Name"] = data.target_names[species]

    return df, data.feature_names

def main():
    df, features = load_df()
    st.title("Iris Classification")
    method = st.sidebar.selectbox('Select the Clustering Method', ['K-Means', 'DBSCAN'])
    st.sidebar.subheader("hyperparameters")
    st.header(method)

    if method == 'K-Means':
        n_clusters = st.sidebar.slider('K', min_value=1, max_value=10, value=3, step=1)
        kmeans_func(df=df, features=features, n_clusters=n_clusters)
        st.sidebar.button("Classify")

    elif method == 'DBSCAN':
        eps = st.sidebar.number_input('eps', min_value=0.01,max_value=None, value=0.5,step=0.1)
        min_samples = st.sidebar.number_input("Min Samples", min_value=1, max_value=None, value=5, step=1)
        DBSCAN_func(df=df,features=features,eps=eps)
        st.sidebar.button("Classify")


def kmeans_func(df,features,n_clusters):
    X = df[features]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(X)
    predicted_clusters = kmeans.predict(X)
    df["Cluster"] = predicted_clusters
    plot_func(df, features)

def DBSCAN_func(df,features,eps=0.5, min_samples=5):
    X = df[features]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    predicted_clusters = dbscan.fit_predict(X)
    df['Cluster'] = predicted_clusters
    plot_func(df, features)

def plot_func(df, features):
    df = df.astype({"Cluster": "category"})
    ax = st.multiselect('Axis', features, max_selections=3)
    if len(ax) ==3:
        fig = px.scatter_3d(df,
                            x=ax[0],
                            y=ax[1],
                            z=ax[2],
                            color="Cluster",
                            height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Select 3 features.")

if __name__ == "__main__":
    main()
