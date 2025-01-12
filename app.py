import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Header aplikasi
st.title("IMDb Movie Recommender System")
st.write("Aplikasi ini memberikan rekomendasi berdasarkan Age Rating dan IMDb Rating.")

# Upload dataset
uploaded_file = st.file_uploader("Upload file dataset IMDb (CSV)", type=["csv"])

if uploaded_file is not None:
    # Baca dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset yang diunggah:")
    st.dataframe(df.head())

    # Persiapan data untuk Surprise
    df_surprise = df.rename(columns={
        "Age Rating": "user_id", 
        "Title": "item_id", 
        "IMDb Rating": "rating"
    })

    # Validasi kolom yang dibutuhkan
    required_columns = ["user_id", "item_id", "rating"]
    if all(col in df_surprise.columns for col in required_columns):
        reader = Reader(rating_scale=(df_surprise["rating"].min(), df_surprise["rating"].max()))
        data = Dataset.load_from_df(df_surprise[required_columns], reader)

        # Split data dan melatih model
        trainset, testset = train_test_split(data, test_size=0.25)
        sim_options = {
            "name": "cosine",
            "user_based": True
        }
        algo = KNNBasic(sim_options=sim_options)
        algo.fit(trainset)

        # Evaluasi model
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions)
        st.write(f"Model berhasil dilatih. RMSE: {rmse:.4f}")

        # Form untuk prediksi
        user_id = st.text_input("Masukkan Age Rating (contoh: PG-13)")
        item_id = st.text_input("Masukkan Judul Film (contoh: The Dark Knight)")

        if st.button("Prediksi Rating"):
            if user_id and item_id:
                pred = algo.predict(uid=user_id, iid=item_id)
                st.write(f"Prediksi rating untuk Age Rating '{user_id}' pada film '{item_id}': {pred.est:.2f}")
            else:
                st.error("Mohon masukkan Age Rating dan Judul Film.")
    else:
        st.error("Dataset harus memiliki kolom: 'user_id', 'item_id', dan 'rating'.")
else:
    st.info("Silakan upload file dataset.")

