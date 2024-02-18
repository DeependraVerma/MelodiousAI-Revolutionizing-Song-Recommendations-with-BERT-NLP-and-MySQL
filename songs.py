from dotenv import load_dotenv
import streamlit as st
import os
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer, util


load_dotenv()

db_user = "root"
db_password = "root"
db_host = "localhost"
db_name = "song_recommender"

connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(connection_string)

def execute_sql_query(sql):
    with engine.connect() as connection:
        result = connection.execute(text(sql))
        rows = result.fetchall()
        return rows

st.set_page_config(page_title="Song Recommender")
st.header("Song Recommender")

song_input = st.text_input("Enter Song Name:", key="song_input")
submit = st.button("Search")

if submit:
    # Fetch song description based on the user input song name
    song_name = song_input
    song_query = f'''SELECT DISTINCT Description FROM songs WHERE `Name of the Song` = "{song_name}" LIMIT 30;'''
    song_description = execute_sql_query(song_query)
    
    if song_description:
        song_description = song_description[0][0]  # Extracting description from the result
        
        # Fetching all songs data
        songs_query = '''SELECT DISTINCT `Name of the Song`, Description FROM songs;'''
        songs_data = execute_sql_query(songs_query)
        
        # Extract song names and descriptions
        all_song_names = [row[0] for row in songs_data]
        all_song_descriptions = [row[1] for row in songs_data]
        
        # Load pre-trained BERT model
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Compute embeddings for user-selected song description and all other song descriptions
        user_embedding = model.encode(song_description, convert_to_tensor=True)
        song_embeddings = model.encode(all_song_descriptions, convert_to_tensor=True)
        
        # Compute cosine similarity between user-selected song embedding and all other song embeddings
        similarity_scores = util.pytorch_cos_sim(user_embedding, song_embeddings)
        
        # Convert similarity_scores to numpy array
        similarity_scores = similarity_scores.cpu().numpy()
        
        # Find top 5 songs with highest cosine similarity
        top_indices = similarity_scores.argsort()[0][-6:-1][::-1]  # Exclude the user-selected song itself
        top_song_names = [all_song_names[i] for i in top_indices]
        
        st.subheader(f"Top 5 Songs Similar to '{song_name}':")
        st.table(top_song_names)
    else:
        st.write("Song not found. Please enter a valid song name.")
