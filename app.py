import pickle
import streamlit as st
import numpy as np


st.header('Book Recommender System Using Machine Learning')
try:
    with open('artifacts/model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print("Error loading the pickled model:", e)
try:
    with open('artifacts/books_name.pkl', 'rb') as file:
        books_name = pickle.load(file)
except FileNotFoundError:
    print("Books data file not found. Please ensure the file exists.")
    # Here you might want to assign some default or empty value to 'books_name'
    books_name = []
except Exception as e:
    print("Error loading 'books_name.pkl':", e)
    # Handle other specific exceptions as needed

try:
    with open('artifacts/final_rating.pkl', 'rb') as file:
        final_rating = pickle.load(file)
except FileNotFoundError:
    print("Rating data file not found. Please ensure the file exists.")
    final_rating = []
except Exception as e:
    print("Error loading 'final_rating.pkl':", e)
    # Handle other specific exceptions as needed

try:
    with open('artifacts/book_pivot.pkl', 'rb') as file:
        book_pivot = pickle.load(file)
except FileNotFoundError:
    print("Book pivot data file not found. Please ensure the file exists.")
    book_pivot = []
except Exception as e:
    print("Error loading 'book_pivot.pkl':", e)
    # Handle other specific exceptions as needed


def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]: 
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url



def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                books_list.append(j)
    return books_list , poster_url       



selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    books_name
)

if st.button('Show Recommendation'):
    recommended_books,poster_url = recommend_book(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_books[1])
        st.image(poster_url[1])
    with col2:
        st.text(recommended_books[2])
        st.image(poster_url[2])

    with col3:
        st.text(recommended_books[3])
        st.image(poster_url[3])
    with col4:
        st.text(recommended_books[4])
        st.image(poster_url[4])
    with col5:
        st.text(recommended_books[5])
        st.image(poster_url[5])
