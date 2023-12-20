import pickle
import streamlit as st 
import numpy as np
import sklearn

st.header('Books Recommender System')
try:
    with open('artifacts/model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print("Error loading the pickled model:", e)

try:
    with open('artifacts/books_name.pkl', 'rb') as file:
        books_name = pickle.load(file)
except Exception as e:
    print("Error loading 'books_name.pkl':", e)

try:
    with open('artifacts/final_rating.pkl', 'rb') as file:
        final_rating = pickle.load(file)
except Exception as e:
    print("Error loading 'final_rating.pkl':", e)

try:
    with open('artifacts/book_pivot.pkl', 'rb') as file:
        book_pivot = pickle.load(file)
except Exception as e:
    print("Error loading 'book_pivot.pkl':", e)

def fetch_poster(suggestion):
    #Getting the name, index and image url of the book
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]:
        ids = np.where(final_rating['title']== name)[0][0]
        ids_index.append(ids)

    for ids in ids_index:
        url = final_rating.iloc[ids]['img_url']
        poster_url.append(url)
    return poster_url

def recommend_book(book_name):
    book_list =[]
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance,suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors = 6)
    
    poster_url = fetch_poster(suggestion)
    
    for i in range (len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            book_list.append(j)
    return book_list,poster_url


selected_books = st.selectbox('Type or select a book', books_name)

if st.button('Show Recommendation'):
    recommendation_books, poster_url = recommend_book(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:
        # Didn't start with index 0 because it returns the entered book's name
        st.text(recommendation_books[1])
        try:
            st.image(poster_url[1])
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with col2:
        st.text(recommendation_books[2])
        st.image(poster_url[2])

    with col3:
        st.text(recommendation_books[3])
        st.image(poster_url[3])

    with col4:
        st.text(recommendation_books[4])
        st.image(poster_url[4])

    with col5:
        st.text(recommendation_books[5])
        st.image(poster_url[5])

