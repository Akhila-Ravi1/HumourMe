import streamlit as st
from predict import PredictJoke

@st.cache_resource()
def load_predict_joke_class():
    return PredictJoke()

predict = load_predict_joke_class()  # Load the class only once

st.title("HumourMe")

st.write("Hi there! Welcome to HumourMe!")

st.write("This is a humour detection app that uses a LSTM model to predict whether a sentence is humorous or not. Try your jokes here!")


# Get user input
user_input = st.text_input("Enter your joke here:")

if user_input:
    # Get the word2vec representation of the user input
    predicted_class = predict.predict_joke(user_input)

    if predicted_class == "Humorous":
        st.subheader("Haha! That's funny! :joy:")
    else:
        st.subheader("Hmm... I don't think that's funny. :neutral_face:")

else:
    st.write("Enter your joke in the text box above!")

