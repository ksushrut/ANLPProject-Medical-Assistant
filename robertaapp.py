import streamlit as st
from robertafinal import load_model, predict

st.set_page_config(page_title="Disease Classification Chatbot")
st.title("Disease Classification Chatbot")

model, tokenizer = load_model()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

input_text = st.text_input("Describe your symptoms here")

if st.button("Submit"):
    if input_text:
        with st.spinner("Predicting..."):
            # Get prediction from the model
            disease_name, precautions = predict(model, tokenizer, input_text)
            st.write(f"The predicted disease is: **{disease_name}**")
            st.write("Precautions:")
            for precaution in precautions:
                st.write(f"- {precaution}")

        st.session_state.chat_history.append({"role": "user", "text": input_text})
        
        # Constructing the assistant's response with disease name and precautions in separate lines
        assistant_response = f"The patient might be suffering from {disease_name}.\n\nThe suggested precautions are:"
        for precaution in precautions:
            assistant_response += f"\n- {precaution}"
        
        # Adding the disclaimer at the end
        assistant_response += "\n\n*Please note that the chatbot's predictions may not always be accurate and should not substitute professional medical advice.*"
        
        st.session_state.chat_history.append({"role": "assistant", "text": assistant_response})

