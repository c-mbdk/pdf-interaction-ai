import streamlit as st
from PIL import Image

# from dotenv import load_dotenv
# load_dotenv()

from utils import process_file, process_user_input


def main():
    
    # Settings
    st.set_page_config(
        layout="wide",
        page_title="PDF Chatbot", 
        page_icon=":page_facing_up:"
    )

    logo_url = 'css/images/magnify-doc.png'
    col1, col2 = st.columns([0.12, 2])
    image = Image.open(logo_url)
    with col1:
        st.image(image, width=75)
        
    with col2:
        st.title('PDF Chatbot')

    # Initialise session state to store chat messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept input from user - document queries
    user_query = st.chat_input(
            "Ask a question about your document", key="user_query"
    )

    if user_query:
        process_user_input(user_query)

    # Upload document widget
    with st.sidebar:
        st.subheader("PDF Upload")
        pdf_doc = st.file_uploader(
            "Upload your PDF here and click on 'Process'",
            type=['pdf']
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                process_file(pdf_doc)
                st.success("Processing Complete")

if __name__ == '__main__':
    main()
