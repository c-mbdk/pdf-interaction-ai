import streamlit as st

# from dotenv import load_dotenv
# load_dotenv()

from utils import process_file, process_user_input


def main():
    
    # Settings
    st.set_page_config(page_title="PDF Chatbot", 
                       page_icon=":page_facing_up:")

    st.header('PDF Chatbot')
    user_query = st.text_input("Ask a question about your document: ")

    if user_query:
        process_user_input(user_query)

    with st.sidebar:
        st.subheader("Uploaded document")
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
