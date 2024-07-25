import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging

load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def extract_questions(text):
    sentences = sent_tokenize(text)
    questions = [sentence.strip() for sentence in sentences if sentence.endswith('?')]
    return questions

def generate_responses(prompt_parts, model):
    try:
        response = model.generate_content(prompt_parts)
        logging.info(f"Response object: {response}")
        
        # Check if the response has the 'text' attribute
        if hasattr(response, 'text'):
            return response.text
        else:
            logging.error("Response object does not have a 'text' attribute")
            raise ValueError("Response object does not have a 'text' attribute")
    
    except Exception as e:
        logging.exception("Error in generate_responses")
        raise e

# Set Streamlit page configuration
st.set_page_config(
    page_title="FAQ Extraction Web App",
    page_icon=":books:",
    layout="wide",
)

# Custom CSS for UI enhancement
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    .header {
        color: #2e3b4e;
        font-weight: bold;
        font-size: 30px;
        text-align: center;
        padding: 20px 0;
    }
    .subheader {
        color: #4a6fa5;
        font-weight: bold;
        font-size: 24px;
        margin-top: 20px;
    }
    .main-container {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .button {
        background-color: #4a6fa5;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.markdown("<div class='header'>FAQ Extraction Web App</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)

            if 'Transcript' in df.columns:
                transcript = df['Transcript'].str.cat(sep='\n')

                cleaned_text = preprocess_text(transcript)

                # Initialize Google Gemini
                genai.configure(api_key=google_api_key)
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 1,
                    "max_output_tokens": 8192,
                }
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                ]

                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    safety_settings=safety_settings,
                    generation_config=generation_config,
                )

                topics = st.multiselect(
                    "Select Topics",
                    ["Ackumen Boiling Management (ABM)", "Cooling Tower", "User Management", "Connected Planning (CP)", "Process View (PV)", "Ackumen General", "Connected Lab", "Ackumen Orders", "Ackumen Data Entry (ADE)", "MCA"]
                )

                if topics:
                    responses = {}
                    for topic in topics:
                        prompt_parts = [cleaned_text, f"Retrieve the top 10 frequently asked questions (FAQs) as questions about {topic} from the given content, along with the respective frequency count of each question which is greater than 35. A question is a Frequently Asked Question only if it exceeds 35 in its occurrence in the text. I need only questions and not answers."]
                        try:
                            response = generate_responses(prompt_parts, model)
                            responses[topic] = response
                        except ValueError as ve:
                            st.error(f"ValueError: {str(ve)} for topic {topic}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred for topic {topic}: {str(e)}")

                    st.markdown("<div class='subheader'>Generated FAQs</div>", unsafe_allow_html=True)
                    for topic, response in responses.items():
                        st.markdown(f"### {topic}")
                        st.text(response)

                    if st.button("Create Excel File and Send Response to It"):
                        excel_writer = pd.ExcelWriter("output.xlsx", engine="xlsxwriter")

                        for topic, response in responses.items():
                            questions = response.split('\n')
                            df_responses = pd.DataFrame({"Question": questions, "Count": [None] * len(questions)})
                            df_responses.to_excel(excel_writer, sheet_name=topic, index=False)

                        excel_writer.close()

                        with open("output.xlsx", "rb") as file:
                            st.download_button(
                                label="Download Excel",
                                data=file.read(),
                                file_name="output.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                else:
                    st.info("Please select at least one topic to proceed.")

            else:
                st.error("The uploaded Excel file does not contain a 'Transcript' column.")
        else:
            st.info("Please upload an Excel file to proceed.")

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
