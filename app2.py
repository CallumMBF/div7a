import streamlit as st
import pandas as pd
import openai
import os

# Access the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
# OpenAI API Key
client = openai.OpenAI(api_key=openai_api_key)
# LLM validation logic
def validate_div7a_logic(data):
    prompt = f"""
    I am an expert in Division 7A calculations. Validate the following uploaded Excel data against the logic:
    {data}
    
    Checklist:
    - Step 1: Validate Binder Setup
    - Step 2: Validate Div7A Calculator Worksheet
    - Step 3: Validate Summary Worksheets
    - Step 4: Apply Division 7A Logic
    - Step 5: Verify Templates
    - Step 6: Review Uploaded Files
    - Step 7: Generate Final Report
    
    Identify any issues or inconsistencies, suggest corrections, and confirm if the files comply.
    """
    response = client.completions.create(
        model="gpt-4o",
        prompt=prompt,
        max_tokens=800
    )
    return response.choices[0].text.strip()

# App UI
st.title("Division 7A Compliance Checker")

st.sidebar.header("Upload Section")
uploaded_files = st.sidebar.file_uploader(
    "Upload your Division 7A Excel files here",
    accept_multiple_files=True,
    type=["xlsx"]
)

if uploaded_files:
    st.header("Uploaded Files")
    for file in uploaded_files:
        st.write(f"Processing file: {file.name}")
        try:
            # Load Excel file
            data = pd.ExcelFile(file)
            st.write("Sheets in file:", data.sheet_names)
            
            # Display content of the first sheet as preview
            sheet = data.parse(data.sheet_names[0])
            st.write(f"Preview of {file.name} - {data.sheet_names[0]}:")
            st.dataframe(sheet)
            
            # Convert to JSON-like structure for LLM
            json_data = sheet.to_dict()
            
            # Validate using LLM
            with st.spinner("Validating..."):
                validation_results = validate_div7a_logic(json_data)
            
            # Show results
            st.subheader(f"Validation Results for {file.name}")
            st.write(validation_results)
        
        except Exception as e:
            st.error(f"Failed to process {file.name}: {e}")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload mock Division 7A Excel files.
2. The app checks the uploaded files against Division 7A compliance rules.
3. Get feedback and suggestions for corrections or improvements.
""")
