import os
import pandas as pd
from openai import OpenAI
import streamlit as st
import json

class AdvancedLoanDetector:
    def __init__(self, api_key=None):
        """
        Initialize the LLM-powered loan detector
        
        :param api_key: OpenAI API key (optional, can use environment variable)
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        # Comprehensive context for loan detection
        self.loan_detection_prompt = """
        You are a financial expert analyzing trial balance account entries to identify loans.

        Loan Identification Criteria:
        1. Direct Indicators:
        - Explicit mention of loan, lending, borrowing
        - Financial instruments like notes receivable
        - Intercompany or director loans
        - Financing arrangements

        2. Contextual Indicators:
        - Interest-bearing assets
        - Repayment schedules
        - Installment-based receivables
        - Credit facilities
        - Deferred payment structures

        3. Characteristic Markers:
        - Principal amount mentioned
        - Secured or unsecured loan references
        - Long-term or short-term financial arrangements

        Analyze the following account entry and determine:
        A. Is this a loan?
        B. What specific characteristics suggest it is a loan?
        C. Provide a confidence score (0-100%)

        Account Entry: "{account_entry}"
        """

    def classify_loan(self, account_entry):
        """
        Use LLM to classify if an account is a loan
        
        :param account_entry: Account name or description
        :return: Dictionary with loan classification details
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Choose an appropriate model
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a financial classification assistant."
                    },
                    {
                        "role": "user", 
                        "content": self.loan_detection_prompt.format(
                            account_entry=account_entry
                        )
                    }
                ],
                max_tokens=300,
                temperature=0.2  # Low temperature for more consistent results
            )
            
            # Parse the response
            result = json.loads(
                response.choices[0].message.content
            )
            
            return {
                'account_entry': account_entry,
                'is_loan': result.get('is_loan', False),
                'confidence': result.get('confidence', 0),
                'loan_characteristics': result.get('loan_characteristics', [])
            }
        
        except Exception as e:
            st.error(f"LLM Analysis Error: {e}")
            return {
                'account_entry': account_entry,
                'is_loan': False,
                'confidence': 0,
                'loan_characteristics': []
            }

def main():
    st.title("Advanced LLM Loan Detector")
    
    # API Key Input
    # api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    api_key = os.getenv('OPENAI_API_KEY')
    # File Upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Trial Balance", 
        type=['xlsx', 'xls']
    )
    
    # Confidence Threshold
    confidence_threshold = st.sidebar.slider(
        "Loan Confidence Threshold", 
        min_value=0, 
        max_value=100, 
        value=50
    )
    
    if uploaded_file and api_key:
        # Read Trial Balance
        trial_balance_df = pd.read_excel(uploaded_file)
        
        # Initialize Loan Detector
        loan_detector = AdvancedLoanDetector(api_key)
        
        # Prepare for Analysis
        st.subheader("Loan Detection Analysis")
        progress_bar = st.progress(0)
        
        # Store Results
        loan_results = []
        
        # Analyze Each Account Entry
        for index, row in trial_balance_df.iterrows():
            # Update Progress
            progress_bar.progress(
                int((index + 1) / len(trial_balance_df) * 100)
            )
            
            # Combine relevant columns for analysis
            account_entry = " ".join(str(val) for val in row if pd.notna(val))
            
            # Classify Loan
            loan_classification = loan_detector.classify_loan(account_entry)
            
            # Filter by Confidence Threshold
            if (loan_classification['is_loan'] and 
                loan_classification['confidence'] >= confidence_threshold):
                loan_results.append(loan_classification)
        
        # Display Results
        if loan_results:
            results_df = pd.DataFrame(loan_results)
            st.dataframe(results_df)
            
            # Optional: Download Results
            st.download_button(
                label="Download Loan Detection Results",
                data=results_df.to_csv(index=False),
                file_name="loan_detection_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("No loans detected above the confidence threshold.")

if __name__ == "__main__":
    main()