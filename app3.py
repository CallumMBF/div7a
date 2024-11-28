import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime, timedelta

class LoanDetector:
    def __init__(self):
        # Comprehensive list of loan-related keywords
        self.loan_keywords = [
            # Direct indicators
            'loan', 'lending', 'credit', 'debt', 'receivable', 
            
            # Financial institution references
            'bank', 'finance', 'credit union', 'mortgage', 
            'note payable', 'notes receivable',
            
            # Specific loan types
            'term loan', 'line of credit', 'revolving credit', 
            'installment', 'bridge loan', 'commercial loan', 
            'personal loan',
            
            # Interest-related terms
            'interest bearing', 'interest accrual', 
            'principal', 'amortization',
            
            # Contextual clues
            'borrowed funds', 'external financing', 
            'capital advance', 'financial liability'
        ]

    def detect_potential_loans(self, trial_balance_df):
        """
        Detect potential loans in the trial balance
        
        :param trial_balance_df: DataFrame containing trial balance
        :return: DataFrame of potential loan assets
        """
        # Ensure the DataFrame is converted to string for easier searching
        potential_loans = trial_balance_df[
            trial_balance_df['Account Name'].str.lower().apply(
                lambda x: any(keyword in str(x).lower() for keyword in self.loan_keywords)
            )
        ]
        
        return potential_loans

    def analyze_loan_file(self, loan_file_path, financial_year_end):
        """
        Analyze loan file to determine loan details
        
        :param loan_file_path: Path to the loan Excel file
        :param financial_year_end: End date of the financial year
        :return: Dictionary of loan analysis details
        """
        try:
            # Read the Excel file
            df = pd.read_excel(loan_file_path)
            
            # Try to find the first transaction date
            # This might need adjustment based on the specific Excel file structure
            date_columns = df.select_dtypes(include=['datetime64']).columns
            
            if len(date_columns) > 0:
                first_transaction_date = df[date_columns[0]].min()
                
                # Check if loan is older than 12 months
                is_older_than_12_months = (financial_year_end - first_transaction_date).days > 365
                
                return {
                    'first_transaction_date': first_transaction_date,
                    'is_older_than_12_months': is_older_than_12_months
                }
            
            return None
        
        except Exception as e:
            st.error(f"Error analyzing loan file: {e}")
            return None

def main():
    st.title("Loan Detection and Analysis Tool")
    
    # Sidebar for file uploads
    st.sidebar.header("Upload Files")
    
    # Trial Balance File Upload
    trial_balance_file = st.sidebar.file_uploader(
        "Upload Trial Balance Excel", 
        type=['xlsx', 'xls']
    )
    
    # Financial Year End Date
    financial_year_end = st.sidebar.date_input(
        "Select Financial Year End Date",
        value=datetime.now()
    )
    
    # Loan Files Directory
    loan_files_directory = st.sidebar.text_input(
        "Path to Loan Files Directory",
        value=""
    )
    
    # Initialize LoanDetector
    loan_detector = LoanDetector()
    
    # Process Files when Trial Balance is uploaded
    if trial_balance_file is not None:
        # Read Trial Balance
        trial_balance_df = pd.read_excel(trial_balance_file)
        
        # Detect Potential Loans
        potential_loans = loan_detector.detect_potential_loans(trial_balance_df)
        
        # Display Potential Loans
        st.subheader("Potential Loans Detected")
        st.dataframe(potential_loans)
        
        # Loan Analysis Section
        if loan_files_directory and os.path.exists(loan_files_directory):
            st.subheader("Loan File Analysis")
            
            # Container to store loan analysis results
            loan_analysis_results = []
            
            # Iterate through potential loans
            for _, loan in potential_loans.iterrows():
                # Try to find matching loan file
                loan_filename = f"{loan['Account Name']}*.xlsx"
                matching_files = [
                    f for f in os.listdir(loan_files_directory) 
                    if re.search(loan_filename, f, re.IGNORECASE)
                ]
                
                if matching_files:
                    loan_file_path = os.path.join(loan_files_directory, matching_files[0])
                    
                    # Analyze Loan File
                    loan_analysis = loan_detector.analyze_loan_file(
                        loan_file_path, 
                        financial_year_end
                    )
                    
                    if loan_analysis:
                        loan_analysis_results.append({
                            'Account Name': loan['Account Name'],
                            'First Transaction Date': loan_analysis['first_transaction_date'],
                            'Older Than 12 Months': loan_analysis['is_older_than_12_months']
                        })
            
            # Display Loan Analysis Results
            if loan_analysis_results:
                loan_analysis_df = pd.DataFrame(loan_analysis_results)
                st.dataframe(loan_analysis_df)
            else:
                st.warning("No matching loan files found or could not analyze files.")

if __name__ == "__main__":
    main()