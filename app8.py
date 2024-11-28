import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import os

class LoanAssessmentApp:
    def __init__(self, openai_api_key=None):
        """Initialize the Loan Assessment Application"""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.loan_detection_prompt = """
        Analyze the account name: '{account_name}'
        
        Examples of potential account names that could be subject to Division 7A reconciliation:
            Loan/Receivable Accounts:
                "Shareholder Loan Account"
                "Director's Loan Account"
                "Related Party Loan"
                "Associate Loan Account"
                "Employee Loan Account"
                "Family Trust Loan Account"
                "Shareholder Current Account"
                "Partner Drawing Account"

            Payment/Transaction Accounts:
                "Director's Drawings"
                "Shareholder Advances"
                "Management Fees Payable"
                "Intercompany Loan Account"
                "Beneficiary Loan Account"
                "Partner Loan Account"
            Investment Related:
                "Share Subscription Advance"
                "Investment Loan Account"
            Mixed Purpose Accounts:
                "Directors Current Account"
                "Associate Current Account"
                "Related Entity Advance"
                "Trust Distribution Loan"
            Other Common Names:
                "Division 7A Loan Account"
                "Unsecured Related Party Loan"
                "Secured Related Party Loan"
                "Personal Expense Account"
                "Family Member Loan Account"

        Loan Detection Criteria:
        1. Is this account likely to represent a loan?
        2. Provide a clear classification: 'Yes' (definite loan), 'Unsure' (potential loan), or 'No' (not a loan)
        3. If 'Yes' or 'Unsure', explain your reasoning in a brief one sentence answer.
        
        Consider:
        - Explicit loan terms (loan, lending, credit)
        - Financial instrument references
        - Contextual indicators of borrowing
        
        Response Format:
        Classification: [Yes/Unsure/No]
        Reasoning: [Explanation of classification]
        """

    def detect_loan(self, account_name):
        """Use LLM to detect if an account is a loan"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial classification expert."},
                    {"role": "user", "content": self.loan_detection_prompt.format(account_name=account_name)}
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            lines = result_text.split('\n')
            classification = lines[0].split(':')[1].strip() if ':' in lines[0] else 'No'
            reasoning = lines[1].split(':')[1].strip() if len(lines) > 1 and ':' in lines[1] else 'No specific reasoning provided'
            
            return {
                'account_name': account_name,
                'classification': classification,
                'reasoning': reasoning
            }
        except Exception as e:
            return {
                'account_name': account_name,
                'classification': 'Error',
                'reasoning': f'Loan Detection Error: {str(e)}'
            }

def check_div7a_status(account_type, balance, workpaper_record):
    """
    Determine if an account needs Div 7A reconciliation based on type, balance, and workpaper
    
    Returns:
    - needs_warning (bool): Whether to show the Div 7A warning
    - status (str): Status message about the account
    """
    div7a_warning = "If this loan is to an owner or an associate of an owner to the business then this account may be subject to div 7a reconciliation, please check and if so, attach the div7a rec calculator worksheet to the record."
    
    # Convert account_type to string and handle None/NaN
    if account_type is None or pd.isna(account_type):
        account_type = ''
    else:
        account_type = str(account_type)
        
    # Convert workpaper_record to string and handle None/NaN
    if workpaper_record is None or pd.isna(workpaper_record):
        workpaper_record = ''
    else:
        workpaper_record = str(workpaper_record)
    
    # No workpaper record case
    if not workpaper_record:
        if account_type.lower() == 'assets':
            return True, div7a_warning
        elif account_type.lower() == 'liabilities':
            if balance < 0:
                return True, div7a_warning
            else:
                return False, "This loan has been classed as a liability and credited. If this is correct then no Div 7A reconciliation required"
    
    # Has workpaper record case
    if any(term.lower() in workpaper_record.lower() for term in ['div 7a', 'division 7a']):
        return False, "Div 7A reconciliation worksheet already attached"
    
    # Has workpaper but not Div 7A
    if account_type.lower() == 'assets':
        return True, div7a_warning
    elif account_type.lower() == 'liabilities':
        if balance < 0:
            return True, div7a_warning
        else:
            return False, "This loan has been classed as a liability and credited. If this is correct then no Div 7A reconciliation required"
    
    return False, f"Account type '{account_type}' classification unclear - please review"

def analyze_account(loan_app, row):
    """Comprehensive analysis for a single account"""
    result = {
        'Account': row['AccountName'],
        'Balance': row['Balance'],
        'SourceAccountType': row['SourceAccountType'],
        'Loan Detection': {
            'Classification': None,
            'Reasoning': None
        },
        'Summary': None,
        'Div7A Status': None,
        'Requires_Warning': False
    }

    # Step 1: Detect if account is a loan
    loan_detection = loan_app.detect_loan(row['AccountName'])
    result['Loan Detection']['Classification'] = loan_detection['classification']
    result['Loan Detection']['Reasoning'] = loan_detection['reasoning']
    
    # Include all accounts in results
    if loan_detection['classification'] == 'No':
        result['Summary'] = 'Not likely to be a loan requiring div 7a reconciliation'
        result['Div7A Status'] = 'No action required'
        return result
    
    # Step 2: For potential loans, check Div 7A status
    if loan_detection['classification'] in ['Yes', 'Unsure']:
        needs_warning, status = check_div7a_status(
            row['SourceAccountType'],
            row['Balance'],
            row.get('WorkpaperRecordsName', '')
        )
        
        result['Requires_Warning'] = needs_warning
        result['Div7A Status'] = status
        result['Summary'] = status if not needs_warning else 'Potential Div 7A loan - requires attention'
    
    return result

def main():
    st.title("Div 7A Loan Assessment Tool")
    
    st.sidebar.header("Configuration")
    openai_api_key = os.getenv('OPENAI_API_KEY')

    trial_balance_file = st.sidebar.file_uploader("Upload Trial Balance", type=['xlsx'])
    
    results = []
    
    if st.sidebar.button("Analyze Accounts") and trial_balance_file and openai_api_key:
        loan_app = LoanAssessmentApp(openai_api_key)
        trial_balance_df = pd.read_excel(trial_balance_file)
        
        # Verify required columns exist
        required_columns = ['AccountName', 'Balance', 'SourceAccountType']
        missing_columns = [col for col in required_columns if col not in trial_balance_df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in trial balance: {', '.join(missing_columns)}")
            return
        
        progress_bar = st.progress(0)
        
        for index, row in trial_balance_df.iterrows():
            progress_bar.progress(int((index + 1) / len(trial_balance_df) * 100))
            account_result = analyze_account(loan_app, row)
            results.append(account_result)
    
    if results:
        results_df = pd.DataFrame(results)
        st.header("Loan Assessment Results")
        
        # Create a sorting helper function to assign priority
        def get_sort_priority(result):
            if result['Requires_Warning']:
                return 0  # Warning accounts first
            elif result['Loan Detection']['Classification'] in ['Yes', 'Unsure']:
                return 1  # Info accounts second
            else:
                return 2  # Checkmark accounts last
        
        # Sort results based on priority
        sorted_results = sorted(results, key=get_sort_priority)
        
        # Display sorted results
        for result in sorted_results:
            # Set emoji and color based on warning status and loan detection
            if result['Loan Detection']['Classification'] == 'No':
                status_color = 'green'
                status_emoji = "✅"  # Green checkmark for no issues
            elif result['Requires_Warning']:
                status_color = 'red'
                status_emoji = "⚠️"  # Warning emoji for attention required
            else:
                status_color = 'orange'
                status_emoji = "ℹ️"  # Info emoji for reviewed accounts
            
            with st.expander(f"{status_emoji} {result['Account']} - {result['Summary']}", expanded=False):
                st.markdown(f"**Summary:** <font color='{status_color}'>{result['Summary']}</font>", unsafe_allow_html=True)
                
                st.subheader("Loan Detection")
                st.write(f"**Classification:** {result['Loan Detection']['Classification']}")
                st.write(f"**Reasoning:** {result['Loan Detection']['Reasoning']}")
                
                if result['Div7A Status']:
                    st.subheader("Div 7A Status")
                    st.write(result['Div7A Status'])

        # Download option for full results
        # Convert sorted results back to DataFrame for download
        sorted_results_df = pd.DataFrame(sorted_results)
        st.download_button(
            label="Download Full Results",
            data=sorted_results_df.to_csv(index=False),
            file_name="loan_assessment_results.csv",
            mime="text/csv"
        )
            
    else:
        st.warning("No accounts analyzed")

if __name__ == "__main__":
    main()