```mermaid
graph TD
    A[Account from Trial Balance] --> B{Check Div 7A Account Type}
    B -->|Not a Div 7A Account type| C[No Div7A Check Required<br>Probability = 0.0]
    
    B -->|Is a Div 7A Account type| D[LLM Analysis of Account Name]
    D --> E[Initial Probability Score 0.0-1.0]
    
    E --> F{Contains 'somewhat vague'<br>in reasoning?}
    F -->|Yes| G[Increase Probability by 0.1]
    F -->|No| H[Keep Original Probability]
    
    G --> I{Check Workpaper Records}
    H --> I
    
    I -->|Contains 'div 7a' terms| J[Set Probability = 1.0<br>Already contains Div 7A record]
    
    I -->|No div 7a terms| K{Check Account Type}
    
    K -->|Liabilities| L{Check Balance}
    L -->|Debit Balance +ve| M[Increase Probability by 0.4]
    L -->|Credit Balance -ve| N[Set Probability = 0.0]
    
    K -->|Assets/Equity| O[Keep Current Probability]
    
    M --> P{Final Probability Check}
    N --> P
    O --> P
    
    P -->|Probability >= 0.5| Q[High Risk - Warning Required]
    P -->|Probability < 0.5| R[Low Risk - No Warning]
    
    Q --> S[Display Results<br>⚠️/ℹ️/✅ Indicator]
    R --> S
