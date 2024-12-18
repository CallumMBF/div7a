```mermaid
flowchart TD
    Start[Upload Trial Balance] --> A[Read Account Entry]
    A --> B{Check Account Combination}
    B -->|Not in div7a_check list| C[Skip Analysis<br>Probability: 0.0]
    B -->|Matches watched combination| D[Batch Process with LLM]
    
    D --> E[Get Initial Probability<br>Based on Account Name Analysis]
    E --> F[Get Account Type & Balance]
    
    F --> G{Is Liability Account?}
    G -->|Yes| H{Has Debit Balance?}
    G -->|No| J[Keep Initial Probability]
    
    H -->|Yes| I[Add 0.1 to Probability<br>Max 1.0]
    H -->|No| J
    
    I --> K{Check Workpaper Records}
    J --> K
    
    K -->|Contains div7a terms| L[Set Final Probability 1.0<br>Mark as Already Documented]
    K -->|No div7a terms| M[Keep Current Probability]
    
    L --> N{Determine Status}
    M --> N
    
    N -->|Probability ≥ 0.4| O[High Risk:<br>Warning Required]
    N -->|Probability < 0.4| P[Low Risk:<br>No Warning Required]
    
    O --> Q[Generate Summary Report]
    P --> Q

    subgraph CheckList["Check List"]
        direction TB
        Z1["Assets:<br>- Loans to Members<br>- Other Assets<br>- Other Liabilities"]
        Z2["Equity:<br>- Adjustments to<br>  Retained Earnings"]
        Z3["Liabilities:<br>- Loans from Members<br>- Other Assets<br>- Other Liabilities<br>- Borrowings"]
        Z4["Other:<br>- Retained Earnings"]
    end

    %% Position the checklist near the initial check
    B ~~~ CheckList

    style Z1 text-align:left
    style Z2 text-align:left
    style Z3 text-align:left
    style Z4 text-align:left

