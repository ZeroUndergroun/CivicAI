import os
import pandas as pd
from typing import Optional
from data.bigquery_client import BigQueryClient
import openai

# -----------------------------
# Environment setup
# -----------------------------
# Ensure your BigQueryClient is initialized correctly as before
openai.api_key = os.environ.get("OPENAI_API_KEY")
bq = BigQueryClient()  # Your BigQuery client class

# -----------------------------
# FIPS Code Lookup (Needed for ACS Data)
# -----------------------------
# The ACS table 'state_2021_1yr' uses the GEO_ID column, 
# which is the FIPS code (e.g., '11' for DC, '25' for MA).
FIPS_TO_STATE_MAP = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas', '06': 'California', 
    '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '11': 'District of Columbia', 
    '12': 'Florida', '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois', 
    '18': 'Indiana', '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', 
    '23': 'Maine', '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', 
    '27': 'Minnesota', '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', 
    '31': 'Nebraska', '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey', 
    '35': 'New Mexico', '36': 'New York', '37': 'North Carolina', '38': 'North Dakota', 
    '39': 'Ohio', '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island', 
    '45': 'South Carolina', '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas', 
    '49': 'Utah', '50': 'Vermont', '51': 'Virginia', '53': 'Washington', 
    '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming'
}

# -----------------------------
# Example SQL templates
# -----------------------------
COVID_QUERY_TEMPLATE = """
SELECT state_name, date, confirmed_cases, deaths
FROM `bigquery-public-data.covid19_nyt.us_states`
WHERE state_name = '{state_name}'
ORDER BY date DESC
LIMIT {limit}
"""

# 💡 FIX: Using a known working table (state_2021_1yr) and Census field (B19013_001E)
# We'll build the economy query dynamically by detecting available income-like
# columns in the target table to avoid unrecognized column errors.
ECONOMY_TABLE = "`bigquery-public-data.census_bureau_acs.state_2021_1yr`"


def detect_income_column(table: str, client: BigQueryClient) -> Optional[str]:
    """Run a lightweight query to inspect column names in the table and
    return a preferred income-like column name if found.

    Preference order: B19013_001E, median_household_income, income_per_capita, income
    """
    # Query one row and inspect columns
    try:
        df = client.run_query_dataframe(f"SELECT * FROM {table} LIMIT 1")
    except Exception:
        return None

    cols = [c.lower() for c in df.columns]
    candidates = ["b19013_001e", "median_household_income", "income_per_capita", "income"]
    for c in candidates:
        if c in cols:
            # return the actual column name as present in the dataframe (case preserved)
            for orig in df.columns:
                if orig.lower() == c:
                    return orig
    return None


def build_economy_query(table: str, income_col: str, limit: int = 5) -> str:
    col_alias = 'median_household_income'
    return f"SELECT geo_id, {income_col} AS {col_alias} FROM {table} ORDER BY {col_alias} DESC LIMIT {limit}"

# -----------------------------
# Function to summarize data with AI
# -----------------------------
def summarize_with_ai(user_question: str, df: pd.DataFrame) -> str:
    """
    Sends user question + dataframe to OpenAI to produce a digestible summary.
    """
    # 💡 FIX: For better LLM output, format the income as a clean integer.
    if 'median_household_income' in df.columns:
        df['median_household_income'] = df['median_household_income'].round(0).astype(int)

    data_str = df.to_string(index=False)

    messages = [
        {"role": "system", "content": "You are a factual assistant that summarizes tabular data clearly. When summarizing income data, explicitly mention which state has the highest income."},
        {"role": "user", "content": f"Question: {user_question}\nData:\n{data_str}\nPlease summarize this for the user."}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content


# -----------------------------
# Helpers: parse user input for state and timeseries
# -----------------------------
def extract_state_from_text(text: str) -> Optional[str]:
    """Try to extract a US state name or FIPS code from the user's free text.

    Returns a normalized state_name (e.g. 'Massachusetts') or None.
    """
    text_lower = text.lower()
    import re

    # 1) Try to find an explicit FIPS code anywhere in the text. Users or data
    # sources sometimes provide '36', '036', '04000US36', or similar. Extract
    # the last two digits when possible which represent the state FIPS.
    fips_match = re.search(r"(\d{2})\b", text_lower)
    if not fips_match:
        # fallback: single-digit fips like '9' -> '09'
        fips_match = re.search(r"\b(\d)\b", text_lower)

    if fips_match:
        fips = fips_match.group(1).zfill(2)
        if fips in FIPS_TO_STATE_MAP:
            return FIPS_TO_STATE_MAP[fips]

    # 2) Look for any known state name in the text
    for code, name in FIPS_TO_STATE_MAP.items():
        if name.lower() in text_lower:
            return name

    return None


def extract_timeseries_from_text(text: str) -> Optional[str]:
    """Identify which timeseries the user asked about (covid, income, etc.)."""
    t = text.lower()
    if "covid" in t or "case" in t or "confirmed" in t:
        return "covid"
    if "income" in t or "economy" in t or "median" in t:
        return "income"
    return None


def process_economy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a dataframe returned from the economy query.

    Ensures geo_id values are normalized, maps to state_name using the
    FIPS_TO_STATE_MAP, renames income columns to 'median_income', and
    returns a cleaned dataframe ready for summarization.
    """
    if df is None or df.empty:
        return df

    # Work on a copy
    df = df.copy()

    # Normalize geo_id if present
    if 'geo_id' in df.columns:
        df['geo_id'] = df['geo_id'].astype(str)

        def normalize_geo_id(val: str) -> Optional[str]:
            if val is None:
                return None
            digits = ''.join([c for c in str(val) if c.isdigit()])
            if len(digits) == 0:
                return None
            fips = digits[-2:]
            return fips.zfill(2)

        df['geo_id_norm'] = df['geo_id'].apply(normalize_geo_id)
        df.insert(0, 'state_name', df['geo_id_norm'].map(FIPS_TO_STATE_MAP))

    # Normalize income column name
    if 'median_household_income' in df.columns:
        df = df.rename(columns={'median_household_income': 'median_income'})
    elif 'income_per_capita' in df.columns:
        df = df.rename(columns={'income_per_capita': 'median_income'})
    elif 'income' in df.columns:
        df = df.rename(columns={'income': 'median_income'})

    # Drop intermediate geo columns if present
    drop_cols = [c for c in ['geo_id', 'geo_id_norm'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Reorder to put state_name first if present
    cols = [c for c in ['state_name', 'median_income'] if c in df.columns]
    if cols:
        df = df[cols]

    return df

# -----------------------------
# Main loop / demo
# -----------------------------
if __name__ == "__main__":
    print("Welcome to CivicAI demo!")
    user_question = input("Ask a question about COVID or income data: ")
    
    df = pd.DataFrame() # Initialize df

    # Simple intent classification based on keywords
    if "covid" in user_question.lower():
        df = bq.run_query_dataframe(COVID_QUERY_TEMPLATE.format(state_name="California", limit=5))
        
    elif "income" in user_question.lower() or "economy" in user_question.lower():
        # Detect a suitable income column and build the query dynamically
        income_col = detect_income_column(ECONOMY_TABLE, bq)
        if income_col:
            query = build_economy_query(ECONOMY_TABLE, income_col, limit=5)
            try:
                df = bq.run_query_dataframe(query)
            except Exception:
                df = pd.DataFrame()
        else:
            # If we couldn't detect a valid column, try a conservative default
            try:
                query = build_economy_query(ECONOMY_TABLE, 'B19013_001E', limit=5)
                df = bq.run_query_dataframe(query)
            except Exception:
                df = pd.DataFrame()
        
        # 💡 Critical Fix: Ensure geo_id is a zero-padded string for correct mapping
        if 'geo_id' in df.columns:
            # Convert to string
            df['geo_id'] = df['geo_id'].astype(str)

            # Some Census/BQ exports use values like '04000US36' or '04000US36 ';
            # normalize by stripping non-digits and taking the last two digits
            def normalize_geo_id(val: str) -> Optional[str]:
                if val is None:
                    return None
                # keep only digits
                digits = ''.join([c for c in val if c.isdigit()])
                if len(digits) == 0:
                    return None
                # state FIPS are the last 2 digits in many GEOIDs
                fips = digits[-2:]
                return fips.zfill(2)

            df['geo_id_norm'] = df['geo_id'].apply(normalize_geo_id)

            # Perform the mapping using normalized geo id
            df.insert(0, 'state_name', df['geo_id_norm'].map(FIPS_TO_STATE_MAP))
            
            # Clean up and rename for the summary/printout
            drop_cols = [c for c in ['geo_id', 'geo_id_norm'] if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)

            # Some queries may call the income field different names; normalize
            if 'median_household_income' in df.columns:
                df = df.rename(columns={'median_household_income': 'median_income'})
            elif 'income_per_capita' in df.columns:
                df = df.rename(columns={'income_per_capita': 'median_income'})
            elif 'income' in df.columns:
                df = df.rename(columns={'income': 'median_income'})

            # Reorder columns for cleaner printout if both exist
            cols = [c for c in ['state_name', 'median_income'] if c in df.columns]
            df = df[cols]

    else:
        print("Sorry, I can only answer questions about COVID or income data in this demo.")
        exit()

    # Only continue if the query returned data
    if not df.empty:
        print("\n--- Raw Data ---")
        print(df)
    
        print("\n--- CivicAI Summary ---")
        summary = summarize_with_ai(user_question, df)
        print(summary)
    else:
        print("\n--- Query Failed ---")
        print("The BigQuery query did not return any results or failed unexpectedly.")