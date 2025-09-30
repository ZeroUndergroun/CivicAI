# test_agent.py
import os
import sys
import pandas as pd

# Ensure the repository root is on sys.path so we can import top-level packages
# when this script is executed from the scripts/ directory.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agent.main import bq, summarize_with_ai, process_economy_df

def test_civicai_agent(user_question: str):
    """
    Tests the CivicAI agent with a given user question.
    Fetches data and generates AI summary.
    """
    # Simple intent classification
    if "covid" in user_question.lower():
        df = bq.run_query_dataframe(
            """
            SELECT state_name, date, confirmed_cases, deaths
            FROM `bigquery-public-data.covid19_nyt.us_states`
            WHERE state_name = 'California'
            ORDER BY date DESC
            LIMIT 5
            """
        )
    elif "income" in user_question.lower() or "economy" in user_question.lower():
        try:
            df = bq.run_query_dataframe(
                """
                SELECT geo_id, B19013_001E AS median_household_income
                FROM `bigquery-public-data.census_bureau_acs.state_2021_1yr`
                ORDER BY median_household_income DESC
                LIMIT 5
                """
            )
        except Exception as e:
            # If the real BigQuery schema differs or query fails, fall back to
            # a small mock DataFrame so the rest of the pipeline can be tested
            print("\n⚠️ BigQuery query failed — falling back to mock data for testing:", e)
            mock = {
                'geo_id': ['04000US36', '36', '9'],
                'median_household_income': [80000, 78000, 70000]
            }
            df = pd.DataFrame(mock)

        # Process economy dataframe (normalize geo_id, map to state_name, rename income)
        df = process_economy_df(df)
    else:
        print("Sorry, this demo agent only handles COVID or income questions.")
        return

    print("\n--- Raw Data ---")
    print(df)

    print("\n--- CivicAI Summary ---")
    try:
        summary = summarize_with_ai(user_question, df)
        print(summary)
    except Exception as e:
        print("\n⚠️ Summarization failed (OpenAI or other error):", e)
        print("Here is the processed DataFrame for inspection:")
        print(df)


if __name__ == "__main__":
    # Example questions to test
    questions = [
        "What are the latest COVID numbers in California?",
        "I heard a rumor online that the economy in 2021 was worse than the economy of today. Is that true?",
        # Extra economy prompts to test geo_id -> state mapping
        "Show me the top states by median household income.",
        "Which state (by FIPS 36) had the highest median household income in 2021?",
        "What does geo_id 04000US36 correspond to?"
    ]

    for q in questions:
        print(f"\n==== Testing question: {q} ====")
        test_civicai_agent(q)
