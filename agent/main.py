import os
import pandas as pd
from typing import Optional, Literal, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import openai
from dotenv import load_dotenv
from data.looker_client import LookerClient, SimpleLookerClient

# -----------------------------
# Environment setup
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize clients once
openai.api_key = os.environ.get("OPENAI_API_KEY")

# -----------------------------
# Configuration & Constants
# -----------------------------
class DataIntent(Enum):
    COVID = "covid"
    INCOME = "income"
    UNKNOWN = "unknown"

@dataclass
class QueryConfig:
    """Configuration for query execution"""
    default_limit: int = 5
    default_state: str = "California"
    cache_timeout: int = 3600  # seconds
    max_retries: int = 3

# -----------------------------
# FIPS Code Lookup (cached)
# -----------------------------
@lru_cache(maxsize=1)
def get_fips_to_state_map() -> Dict[str, str]:
    """Cached FIPS to state mapping"""
    return {
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
# SQL templates
# -----------------------------
SQL_TEMPLATES = {
    "covid": """
        SELECT state_name, date, confirmed_cases, deaths
        FROM `bigquery-public-data.covid19_nyt.us_states`
        WHERE state_name = '{state_name}'
        ORDER BY date DESC
        LIMIT {limit}
    """,
    "economy": """
        SELECT geo_id, {income_col} AS median_household_income
        FROM {table}
        ORDER BY median_household_income DESC
        LIMIT {limit}
    """
}

ECONOMY_TABLE = "`bigquery-public-data.census_bureau_acs.state_2021_1yr`"

# -----------------------------
# Query Intent Detector
# -----------------------------
class IntentDetector:
    """Improved intent detection with keyword scoring"""
    
    INTENT_KEYWORDS = {
        DataIntent.COVID: ["covid", "case", "confirmed", "death", "pandemic", "infection", "virus"],
        DataIntent.INCOME: ["income", "economy", "median", "household", "salary", "wage", "earnings", "wealth"]
    }
    
    @classmethod
    def detect(cls, text: str) -> DataIntent:
        """Detect intent from text using keyword scoring"""
        text_lower = text.lower()
        scores = {}
        
        for intent, keywords in cls.INTENT_KEYWORDS.items():
            scores[intent] = sum(1 for kw in keywords if kw in text_lower)
        
        if not any(scores.values()):
            return DataIntent.UNKNOWN
            
        return max(scores, key=scores.get)

# -----------------------------
# Data Processing Pipeline
# -----------------------------
class DataProcessor:
    """Centralized data processing logic"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def extract_state(text: str) -> Optional[str]:
        """Extract state name from text with caching"""
        import re
        text_lower = text.lower()
        fips_map = get_fips_to_state_map()
        
        # Check for FIPS code
        fips_match = re.search(r"\b(\d{1,2})\b", text_lower)
        if fips_match:
            fips = fips_match.group(1).zfill(2)
            if fips in fips_map:
                return fips_map[fips]
        
        # Check for state name
        for _, name in fips_map.items():
            if name.lower() in text_lower:
                return name
        
        return None
    
    @staticmethod
    def process_economy_df(df: pd.DataFrame) -> pd.DataFrame:
        """Optimized economy dataframe processing"""
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # Process geo_id if present
        if "geo_id" in df.columns:
            fips_map = get_fips_to_state_map()
            # Vectorized operation for better performance
            df["state_fips"] = df["geo_id"].astype(str).str.extract(r"(\d{2})$")[0].str.zfill(2)
            df["state_name"] = df["state_fips"].map(fips_map)
            df = df.drop(columns=["geo_id", "state_fips"], errors='ignore')
        
        # Standardize column names
        df = df.rename(columns={
            "median_household_income": "median_income",
            "income_per_capita": "median_income",
            "income": "median_income"
        })
        
        # Select relevant columns
        relevant_cols = ["state_name", "median_income"]
        available_cols = [c for c in relevant_cols if c in df.columns]
        
        return df[available_cols] if available_cols else df

# -----------------------------
# Query Executor
# -----------------------------
class QueryExecutor:
    """Handles query execution with retry logic and error handling"""
    
    def __init__(self, bq_client, config: QueryConfig = QueryConfig()):
        self.bq = bq_client
        self.config = config
        self.processor = DataProcessor()
    
    def execute_with_retry(self, query: str, retries: int = None) -> pd.DataFrame:
        """Execute query with retry logic"""
        retries = retries or self.config.max_retries
        
        for attempt in range(retries):
            try:
                return self.bq.run_query_dataframe(query)
            except Exception as e:
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    raise
        
        return pd.DataFrame()
    
    def run_covid_query(self, state: str = None, limit: int = None) -> pd.DataFrame:
        """Execute COVID data query"""
        state = state or self.config.default_state
        limit = limit or self.config.default_limit
        
        query = SQL_TEMPLATES["covid"].format(state_name=state, limit=limit)
        return self.execute_with_retry(query)
    
    def run_income_query(self, limit: int = None) -> pd.DataFrame:
        """Execute income data query with column detection"""
        limit = limit or self.config.default_limit
        
        # Detect income column
        income_col = self._detect_income_column() or "B19013_001E"
        
        query = SQL_TEMPLATES["economy"].format(
            table=ECONOMY_TABLE,
            income_col=income_col,
            limit=limit
        )
        
        df = self.execute_with_retry(query)
        return self.processor.process_economy_df(df)
    
    @lru_cache(maxsize=1)
    def _detect_income_column(self) -> Optional[str]:
        """Cached income column detection"""
        try:
            df = self.bq.run_query_dataframe(f"SELECT * FROM {ECONOMY_TABLE} LIMIT 1")
            candidates = ["b19013_001e", "median_household_income", "income_per_capita", "income"]
            
            for candidate in candidates:
                for col in df.columns:
                    if col.lower() == candidate:
                        return col
        except Exception as e:
            logger.error(f"Failed to detect income column: {e}")
        
        return None

# -----------------------------
# AI Summarization
# -----------------------------
class AISummarizer:
    """Handles AI-based summarization with caching"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._cache = {}
    
    def summarize(self, question: str, df: pd.DataFrame) -> str:
        """Generate AI summary with caching"""
        # Create cache key
        cache_key = hash((question, df.to_string()))
        
        if cache_key in self._cache:
            logger.info("Using cached summary")
            return self._cache[cache_key]
        
        # Prepare data for summarization
        df_preview = self._prepare_data(df)
        
        messages = [
            {
                "role": "system",
                "content": "You are a factual assistant that summarizes tabular data clearly and concisely."
            },
            {
                "role": "user",
                "content": f"Question: {question}\nData:\n{df_preview}\nProvide a clear, concise summary."
            }
        ]
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=500
            )
            summary = response.choices[0].message.content
            self._cache[cache_key] = summary
            return summary
        except Exception as e:
            logger.error(f"AI summarization failed: {e}")
            return "Unable to generate summary at this time."
    
    def _prepare_data(self, df: pd.DataFrame) -> str:
        """Prepare dataframe for AI consumption"""
        df_copy = df.copy()
        
        # Round numeric columns for cleaner display
        numeric_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if 'income' in col.lower():
                df_copy[col] = df_copy[col].round(0).astype(int)
        
        # Limit to top 10 rows for efficiency
        return df_copy.head(10).to_csv(index=False)

# -----------------------------
# Main CivicAI Class
# -----------------------------
class CivicAI:
    """Main CivicAI agent with optimized pipeline"""
    
    def __init__(self, bq_client, config: QueryConfig = QueryConfig()):
        self.executor = QueryExecutor(bq_client, config)
        self.summarizer = AISummarizer()
        self.processor = DataProcessor()
        self.config = config
        self.looker = LookerClient() 
    
    def process_query_with_looker(self, user_question: str) -> Dict[str, Any]:
        """Enhanced version that includes Looker dashboard URL"""
        result = self.process_query(user_question)
        
        if result.get('success'):
            # Generate Looker URL based on query type and data
            query_type = result['metadata'].get('intent', 'covid')
            state = result['metadata'].get('state')
            
            looker_url = self.looker.get_embed_url(
                query_type=query_type,
                state=state
            )
            
            result['looker_url'] = looker_url
        
        return result
    
    def process_query(self, user_question: str, source: str = "bigquery") -> Dict[str, Any]:
        """
        Process user query and return results with summary
        
        Returns:
            Dict containing 'data', 'summary', and 'metadata'
        """
        logger.info(f"Processing query: {user_question}")
        
        # Detect intent
        intent = IntentDetector.detect(user_question)
        
        if intent == DataIntent.UNKNOWN:
            return {
                "error": "Unable to determine query intent. Please ask about COVID or income data.",
                "data": pd.DataFrame(),
                "summary": None,
                "metadata": {"intent": "unknown"}
            }
        
        # Execute appropriate query
        try:
            if intent == DataIntent.COVID:
                state = self.processor.extract_state(user_question) or self.config.default_state
                df = self.executor.run_covid_query(state)
                metadata = {"intent": "covid", "state": state}
            
            elif intent == DataIntent.INCOME:
                df = self.executor.run_income_query()
                metadata = {"intent": "income"}
            
            else:
                raise ValueError(f"Unsupported intent: {intent}")
            
            # Generate summary if data exists
            summary = None
            if not df.empty:
                summary = self.summarizer.summarize(user_question, df)
            
            return {
                "data": df,
                "summary": summary,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "error": str(e),
                "data": pd.DataFrame(),
                "summary": None,
                "metadata": {"error": str(e)},
                "success": False
            }
    
    def batch_process(self, questions: list) -> list:
        """Process multiple questions in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.process_query, q): q for q in questions}
            
            for future in as_completed(futures):
                question = futures[future]
                try:
                    result = future.result(timeout=30)
                    result["question"] = question
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process '{question}': {e}")
                    results.append({
                        "question": question,
                        "error": str(e),
                        "success": False
                    })
        
        return results

# -----------------------------
# Main execution function
# -----------------------------
def main():
    """Optimized main function with better error handling and UX"""
    from data.bigquery_client import BigQueryClient
    
    print("🏛️ Welcome to CivicAI - Your Transparent Data Assistant")
    print("-" * 50)
    
    # Initialize components
    try:
        bq = BigQueryClient()
        civic_ai = CivicAI(bq)
        logger.info("CivicAI initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize CivicAI: {e}")
        return
    
    # Interactive mode
    while True:
        print("\n📊 What would you like to know about COVID or income data?")
        print("(Type 'exit' to quit, 'batch' for batch mode)")
        
        user_input = input("\n> ").strip()
        
        if user_input.lower() == 'exit':
            print("\n👋 Thank you for using CivicAI!")
            break
        
        elif user_input.lower() == 'batch':
            print("\nEnter questions (one per line, empty line to finish):")
            questions = []
            while True:
                q = input().strip()
                if not q:
                    break
                questions.append(q)
            
            if questions:
                print(f"\n⏳ Processing {len(questions)} questions...")
                results = civic_ai.batch_process(questions)
                
                for result in results:
                    print(f"\n❓ Question: {result['question']}")
                    if result.get('success'):
                        print(f"✅ Summary: {result['summary']}")
                    else:
                        print(f"❌ Error: {result.get('error')}")
        
        else:
            # Process single query
            result = civic_ai.process_query(user_input)
            
            if result.get('success'):
                df = result['data']
                
                if not df.empty:
                    print("\n📈 Raw Data Preview:")
                    print("-" * 50)
                    print(df.head())
                    
                    print("\n💡 CivicAI Analysis:")
                    print("-" * 50)
                    print(result['summary'])
                else:
                    print("\n⚠️ No data found for your query.")
            else:
                print(f"\n❌ Error: {result.get('error')}")
                print("Please try rephrasing your question or check your connection.")

if __name__ == "__main__":
    main()