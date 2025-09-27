# data/bigquery_client.py

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
import pandas as pd


class BigQueryClient:
    def __init__(self, project_id=None):
        """
        Initialize the BigQuery client.

        Args:
            project_id (str, optional): GCP project ID. 
                                        If None, uses default credentials/project.
        """
        try:
            self.client = bigquery.Client(project=project_id)
            print(f"✅ Connected to BigQuery (project={self.client.project})")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize BigQuery client: {e}")

    def run_query_dataframe(self, query: str) -> pd.DataFrame:
        """
        Run a query and return results as a Pandas DataFrame.

        Args:
            query (str): SQL query string.

        Returns:
            pd.DataFrame: Query results.
        """
        try:
            results = self.client.query(query)
            df = results.to_dataframe()
            return df
        except GoogleAPIError as e:
            raise RuntimeError(f"❌ BigQuery API error: {e}")
        except Exception as e:
            raise RuntimeError(f"❌ Unexpected error in run_query_dataframe: {e}")

    def run_query_dict(self, query: str) -> list[dict]:
        """
        Run a query and return results as a list of dictionaries.

        Args:
            query (str): SQL query string.

        Returns:
            list[dict]: Query results.
        """
        try:
            results = self.client.query(query)
            rows = [dict(row) for row in results]
            return rows
        except GoogleAPIError as e:
            raise RuntimeError(f"❌ BigQuery API error: {e}")
        except Exception as e:
            raise RuntimeError(f"❌ Unexpected error in run_query_dict: {e}")
