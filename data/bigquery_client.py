import os
import argparse
from google.cloud import bigquery


def run_query(query: str, sa_path: str = None):
    """Run a BigQuery SQL query and return a pandas DataFrame.

    Auth behavior:
    - If `sa_path` is provided, sets `GOOGLE_APPLICATION_CREDENTIALS` to it for the process.
    - Otherwise, uses Application Default Credentials (ADC) which works when `gcloud auth application-default login` is configured or when running on GCP.
    """
    # If service account provided, ensure file exists and set env var for the client library
    if sa_path:
        if not os.path.isfile(sa_path):
            raise FileNotFoundError(f"Service account file not found: {sa_path}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

    try:
     client = bigquery.Client()
    except Exception:
    # Any error from the client likely indicates missing credentials/configuration.
        raise RuntimeError(
      "Could not find Google Cloud credentials. "
    """BigQuery helper with a guest/demo fallback.

    Usage:
      - Guest/demo: python data/bigquery_client.py --guest
      - With service account JSON: python data/bigquery_client.py --sa path/to/sa.json

    Guest mode loads `demo/sample_covid.csv` from the repository so contributors and
    hackathon participants can run the tool without Google credentials.
    """

    import os
    import argparse
    from typing import Optional

    """BigQuery helper with a guest/demo fallback.

    Usage:
      - Guest/demo: python data/bigquery_client.py --guest
      - With service account JSON: python data/bigquery_client.py --sa path/to/sa.json

    Guest mode loads `demo/sample_covid.csv` from the repository so contributors and
    hackathon participants can run the tool without Google credentials.
    """

    import os
    import argparse
    """Cleaned placeholder file. The real implementation will be re-added next."""
    print("cleaned")