from data.bigquery_client import BigQueryClient

bq = BigQueryClient()

query = """
SELECT
  geo_id AS state,
  median_income
FROM
  `bigquery-public-data.census_bureau_acs.state_2019_1yr`
ORDER BY
  median_income DESC
LIMIT 10;
"""

# If you want a Pandas DataFrame
df = bq.run_query_dataframe(query)
print("DataFrame output:")
print(df)

# If you want plain Python dicts
rows = bq.run_query_dict(query)
print("\nDict output:")
for row in rows:
    print(row)
