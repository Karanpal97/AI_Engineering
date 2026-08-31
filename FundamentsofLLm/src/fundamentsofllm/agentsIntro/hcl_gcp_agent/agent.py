import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from google.cloud import bigquery


load_dotenv()


# --------------------------------
# Configuration
# --------------------------------

PROJECT_ID = "project-b2f1328b-69cb-4843-a87"
DATASET_ID = "hcl_demo"
TABLE_ID = "orders_clean"

client = OpenAI(
    api_key=os.getenv("OPEN_AI_KEY"),
    base_url=os.getenv("base_url")
)

bq_client = bigquery.Client(
    project=PROJECT_ID
)


# --------------------------------
# Database schema
# --------------------------------

SCHEMA = """
Table: hcl_demo.orders_clean

Columns:

order_id     INT64
customer     STRING
product      STRING
category     STRING
quantity     INT64
unit_price   FLOAT64
order_date   DATE
city         STRING
"""


# --------------------------------
# Generate SQL
# --------------------------------

def generate_sql(question: str) -> str:

    prompt = f"""
You are a SQL analyst.

You have access to this BigQuery table:

{SCHEMA}

Generate a BigQuery Standard SQL query that answers the user's question.

Rules:
- Only use hcl_demo.orders_clean
- Only generate SELECT queries
- Never modify data
- Do not use INSERT
- Do not use UPDATE
- Do not use DELETE
- Do not use DROP
- Do not use CREATE
- Do not use ALTER
- Return ONLY SQL
- Do not use markdown

User question:

{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()


# --------------------------------
# Execute BigQuery
# --------------------------------

def execute_query(sql: str):

    print("\nGenerated SQL:")
    print(sql)

    query_job = bq_client.query(sql)

    rows = query_job.result()

    results = []

    for row in rows:
        results.append(dict(row.items()))

    return results


# --------------------------------
# Generate final answer
# --------------------------------

def generate_answer(question: str, sql: str, results):

    prompt = f"""
You are a data analyst.

User question:
{question}

SQL used:
{sql}

BigQuery result:
{json.dumps(results, default=str)}

Answer the user's question using ONLY the BigQuery result.

Rules:
- Do not invent information.
- Keep the answer concise.
- Mention important numbers.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()


# --------------------------------
# Agent
# --------------------------------

def run_agent(question: str):

    sql = generate_sql(question)

    results = execute_query(sql)

    answer = generate_answer(
        question,
        sql,
        results
    )

    return answer


# --------------------------------
# Chat loop
# --------------------------------

if __name__ == "__main__":

    print("BigQuery AI Agent")
    print("Type 'exit' to quit.")

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        try:

            answer = run_agent(question)

            print("\nAgent:", answer)

        except Exception as e:

            print("\nError:", e)