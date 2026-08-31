import os
import json
from typing import Callable, Any

from dotenv import load_dotenv
from openai import OpenAI
from google.cloud import bigquery
from pydantic import BaseModel, Field


load_dotenv()


# ============================================================
# Configuration
# ============================================================

PROJECT_ID = "project-b2f1328b-69cb-4843-a87"
DATASET_ID = "hcl_demo"
TABLE_ID = "orders_clean"

FULL_TABLE_NAME = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"


client = OpenAI(
    api_key=os.getenv("OPEN_AI_KEY"),
    base_url=os.getenv("base_url")
)


bq_client = bigquery.Client(
    project=PROJECT_ID
)


# ============================================================
# Database Schema
# ============================================================

SCHEMA = f"""
Table: {FULL_TABLE_NAME}

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


# ============================================================
# Pydantic Models
# ============================================================

class ExecuteBigQueryArgs(BaseModel):
    """
    Validated arguments for the execute_bigquery tool.
    """

    sql: str = Field(
        ...,
        description=(
            "A valid BigQuery Standard SQL SELECT query "
            f"using only the table {FULL_TABLE_NAME}"
        )
    )


class BigQueryResult(BaseModel):
    """
    Structured representation of a successful BigQuery result.
    """

    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rows returned by BigQuery"
    )


class BigQueryError(BaseModel):
    """
    Structured representation of a BigQuery error.
    """

    error: str


class ToolCallInfo(BaseModel):
    """
    Structured information about a tool call.
    """

    name: str = Field(
        ...,
        description="Name of the tool that was called"
    )

    arguments: dict[str, Any] = Field(
        ...,
        description="Validated arguments passed to the tool"
    )

    result: str = Field(
        ...,
        description="Result returned by the tool"
    )


# ============================================================
# BigQuery Tool
# ============================================================

def execute_bigquery(sql: str) -> str:
    """
    Execute a read-only BigQuery SQL query.
    """

    print("\n[BigQuery SQL]")
    print(sql)

    sql_upper = sql.strip().upper()

    # --------------------------------------------------------
    # Basic SELECT validation
    # --------------------------------------------------------

    if not sql_upper.startswith("SELECT"):
        return json.dumps({
            "error": "Only SELECT queries are allowed."
        })

    forbidden_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "MERGE"
    ]

    for keyword in forbidden_keywords:

        if keyword in sql_upper:

            return json.dumps({
                "error": f"Forbidden SQL operation: {keyword}"
            })

    # --------------------------------------------------------
    # Execute BigQuery
    # --------------------------------------------------------

    try:

        query_job = bq_client.query(sql)

        rows = query_job.result()

        results = []

        for row in rows:
            results.append(dict(row.items()))

        return json.dumps(
            {
                "rows": results
            },
            default=str
        )

    except Exception as e:

        return json.dumps({
            "error": str(e)
        })


# ============================================================
# Tool Definition
# ============================================================

TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "execute_bigquery",

            "description": (
                "Execute a read-only BigQuery SQL query. "
                "Use this tool whenever the user asks a question "
                "that requires information from the orders database."
            ),

            "parameters": ExecuteBigQueryArgs.model_json_schema(),
        }
    }
]


# ============================================================
# Available Tools
# ============================================================

AVAILABLE_TOOLS: dict[str, Callable[..., str]] = {

    "execute_bigquery": execute_bigquery,

}


# ============================================================
# Tool Argument Models
# ============================================================

TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {

    "execute_bigquery": ExecuteBigQueryArgs,

}


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = f"""
You are a helpful data analyst AI agent.

You have access to this BigQuery table:

{SCHEMA}

Your job is to answer questions about the data.

IMPORTANT RULES:

1. Whenever the user's question requires information from
   BigQuery, use the execute_bigquery tool.

2. Generate BigQuery Standard SQL.

3. You may ONLY query this table:

   {FULL_TABLE_NAME}

4. Only generate SELECT queries.

5. Never modify data.

6. Never use:

   INSERT
   UPDATE
   DELETE
   DROP
   CREATE
   ALTER
   TRUNCATE
   MERGE

7. After receiving the BigQuery result, answer the user's
   question clearly using the returned data.

8. Never invent values that are not present in the result.

9. If the user's question does not require BigQuery,
   answer normally without calling the tool.

10. If a BigQuery query fails, inspect the error and generate
    a corrected query when possible.

11. Do not expose internal reasoning or chain-of-thought.
    Only provide the final answer and useful information
    such as the SQL query when appropriate.
"""


# ============================================================
# Agent
# ============================================================

def run_agent(user_query: str) -> None:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": user_query
        }
    ]

    print("\n" + "=" * 60)
    print(f"User: {user_query}")
    print("=" * 60)

    # ========================================================
    # Agent Loop
    # ========================================================

    while True:

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOL_DEFINITION,
            tool_choice="auto"
        )

        choice = response.choices[0]

        assistant_msg = choice.message

        # ----------------------------------------------------
        # Add assistant message to conversation
        # ----------------------------------------------------

        messages.append(assistant_msg)

        # ----------------------------------------------------
        # If model produced normal text
        # ----------------------------------------------------

        if assistant_msg.content:

            print(
                f"\nAssistant: {assistant_msg.content}"
            )

        # ----------------------------------------------------
        # If model doesn't want to call a tool
        # ----------------------------------------------------

        if choice.finish_reason == "stop":

            break

        # ----------------------------------------------------
        # No tool calls
        # ----------------------------------------------------

        if not assistant_msg.tool_calls:

            break

        # ====================================================
        # Handle Tool Calls
        # ====================================================

        for tool_call in assistant_msg.tool_calls:

            fn_name = tool_call.function.name

            raw_args = json.loads(
                tool_call.function.arguments
            )

            print(
                f"\n[Tool Call] {fn_name}"
            )

            # ------------------------------------------------
            # Find Pydantic argument model
            # ------------------------------------------------

            arg_model = TOOL_ARG_MODELS.get(fn_name)

            if arg_model is None:

                raise ValueError(
                    f"No argument model found for tool: {fn_name}"
                )

            # ------------------------------------------------
            # Validate arguments with Pydantic
            # ------------------------------------------------

            validated_args = arg_model.model_validate(
                raw_args
            )

            print(
                "[Validated Arguments]",
                validated_args.model_dump()
            )

            # ------------------------------------------------
            # Find actual Python function
            # ------------------------------------------------

            fn = AVAILABLE_TOOLS.get(fn_name)

            if fn is None:

                raise ValueError(
                    f"No implementation found for tool: {fn_name}"
                )

            # ------------------------------------------------
            # Execute tool
            # ------------------------------------------------

            raw_result = fn(
                **validated_args.model_dump()
            )

            # ------------------------------------------------
            # Parse result
            # ------------------------------------------------

            result_data = json.loads(raw_result)

            if "error" in result_data:

                parsed_result = BigQueryError.model_validate(
                    result_data
                )

                print(
                    "[Tool Error]",
                    parsed_result.error
                )

            else:

                parsed_result = BigQueryResult.model_validate(
                    result_data
                )

                print(
                    "[Tool Result]",
                    parsed_result.model_dump()
                )

            # ------------------------------------------------
            # Create structured tool information
            # ------------------------------------------------

            call_info = ToolCallInfo(

                name=fn_name,

                arguments=validated_args.model_dump(),

                result=parsed_result.model_dump_json()

            )

            # ------------------------------------------------
            # Send result back to LLM
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",

                    "tool_call_id": tool_call.id,

                    "content": call_info.model_dump_json()
                }
            )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "BigQuery AI Agent V2"
    )

    print(
        "Type 'exit' to quit."
    )

    print("-" * 45)

    while True:

        query = input("\nYou: ").strip()

        if not query:
            continue

        if query.lower() in (
            "quit",
            "exit",
            "q"
        ):

            print("Goodbye!")

            break

        try:

            run_agent(query)

        except Exception as e:

            print(
                f"\nAgent Error: {e}"
            )


if __name__ == "__main__":

    main()