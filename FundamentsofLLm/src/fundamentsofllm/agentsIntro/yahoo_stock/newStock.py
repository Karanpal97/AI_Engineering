import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from yahoo_finance import get_stock_price
from yahoo_finance import StockPriceResult, StockPriceError
from pydantic import Field,BaseModel
from typing import Callable

load_dotenv()

class GetStockPriceArguments(BaseModel):
    """this tells us the orguments require for a tool"""
    ticker_symbol: str=Field(...,description="this is the code for a stock of a company")


class toolInfo(BaseModel):
    """Structure represent the simple tool call """
    name: str=Field(...,description="the name of tool we calles"),
    arguments: dict=Field(...,description="the arguments required by a tool")
    result: str=Field(...,description="the result by a tool")


client=OpenAI(
    api_key=os.getenv("OPEN_AI_KEY"),
    base_url=os.getenv("base_url")
)

SYSTEM_PROMPT = """
You are a helpful stock market assistant. When the user asks about a stock price, 
think step-by-step about what you need to do, then call the get_stock_price tool with 
the correct ticker symbol. After receiving the result, present the information in a clear,
and friendly manner.

IMPORTANT: Always call the get_stock_price tool with the correct ticker symbol. Don't pick up
the price from our old conversations, as the price keeps on changing. Donot assume the price
from previous conversations. Always whenever a price for a stock is asked, we need to call the
relevant tools again.

Always reason out loud before acting so the user can follow your chain of thought.
"""

FEW_SHOT_EXAMPLES = [
    # Example 1: Single stock lookup
    {
        "role": "user",
        "content": "How much is Apple stock right now?"
    },
    {
        "role": "assistant",
        "content": (
            "The user is asking about Apple's stock price. Apple's ticker symbol is AAPL."
            "Let me fetch the current price of Apple stock."
        ),
        "tool_calls": [
            {
                "id": "tool_call_01",
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "arguments": json.dumps({
                        "ticker_symbol": "AAPL"
                    })
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "tool_call_01",
        "content": json.dumps({"ticker": "AAPL", "price": 200.3, "currency": "USD"}),
    },
    {
        "role": "assistant",
        "content": (
            "Apple (AAPL) is currently trading at $200.30 per share in USD."
            "Keep in mind this is the last traded price and may shift slightly by the time you check again."
        )
    },
    # Example 2: Multi stock comparison
    {
        "role": "user",
        "content": "Compare the stock prices of Google and Microsoft"
    },
    {
        "role": "assistant",
        "content": (
            "The user wants to compare two stocks. Google's parent company "
            "Alphabet trades under the ticker symbol GOOGL, and Microsoft trades "
            "under the ticker symbol MSFT."
            "Let me fetch the current prices of both stocks."
        ),
        "tool_calls": [
            {
                "id": "tool_call_02",
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "arguments": json.dumps({
                        "ticker_symbol": "GOOGL"
                    })
                }
            },
            {
                "id": "tool_call_03",
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "arguments": json.dumps({
                        "ticker_symbol": "MSFT"
                    })
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "tool_call_02",
        "content": json.dumps({"ticker": "GOOGL", "price": 280.12, "currency": "USD"}),
    },
    {
        "role": "tool",
        "tool_call_id": "tool_call_03",
        "content": json.dumps({"ticker": "MSFT", "price": 235.77, "currency": "USD"}),
    },
    {
        "role": "assistant",
        "content": (
            "Here's the comparison:\n\n"
            "- Alphabet/Google (GOOGL) is trading at $280.12 per share in USD.\n"
            "- Microsoft (MSFT) is trading at $235.77 per share in USD.\n\n"
            "Microsoft's stock is currently cheaper than Google's."
        )
    }
    
    
]
TOOL_DEFINATION=[
    {
        "type":"function",
        "function":{
            "name":"get_stock_price",
            "description":"Get the currect price of a stock given it's ticket symbol",
            "parameters":GetStockPriceArguments.model_json_schema()
        }
    }
]

AVAILABLE_TOOLS: dict[str, Callable[..., str]] = {
    "get_stock_price": get_stock_price,
}

TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "get_stock_price": GetStockPriceArguments,
}

def run_agent(user_query: str)->str:
    """This is the agent which fetch a updated stock price"""
    messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
                *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": user_query},
    ]

    print(f"\n{'='*60}")
    print(f"User: {user_query}")
    print(f"{'='*60}\n")

    while True:
        """Single turn agentic loop with chain of though reasoning"""
        result=client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOL_DEFINATION
        )

        choice=result.choices[0]
        assistance_msg=choice.message

        messages.append(assistance_msg)


        if assistance_msg.content:
            print(f"Assistant (thinking): {assistance_msg.content}\n")

        if choice.finish_reason=="stop":
            break

        if not assistance_msg.tool_calls:
            break

        for tool_call in assistance_msg.tool_calls:
            tool_name=tool_call.function.name
            raw_args=json.loads(tool_call.function.arguments)

            arg_model=TOOL_ARG_MODELS[tool_name]

            if arg_model is None:
                raise ValueError(f"No arguments model fond for tool:{tool_name}")

            validate_args=arg_model.model_validate(raw_args)
            print(f"[Tool Call] {tool_name}{validate_args.model_dump()}")

            fn=AVAILABLE_TOOLS[tool_name]
            result=fn(**validate_args.model_dump())

            raw_result=json.loads(result)

            if "error" in raw_result:
                parsed_result=StockPriceError.model_validate(raw_result)
                print(f" [Tool Error] {parsed_result.error}")
            else:
                parsed_result=StockPriceResult.model_validate(raw_result)
                print(f" [Tool Result] {parsed_result.model_dump()}")

            call_info=toolInfo(
                name=tool_name,
                arguments=validate_args.model_dump(),
                result=parsed_result.model_dump_json()
            )

            messages.append({
                 "role":"tool",
                 "content":call_info.model_dump_json(),
                 "tool_call_id":tool_call.id
            })

def main():
    print("Stock price agent (type 'quit' to exit) ")
    print("-" * 45)

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        run_agent(query)

if __name__ == "__main__":
    main()

            


        







