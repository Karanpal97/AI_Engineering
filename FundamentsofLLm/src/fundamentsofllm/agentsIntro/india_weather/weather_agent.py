import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import os
import json
from openai import OpenAI
from typing import Callable
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from open_meteo import get_weather, WeatherResult, WeatherError

load_dotenv()

# ── Pydantic arg / info models ───────────────────────────────────────────────

class GetWeatherArgs(BaseModel):
    """Validated arguments for the get_weather tool"""
    city_name: str = Field(
        ...,
        description=(
            "The name of an Indian city whose weather is requested. "
            "Examples: Mumbai, Delhi, Bengaluru, Chennai, Hyderabad, Kolkata, "
            "Pune, Ahmedabad, Jaipur, Lucknow, Surat, Nagpur, Indore, Patna, "
            "Bhopal, Chandigarh, Coimbatore, Visakhapatnam, Kochi, Vadodara."
        ),
    )


class ToolCallInfo(BaseModel):
    """Structured representation of a single tool call result"""
    name: str = Field(..., description="The name of the tool that was called")
    arguments: dict = Field(..., description="The arguments passed to the tool")
    result: str = Field(..., description="The JSON result returned by the tool")


# ── OpenAI client ────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("OPEN_AI_KEY"),
    base_url=os.getenv("base_url"),
)

# ── Tool definition for the LLM ─────────────────────────────────────────────

TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current real-time weather for any Indian city or local area. "
                "Returns temperature (°C), feels-like temperature, humidity, wind speed "
                "and a weather condition description."
            ),
            "parameters": GetWeatherArgs.model_json_schema(),
        },
    }
]

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a friendly and knowledgeable Indian weather assistant (मौसम सहायक).
Your job is to provide accurate, real-time weather information for Indian cities,
towns, and local areas.

When a user asks about the weather:
1. Think step-by-step: identify which city / cities they are asking about.
2. Call the get_weather tool for each city.
3. After receiving the result, respond in a warm, conversational tone —
   include temperature in both °C, feels-like info, humidity, wind, and
   a short practical tip (e.g., "carry an umbrella", "great day for a walk").

IMPORTANT RULES:
- Always call get_weather for every city mentioned; never guess or reuse old data.
- If the user writes a city name in Hindi or a local dialect, translate it to
  its standard English spelling before calling the tool (e.g., "मुम्बई" → "Mumbai").
- After each answer, always reason out loud so the user sees your thought process.
- Use relatable Indian references (e.g., mention the monsoon, chai weather, etc.)
  to make the response feel local and engaging.
"""

# ── Few-shot examples ────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    # ── Example 1: Single city ──────────────────────────────────────────────
    {
        "role": "user",
        "content": "How's the weather in Mumbai right now?",
    },
    {
        "role": "assistant",
        "content": (
            "The user is asking about Mumbai's current weather. "
            "Let me call get_weather with city_name='Mumbai'."
        ),
        "tool_calls": [
            {
                "id": "tc_01",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city_name": "Mumbai"}),
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_01",
        "content": json.dumps({
            "city": "Mumbai",
            "temperature_c": 31.2,
            "feels_like_c": 36.5,
            "humidity_percent": 82,
            "wind_speed_kmh": 18.4,
            "weather_condition": "Partly cloudy",
            "is_day": True,
        }),
    },
    {
        "role": "assistant",
        "content": (
            "🌤️ **Mumbai** right now:\n\n"
            "- 🌡️ Temperature: **31.2 °C** (feels like **36.5 °C** — quite humid!)\n"
            "- 💧 Humidity: **82%**\n"
            "- 💨 Wind: **18.4 km/h**\n"
            "- ☁️ Condition: **Partly cloudy**\n\n"
            "It's a warm, muggy daytime — perfect chai-cutting weather! "
            "If you're heading out, carry a water bottle and wear light cotton clothes. 🥤"
        ),
    },
    # ── Example 2: Two cities compared ─────────────────────────────────────
    {
        "role": "user",
        "content": "Compare the weather in Delhi and Bengaluru today.",
    },
    {
        "role": "assistant",
        "content": (
            "The user wants a weather comparison between Delhi and Bengaluru. "
            "I'll call get_weather for both cities simultaneously."
        ),
        "tool_calls": [
            {
                "id": "tc_02",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city_name": "Delhi"}),
                },
            },
            {
                "id": "tc_03",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city_name": "Bengaluru"}),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_02",
        "content": json.dumps({
            "city": "Delhi",
            "temperature_c": 38.5,
            "feels_like_c": 42.1,
            "humidity_percent": 35,
            "wind_speed_kmh": 12.0,
            "weather_condition": "Clear sky",
            "is_day": True,
        }),
    },
    {
        "role": "tool",
        "tool_call_id": "tc_03",
        "content": json.dumps({
            "city": "Bengaluru",
            "temperature_c": 24.3,
            "feels_like_c": 25.0,
            "humidity_percent": 68,
            "wind_speed_kmh": 9.5,
            "weather_condition": "Partly cloudy",
            "is_day": True,
        }),
    },
    {
        "role": "assistant",
        "content": (
            "Here's today's weather comparison 🇮🇳:\n\n"
            "| | Delhi 🏙️ | Bengaluru 🌳 |\n"
            "|---|---|---|\n"
            "| 🌡️ Temp | 38.5 °C (feels 42.1 °C) | 24.3 °C (feels 25.0 °C) |\n"
            "| 💧 Humidity | 35% | 68% |\n"
            "| 💨 Wind | 12.0 km/h | 9.5 km/h |\n"
            "| ☁️ Condition | Clear sky ☀️ | Partly cloudy 🌤️ |\n\n"
            "Delhi is scorching hot today — stay indoors during peak afternoon hours "
            "and keep a cold paani bottle handy! 🥵\n\n"
            "Bengaluru, true to its 'Garden City' reputation, is pleasantly cool — "
            "a great day for a walk in Cubbon Park. 🌿"
        ),
    },
]

# ── Tool registry ─────────────────────────────────────────────────────────────

AVAILABLE_TOOLS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
}

TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "get_weather": GetWeatherArgs,
}

# ── Agentic loop ──────────────────────────────────────────────────────────────

def run_agent(user_query: str) -> None:
    """Run a single user query through the weather agent agentic loop."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'='*60}")
    print(f"You: {user_query}")
    print(f"{'='*60}\n")

    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOL_DEFINITION,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        messages.append(assistant_msg)

        if assistant_msg.content:
            print(f"Assistant (thinking): {assistant_msg.content}\n")

        if choice.finish_reason == "stop":
            break

        if not assistant_msg.tool_calls:
            break

        # ── Execute each tool call ────────────────────────────────────────
        for tool_call in assistant_msg.tool_calls:
            fn_name = tool_call.function.name
            raw_args = json.loads(tool_call.function.arguments)

            arg_model = TOOL_ARG_MODELS.get(fn_name)
            if arg_model is None:
                raise ValueError(f"No argument model found for tool: {fn_name}")

            validated_args = arg_model.model_validate(raw_args)
            print(f" [Tool Call] {fn_name}({validated_args.model_dump()})")

            fn = AVAILABLE_TOOLS[fn_name]
            raw_result = fn(**validated_args.model_dump())

            parsed_json = json.loads(raw_result)

            if "error" in parsed_json:
                parsed_result = WeatherError.model_validate(parsed_json)
                print(f" [Tool Error] {parsed_result.error}")
            else:
                parsed_result = WeatherResult.model_validate(parsed_json)
                print(f" [Tool Result] {parsed_result.model_dump_json()}")

            call_info = ToolCallInfo(
                name=fn_name,
                arguments=validated_args.model_dump(),
                result=parsed_result.model_dump_json(),
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": call_info.model_dump_json(),
            })


# ── Interactive REPL ──────────────────────────────────────────────────────────

def main() -> None:
    print("🌦️  India Weather Agent  🇮🇳")
    print("Ask about weather in any Indian city (type 'quit' to exit)")
    print("-" * 55)

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye! Stay weather-ready! ☂️")
            break

        run_agent(query)


if __name__ == "__main__":
    main()
