from openai import OpenAI;
from dotenv import load_dotenv
import os
import sys
from system_prompt import SYSTEM_PROMPT

load_dotenv()

try:
    client=OpenAI(api_key=os.getenv("api_key"),base_url=os.getenv("base_url"))
except Expection as e:
    print(f"Error in creating OpenAI client {e}")
    sys.exit(1)


def main():
    messages=[
        {
            "role":"system",
            "content":SYSTEM_PROMPT
        }
    ]

    print("-"*50)
    print("TakBuddy: Hey there! Whats there in your mind today?")
    print("-"*50)



    while True:
        try:
            user_input=input("\n You: ").strip()
            if user_input.lower() in ["quit","exit"]:
                print(f"\n TaskBuddy: Goodby")
                break

            if not user_input.strip():
                continue


            messages.append({"role":"user","content":user_input})
            
            response=client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages
            )


            reply=response.choices[0].message.content
            print(f"\n TaskBuddy: {reply}")

            messages.append({"role":"assistant","content":reply})
        except KeyboardInterrupt:
            print("\n Taskbuddy: Goodbye!")
            break
        except Exception as e:
            print(f"An error occured {e}")


if __name__=="__main__":
    main()

