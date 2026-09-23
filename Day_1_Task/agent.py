"""System 3: an AI agent. LLM + tools + loop."""
import json
from Day_1_Task.config import client, MODEL, QUESTIONS, banner
from Day_1_Task.tools import TOOLS, TOOL_FUNCTIONS

SYSTEM_PROMPT = (
    "You are a college fee assistant. Never guess a fee: always use get_course_fee. "
    "Use calculator for any arithmetic. Available course codes: CS101, AI202, DS303. "
    "If no tool is needed, answer directly."
)


def agent(question, max_steps=6, verbose=True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    for step in range(1, max_steps + 1):

        # 1. REASON: ask the LLM what to do next
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.6,
            parallel_tool_calls=False,
        )

        message = response.choices[0].message

        # 2. If no tool is requested, the LLM has finished
        if not message.tool_calls:
            return message.content.strip()

        # 3. Add the assistant's tool request
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments
                    }
                }
                for call in message.tool_calls
            ]
        })

        # 4. ACT and OBSERVE: run the tool
        for call in message.tool_calls:
            name = call.function.name
            arguments = json.loads(call.function.arguments or "{}")

            function = TOOL_FUNCTIONS.get(name)

            if function:
                result = function(**arguments)
            else:
                result = f"Unknown tool: {name}"

            if verbose:
                print(
                    f"   step {step}: "
                    f"{name}({arguments}) -> {result}"
                )

            # Send tool result back to the model
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": str(result),
            })

    return "Stopped: maximum steps reached without a final answer."


if __name__ == "__main__":
    banner("SYSTEM 3: AI AGENT")

    for question in QUESTIONS:
        print("Q:", question)
        print("A:", agent(question))
        print("-" * 70)