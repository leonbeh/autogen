import pandas as pd
from ollama import chat
from pydantic import BaseModel
from enum import Enum

codes_activity_df = pd.read_csv("input/activity_types.csv")
codes_activity_df = codes_activity_df.dropna(subset=["Codes_Activity_Type", "Description"])
codes_activity_dict = dict(zip(codes_activity_df["Codes_Activity_Type"], codes_activity_df["Description"]))

activity_codes = codes_activity_df["Codes_Activity_Type"].tolist()

class ActivityType(str, Enum):
    report_task = "report_task"
    acknowledge_task = "acknowledge_task"
    provide_initial_assessment = "provide_initial_assessment"
    provide_recommendation = "provide_recommendation"
    request_elaboration = "request_elaboration"
    provide_elaborated_analysis = "provide_elaborated_analysis"
    request_comparison = "request_comparison"
    provide_comparative_analysis = "provide_comparative_analysis"
    provide_external_data = "provide_external_data"
    request_projection = "request_projection"
    provide_projection = "provide_projection"
    challenge_logic = "challenge_logic"
    negotiate_approach = "negotiate_approach"
    final_decision = "final_decision"
    define_next_steps = "define_next_steps"
    provide_expert_input = "provide_expert_input"
    request_recommendation = "request_recommendation"
    acknowledge = "acknowledge"
    state_decision = "state_decision"

codes_context_df = pd.read_csv("input/contexts.csv")
codes_context_df = codes_context_df.dropna(subset=["Codes_Context", "Description"])
codes_context_dict = dict(zip(codes_context_df["Codes_Context"], codes_context_df["Description"]))

context_codes = codes_context_df["Codes_Context"].tolist()

class Context(str, Enum):
    task_presentation = "task_presentation"
    analysis = "analysis"
    evaluation = "evaluation"
    planning = "planning"
    discussion = "discussion"
    decision = "decision"

class MessageClassification(BaseModel):
    activity: ActivityType
    context: Context

def format_activity_codes():
    formatted = ""
    for code in activity_codes:
        formatted += f"{code}: {codes_activity_dict[code]}\n"
    return formatted

def format_context_codes():
    formatted = ""
    for code in context_codes:
        formatted += f"{code}: {codes_context_dict[code]}\n"
    return formatted

def get_system_prompt():
    activity_codes_descriptions = format_activity_codes()
    context_codes_descriptions = format_context_codes()
    system_prompt = (
        f"You are a qualitative coder who classifies chat messages into activity types and contexts.\n"
        f"\nThese are the activity types:\n{activity_codes_descriptions}"
        f"\nThese are the contexts:\n{context_codes_descriptions}\n"
        f"For each message, you will received additional context:\n"
        f"- Actor: The person/agent who wrote the message (user_proxy is the user, all other actors are AI agents)\n"
        f"- Event ID: The position of the message within the chat (chats always start at 1)\n"
        f"Use the actor and event ID to understand the conversation flow when assigning exactly one activity type and context."
    )
    return system_prompt

def get_user_prompt(message_text, actor, event_id):
    system_prompt = get_system_prompt()
    user_prompt = f"{system_prompt}\nNow, classify the following message into one activity type and one context, respectively:\n"
    user_prompt += f"Message: {message_text}\n"
    user_prompt += f"Actor: {actor}\n"
    user_prompt += f"Event ID: {event_id}\n"
    return user_prompt

def classify_message(message_text, actor, event_id):
    user_prompt = get_user_prompt(message_text, actor, event_id)
    response = chat(
      model="phi4:14b",
      messages=[{"role": "user", "content": user_prompt}],
      format=MessageClassification.model_json_schema()
    )

    codes = MessageClassification.model_validate_json(response.message.content)
    return codes

def main():
    messages_df = pd.read_csv("input/messages.csv")
    results = []
    total_messages = len(messages_df)

    for idx, (_, row) in enumerate(messages_df.iterrows(), 1):
        codes = classify_message(row["Message"], row["Actor"], row["Event_ID"])
        results.append(codes)

        if idx % 100 == 0:
            print(f"Processed {idx} messages")
            results_df = pd.DataFrame([{"activity": r.activity.value, "context": r.context.value} for r in results])
            results_df.to_excel(f"output/classified_messages_phi4_{idx}.xlsx", index=False)
            print(f"Saved progress to output/classified_messages_phi4_{idx}.xlsx")

    print(f"Processed all {total_messages} messages")
    results_df = pd.DataFrame([{"activity": r.activity.value, "context": r.context.value} for r in results])
    results_df.to_excel("output/classified_messages_phi4_final.xlsx", index=False)
    print(f"Final results saved to output/classified_messages_phi4_final.xlsx")

if __name__ == "__main__":
    main()