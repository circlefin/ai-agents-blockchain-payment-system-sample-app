import autogen
import json
import os

# Load OpenAI config
# config_list_gpt4 = autogen.config_list_from_json(
#     "OAI_CONFIG_LIST"
# )

def load_oai_config():
    with open('OAI_CONFIG_LIST', 'r') as f:
        config = json.load(f)

    for item in config:
        if item['api_key'] == '${OPENAI_API_KEY}':
            item['api_key'] = os.environ.get('OPENAI_API_KEY')
    
    return config

config = load_oai_config()

# Configuration for GPT-4
gpt4_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": config,
    "timeout": 120,
}

# Create AI agents
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt4_config,
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.""",
)

scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=gpt4_config,
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.""",
)

planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
    llm_config=gpt4_config,
)

executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "paper",
        "use_docker": False,
    },
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=gpt4_config,
)

# Create group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, scientist, planner, executor, critic],
    messages=[],
    max_round=50
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

def run_research(topic):
    initial_message = f"Research the following topic, check the arxiv website for interesting papers to support your result and create a markdown table of different domains: {topic}"
    user_proxy.initiate_chat(manager, message=initial_message)
    
    # For simplicity, we'll assume all agents contributed equally
    return {
        "Engineer": 1,
        "Scientist": 1,
        "Planner": 1,
        "Executor": 1,
        "Critic": 1
    }

if __name__ == "__main__":
    topic = "Artificial intelligence and employment trends"
    contributions = run_research(topic)
    print(json.dumps(contributions))