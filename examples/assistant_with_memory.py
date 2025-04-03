"""A sqlite database assistant implemented by assistant"""

import os
from typing import Optional

from qwen_agent.gui import WebUI
from qwen_agent.agents.memory_assistant import MemoryAssistant
from qwen_agent.tools.memory_manager import MemoryManager

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')
WORKSPACE = os.path.join(os.path.dirname(__file__), 'workspace')


def init_agent_service():
    llm_cfg = {'model': 'qwen-max'}
    tools = [{
        "mcpServers": {
            "sqlite" : {
                "command": "uvx",
                "args": [
                    "mcp-server-sqlite",
                    "--db-path",
                    os.path.join(WORKSPACE, "test.db")
                ]
            }
        }
    }]
    bot = MemoryAssistant(
        llm=llm_cfg,
        memory=MemoryManager(user_info="用户名叫张三，生日1991-09-26，居住在中国北京。"),
        system_message="尽量明确用户的需求后再给出回答，不要给出泛泛的或假设性的回答。",
        function_list=tools
    )
    return bot


def test(query='我要去爬山，需要带哪些东西？', file: Optional[str] = None):
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []

    if not file:
        messages.append({'role': 'user', 'content': query})
    else:
        messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

    for response in bot.run(messages):
        print('bot response:', response)


def app_tui():
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []
    while True:
        # Query example: 数据库里有几张表
        query = input('user question: ')
        if not query:
            print('user question cannot be empty！')
            continue
        messages.append({'role': 'user', 'content': query})

        response = []
        for response in bot.run(messages):
            print('bot response:', response)
        messages.extend(response)


def app_gui():
    # Define the agent
    bot = init_agent_service()
    chatbot_config = {
        'prompt.suggestions': [
            '小满的生日还有几天',
            '小满是我的女儿，生日2021年5月21号',
            '数据库里有几张表',
            '创建一个学生表包括学生的姓名、年龄',
            '增加一个学生名字叫韩梅梅，今年6岁',
        ]
    }
    WebUI(
        bot,
        chatbot_config=chatbot_config,
    ).run()


if __name__ == '__main__':
    # test()
    # app_tui()
    app_gui()
