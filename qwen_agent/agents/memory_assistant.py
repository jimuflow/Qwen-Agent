from typing import Dict, Iterator, List, Optional, Union, Literal

from qwen_agent.agents import Assistant
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.tools import BaseTool
from qwen_agent.tools.memory_manager import MemoryManager


class MemoryAssistant(Assistant):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 memory: Optional[MemoryManager] = None):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files)
        self.memory = memory
        

    def _run(self,
             messages: List[Message],
             lang: Literal['en', 'zh'] = 'en',
             knowledge: str = '',
             **kwargs) -> Iterator[List[Message]]:
        new_message = self.memory.prepend_memory_info_to_sys(messages) if self.memory else messages
        
        for rsp in super()._run(new_message, lang=lang, knowledge=knowledge, **kwargs):
            yield rsp
