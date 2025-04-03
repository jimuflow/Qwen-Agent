"""
记忆管理接口模块
提供记忆的存储和向量检索功能
"""
import json
import copy
import os
import sqlite3
from datetime import datetime
from typing import List, Tuple, Optional, Union, Dict
from qwen_agent.llm.schema import SYSTEM, USER, ContentItem, Message, ASSISTANT
from qwen_agent.utils.utils import format_as_text_message
from qwen_agent.llm import get_chat_model
from qwen_agent.llm import BaseChatModel
from qwen_agent.log import logger
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError('请安装 langchain: `pip install langchain`') from exc

try:
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_community.vectorstores import FAISS
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        '请安装 langchain_community: `pip install langchain_community`, '
        '并安装 faiss: `pip install faiss-cpu` 或 `pip install faiss-gpu` (对于支持CUDA的GPU)') from exc

MEMORY_PROMPT = """
# 当前对话背景
当前时间：{current_time}
用户信息：{user_info}
# 关于用户的记忆
memory中包含多条记忆，每一条记忆的格式为：记忆ID @ 记忆时间: 记忆内容。

<memory>
{memory}
</memory>
"""

EXTRACT_MEMORY_PROMPT = """
请根据下面这段对话，为AI助手生成有价值的记忆内容，比如用户相关信息、事件等，以便在后续对话中提供给AI助手，只需要输出记忆内容，不要添加解释或说明。如果没有需要记录的内容，则输出“无”。

<dialogue>
{dialogue}
</dialogue>
"""

CHECK_MEMORY_PROMPT="""
请检查下面这个记忆与已有记忆的关系，并生成检查结果。

# 待检查记忆
{new_memory}

# 已有记忆
memory中包含多条记忆，每一条记忆的格式为：记忆ID @ 记忆时间: 记忆内容。
<memory>
{old_memory}
</memory>

# 检查结果输出格式
使用json格式输出，格式如下：
```json
{{
	"thinking": "你的思考过程",
	"result": {{"relation":"关系","related_memories":["记忆ID1","记忆ID2"],"new_memory": "新记忆内容"}}
}}
```
有四种关系，包含、部分包含、不包含、不一致，说明如下：
- 包含：待检查记忆完全包含在已有记忆中，相关的所有已有记忆ID保存在related_memories字段，new_memory置空。
- 部分包含：待检查记忆只有一部分包含在已有记忆中，需要从待检查记忆中抽取未被包含的记忆内容保持到new_memory字段，相关的已有记忆ID保存在related_memories字段。
- 不包含：已有记忆中完全不存在待检查记忆，related_memories和new_memory置空。
- 不一致：已有记忆中存在相关记忆，但是内容不一致，需要更新已有记忆，需要更新的已有记忆ID保存在related_memories字段，更新之后的记忆内容保持到new_memory字段。
"""

class MemoryManager:
    """记忆管理类，提供记忆的存储和向量检索功能"""
    
    def __init__(self, db_path: str = "", embedding_model: str = "text-embedding-v1",memory_llm: Optional[Union[Dict, BaseChatModel]] = None, user_info: str = ""):
        """
        初始化记忆管理器
        
        Args:
            db_path: SQLite数据库路径
            embedding_model: 使用的嵌入模型名称
            memory_llm: 记忆LLM
            user_info: 用户信息
        """
        if not db_path:
            workspace=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'workspace')
            if not os.path.exists(workspace):
                os.makedirs(workspace)
            db_path = os.path.join(workspace, 'memory.db')
            logger.info(f'Using default db path: {db_path}')
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.user_info = user_info
        self._init_db()
        
        # 初始化文本分割器，用于将记忆内容分块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        # 初始化嵌入模型
        self.embeddings = DashScopeEmbeddings(
            model=embedding_model,
            dashscope_api_key=os.getenv('DASHSCOPE_API_KEY', '')
        )
        
        # 初始化记忆LLM
        if not memory_llm:
            memory_llm={
                'model':'qwen2.5-7b-instruct',
                'generate_cfg': {
                    'top_p': 1.0,
                    'temperature':0.0
                }
            }
            self.memory_llm=get_chat_model(memory_llm)
            logger.info(f'Using default memory llm: {memory_llm}')
        elif isinstance(memory_llm, dict):
            self.memory_llm = get_chat_model(memory_llm)
        else:
            self.memory_llm = memory_llm
        
        
    
    def _init_db(self):
        """初始化SQLite数据库，创建记忆表"""
        logger.info(f'Initializing database at {self.db_path}')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建记忆表，包含ID、记忆时间、记忆内容
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            content TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_memory(self, content: str, timestamp: str = None) -> int:
        """
        向数据库中插入一条记忆记录
        
        Args:
            content: 记忆内容
            timestamp: 记忆时间，如果不提供则使用当前时间
            
        Returns:
            int: 记忆ID
        """
        if not timestamp:
            timestamp = datetime.now().isoformat()
        
        logger.info(f'Inserting memory: {content} at {timestamp}')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO memories (timestamp, content) VALUES (?, ?)",
            (timestamp, content)
        )
        
        # 获取自增ID
        memory_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return memory_id
    
    def get_all_memories(self) -> List[Tuple[int, str, str]]:
        """
        获取所有记忆记录
        
        Returns:
            List[Tuple[int, str, str]]: 记忆记录列表，每条记录包含ID、时间戳和内容
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, timestamp, content FROM memories")
        memories = cursor.fetchall()
        
        conn.close()
        
        return memories
    
    def vector_search(self, query: str, top_n: int = 5) -> List[Tuple[Document, float]]:
        """
        向量查询，检索与查询串最相关的记忆记录
        
        Args:
            query: 查询串
            top_n: 返回的最相关记忆数量
            
        Returns:
            List[Tuple[Document, float]]: 相关记忆及其相似度分数列表
        """
        # 获取所有记忆
        memories = self.get_all_memories()
        
        if not memories:
            return []
        
        # 将记忆内容转换为Document对象并分块
        all_chunks = []
        for memory_id, timestamp, content in memories:
            chunks = self.text_splitter.split_text(content)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "memory_id": memory_id,
                        "timestamp": timestamp
                    }
                )
                all_chunks.append(doc)
        
        if not all_chunks:
            return []
        
        # 构建FAISS向量数据库
        db = FAISS.from_documents(all_chunks, self.embeddings)
        
        # 执行相似度搜索
        results = db.similarity_search_with_score(query, k=min(top_n, len(all_chunks)))
        
        return results
    
    def delete_memory(self, memory_ids: List[int]):
        """
        删除指定ID的记忆记录
        
        Args:
            memory_ids: 记忆ID列表
        """
        if not memory_ids:
            return
        logger.info(f'Deleting memories: {memory_ids}')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 删除记忆记录
        cursor.executemany("DELETE FROM memories WHERE id = ?", [(id,) for id in memory_ids])
        
        conn.commit()
        conn.close()
    
    def prepend_memory_info_to_sys(self, messages: List[Message]) -> List[Message]:
        if not messages or messages[-1].role != USER:
            return messages
        # 获取记忆，并追加到系统提示词中
        messages = copy.deepcopy(messages)
        query=format_as_text_message(messages[-1],False).content
        results = self.vector_search(query)
        memory = '\n'.join([f'M{doc.metadata["memory_id"]}@{doc.metadata["timestamp"][:10]}: {doc.page_content}' for doc, _ in results])
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sys_memory_prompt = MEMORY_PROMPT.format(memory=memory, current_time=current_time, user_info=self.user_info)
        if messages[0].role == SYSTEM:
            if isinstance(messages[0].content, str):
                messages[0].content += '\n\n' + sys_memory_prompt
            else:
                assert isinstance(messages[0].content, list)
                messages[0].content += [ContentItem(text='\n\n' + sys_memory_prompt)]
        else:
            messages = [Message(role=SYSTEM, content=sys_memory_prompt)] + messages
        # 更新记忆
        self._update_memory(messages,memory)
        return messages
    
    def _update_memory(self, messages: List[Message], old_memory: str = ''):
        i=len(messages)-2
        while i>=0:
            if messages[i].role in (USER, SYSTEM):
                last_round_dialogue=messages[i+1:]
                break
            i-=1
        else:
            last_round_dialogue=messages
        dialogue="\n".join([f'{msg.role}: {format_as_text_message(msg, False).content}' for msg in last_round_dialogue if msg.role==USER or msg.role==ASSISTANT and not msg.function_call])
        new_messages = [{
            'role': 'user',
            'content': EXTRACT_MEMORY_PROMPT.format(dialogue=dialogue)
        }]
        *_, last=self.memory_llm.chat(new_messages)
        new_memory = last[-1]['content']
        if len(new_memory.strip())<5:
            return
        if not old_memory or not old_memory.strip():
            self.insert_memory(new_memory)
            return
        new_messages = [{
            'role': 'user',
            'content': CHECK_MEMORY_PROMPT.format(old_memory=old_memory, new_memory=new_memory)
        }]
        *_, last=self.memory_llm.chat(new_messages)
        check_result = last[-1]['content'].strip()
        if check_result.startswith("```json"):
            check_result = check_result[7:]
        if check_result.endswith("```"):
            check_result = check_result[:-3]
        try:
            check_result = json.loads(check_result.strip())
        except ValueError:
            logger.warning(f'Invalid check result: {check_result}')
            return
        if check_result['result']['relation'] == '包含':
            return
        if check_result['result']['relation'] == '部分包含':
            new_memory = check_result['result']['new_memory']
        if check_result['result']['relation'] == '不一致':
            related_memories = check_result['result']['related_memories']
            parsed_mem_ids=[self._parse_mem_id(mem_id) for mem_id in related_memories]
            parsed_mem_ids=[mem_id for mem_id in parsed_mem_ids if mem_id is not None]
            self.delete_memory(parsed_mem_ids)
        if new_memory:
            self.insert_memory(new_memory)
    
    def _parse_mem_id(self, mem_id: str) -> int:
        try:
            if mem_id.startswith('M'):
                mem_id=mem_id[1:]
            if '@' in mem_id:
                mem_id=mem_id.split('@')[0].strip()
            return int(mem_id)
        except ValueError:
            logger.warning(f'Invalid memory ID: {mem_id}')
            return None



# 使用示例
if __name__ == "__main__":
    # 初始化记忆管理器
    memory_manager = MemoryManager()
    
    # 插入一些测试记忆
    memory_manager.insert_memory("巴黎是法国的首都，位于塞纳河畔，以埃菲尔铁塔和卢浮宫闻名。")
    memory_manager.insert_memory("机器学习是人工智能的一个分支，主要研究计算机如何从数据中学习模式。")
    memory_manager.insert_memory("FAISS是Facebook开发的向量数据库库，支持高效的相似性搜索和聚类。")
    memory_manager.insert_memory("Transformer模型由Google在2017年提出，采用自注意力机制处理序列数据。")
    memory_manager.insert_memory("用户的女儿叫小满，今年4岁。")
    
    # 执行向量查询
    query = "FAISS是由谁开发的？"
    results = memory_manager.vector_search(query, top_n=3)
    
    print(f"查询: {query}")
    print("相关记忆:")
    for doc, score in results:
        print(f"- 内容: {doc.page_content}")
        print(f"  相似度分数: {score}")
        print(f"  记忆ID: {doc.metadata['memory_id']}")
        print(f"  时间戳: {doc.metadata['timestamp']}")
        print()
