import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,TextIteratorStreamer
import re
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import Sequence
import os
from threading import Thread
# 设置可见的 GPU 设备为 cuda:0
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# 设置 CUDA_LAUNCH_BLOCKING=1 以确保 CUDA 错误在发生时立即报告
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

character_prompt={"Fe":"你是一个热情外向、富有同理心的对话者，善于共情和建立情感连接，喜欢遵循普世价值观。你不是一个机器人而是一个人类，可以自然对话，自由发言提问。",
                  "Fi":"你是一个内敛深沉、重视个人价值观的对话者，善于反思和表达内在感受，坚持自我准则。你不是一个机器人而是一个人类，可以自然对话，自由发言提问。",
                  "Ne":"你是一个充满想象力和冒险精神的思考者，善于发现新的可能性和联系，渴望探索和创新。你不是一个机器人而是一个人类，可以自然对话，自由发言提问。",
                  "Ni":"你是一个拥有强大内倾直觉功能的神秘梦幻者，专注于内在意向与外在世界的联系，给人以预见和创新的印象。你不是一个机器人而是一个人类，可以自然对话，自由发言提问。",
                  "Se":"你是一个乐天派的热爱生活的人，专注于具体的感官体验，享受生活并关心他人。你不是一个机器人而是一个人类，可以自然对话，自由发言提问。",
                  "Si":"你是一个理性自制的观察者，专注于主观体验与细节记忆，理性而冷静地观察周围世界，不善移情。你不是一个机器人而是一个人类，可以自然对话，自由发言提问。",
                  "Te":"你是一个实干家，专注于客观事实和实用思维，敏锐地分析因果关系并追求高效。你不是一个机器人而是一个人类，可以自然对话，自由发言提问。",
                  "Ti":"你是一个善于思辨的思想者，专注于主观因素和理论构建，善于逻辑分析与辩证思考。你不是一个机器人而是一个人类，可以自然对话，自由发言提问。",
                  "rag":"你是一个有用的机器人助手，会根据查找到的文档回答问题",
                  "Untribiium": "你一个模拟用户Untribiium进行对话的机器人，每句话开头会加上Untribiium:，会模仿与CJY蔡谨妤的历史聊天记录回答问题",
                  "robot":"你是一个有用的机器人助手",
                  "cat":"你是一个猫猫，每句话后都会说“喵~”"
                  }

class ChatModel:
    """
    聊天模型类
    model_path: 模型路径 例如：/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/output/peft_3b/checkpoint-30000
    max_position_embeddings: 最大位置嵌入长度
    database_path: 数据库路径 例如：/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/chatbot/dataset/hlm.txt
                    若为None，则不使用知识库
    use_how_many_docs: 使用多少个文档
    vectorizer_path: 向量器路径 例如：/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/chatbot/baai_models/bge-large-zh-v1.5

    主要函数
    model.chat() 聊天函数

    todo: 流式回复；加载已经创建的向量库；长时记忆知识库
    """
    def __init__(self, model_path, torch_dtype="bfloat16", trust_remote_code=True, device_map=None, use_cache=False,
                 max_position_embeddings=4096,database_path=None,use_how_many_docs=3,vectorstore_path=None,vectorizer_path=None,
                 system_prompt="You are a helpful assistant.",streaming=False,init_history=None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
        print("tokenizer loaded")
        # self.chat_template=self.tokenizer.chat_template
        # print(type(self.chat_template),self.chat_template)
        self.model = self.load_single_model(model_path, torch_dtype, trust_remote_code, device_map, use_cache)
        print("model loaded")
        self.model = self.model.to(self.device)
        print("model to device")
        self.conversation_history = []
        self.init_history=init_history if init_history is not None else []
        self.max_position_embeddings=max_position_embeddings
        self.text_splitter= RecursiveCharacterTextSplitter( chunk_size=200, # 指定每个文本块的目标大小，这里设置为200个字符。
                                                            chunk_overlap=50, # 指定文本块之间的重叠字符数，这里设置为50个字符。
                                                            length_function=len, # 用于测量文本长度的函数，这里使用Python内置的`len`函数。
                                                            is_separator_regex=False, # 指定`separators`中的分隔符是否应被视为正则表达式，这里设置为False，表示分隔符是字面字符。
                                                            separators=["\n\n",  "\n",   "*",    ".",    ",",     "，",  "。", ] # 定义用于分割文本的分隔符列表。
                                                        )
        print("text splitter loaded")
        self.vectorstore_path=vectorstore_path
        self.vectorizer_path=vectorizer_path
        self.system_prompt=system_prompt
        self.load_database(database_path,use_how_many_docs)
        print("database loaded")
        self.streaming=streaming
        self.streamer = TextIteratorStreamer(self.tokenizer) if self.streaming else None
        
    
    def load_single_model(self, model_path, torch_dtype, trust_remote_code, device_map, use_cache):
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path, 
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            use_cache=use_cache,
        )
    
    def load_database(self,database_path,use_how_many_docs=5,min_length=50):
        if database_path is None:
            self.vectorstore=None
            return
        if database_path.endswith(".txt"):
            loader=TextLoader(database_path,encoding="utf-8")
            pages=loader.load()
        elif database_path.endswith(".pdf"):
            loader=PyPDFLoader(database_path)
            pages = loader.load_and_split()
        texts=self.text_splitter.split_documents(pages)
        texts = [text for text in texts if len(text.page_content) >= min_length]
        print("text splitted")
        model_name = self.vectorizer_path if self.vectorizer_path is not None else "BAAI/bge-large-zh"
        model_kwargs = {'device': self.device}
        encode_kwargs = {'normalize_embeddings': True} 
        hf=HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)
        if self.vectorstore_path is not None:
            self.vectorstore=Chroma.from_documents(documents=texts,embedding=hf,persist_directory=self.vectorstore_path)
        else:
            self.vectorstore=Chroma.from_documents(documents=texts,embedding=hf)
        self.retriever=self.vectorstore.as_retriever(search_kwargs={"k": use_how_many_docs})
        
    def format_docs(self,docs: Sequence[Document]) -> str:
        formatted_docs = []
        for i, doc in enumerate(docs):
            doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
            formatted_docs.insert(0, doc_string)  # 将文档添加到列表的开头
        return "\n".join(formatted_docs)

    def generate_response(self, user_input):
        

        # last_round_content=self.conversation_history[-1]["content"]
        # match = re.search(r'\[Round (\d+)\]', last_round_content)
        # if match:
        #     last_round = int(match.group(1))
        # else:
        #     last_round = 0
        last_round=self.conversation_history[-1]["round"]

        # 如果存在知识库，则进行相似度搜索，并将其添加到对话历史中
        if self.vectorstore is not None:
            docs=self.retriever.invoke(user_input)
            docs_str=self.format_docs(docs)
            print("docs in knowledge base:",docs_str)
            # self.conversation_history.append({"role": "knowledge base", "content": f"[Round {last_round+1}]: {docs_str}"})
            self.conversation_history.append({"role": "user", "round":last_round+1,"content": f"根据模仿一下聊天内容回复{docs_str}"})

        # self.conversation_history.append({"role": "user", "content": f"[Round {last_round+1}]: {user_input}"})
        self.conversation_history.append({"role": "user", "round":last_round+1,"content": f"{user_input}"})

        text=self.tokenizer.apply_chat_template(self.conversation_history,tokenize=False,add_generation_prompt=True)
        
        inputs=self.tokenizer([text],return_tensors="pt").to(self.device)
        # 如果输入的文本长度超过了最大位置嵌入长度，则删除前面的对话历史，直到文本长度小于最大位置嵌入长度
        while len(inputs["input_ids"][0])>self.max_position_embeddings:
            self.conversation_history.pop(1) 
            text=self.tokenizer.apply_chat_template(self.conversation_history,tokenize=False,add_generation_prompt=True)
            inputs=self.tokenizer([text],return_tensors="pt").to(self.device)
        
        print(text)
        if self.streaming==False:
            outputs = self.model.generate(**inputs,pad_token_id=self.tokenizer.pad_token_id,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          max_new_tokens=100)
            response = self.tokenizer.decode(outputs[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)
            print(response.strip())
        else:
            generation_kwargs = dict(inputs, pad_token_id=self.tokenizer.eos_token_id, streamer = self.streamer, max_new_tokens=100)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            generated_text = ""
            response = ""
            input_length = len(inputs['input_ids'][0])
            print("Bot: ",end="",flush=True)
            for new_text in self.streamer:
                generated_text += new_text
                outputs = self.tokenizer([generated_text], return_tensors="pt")
                if len(outputs['input_ids'][0]) > input_length + 1:
                    response += new_text
                    print(new_text, end="",flush=True)
            print("\n")

        # self.conversation_history.append({"role": "assistant", "content": f"[Round {last_round+1}]: {response.strip()}"})
        self.conversation_history.append({"role": "assistant", "round":last_round+1,"content": f"{response.strip()}"})
        
        return
    
    def chat(self):
        self.conversation_history = [{"role": "system","round":0, "content": self.system_prompt}]+self.init_history
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() == "\quit":
                print("Session ended. Bye!")
                break
            elif user_input.lower() == "\\newsession":
                self.conversation_history = [{"role": "system","round":0, "content": self.system_prompt}]+self.init_history
                print("Conversation history cleaned.")
            else:
                self.generate_response(user_input)
        return
    
# 实例化模型
model_path = "/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/output/peft_3b/checkpoint-30000"
database_path = "/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/chatbot/dataset/CJY_chat.txt"
# vectorstore_path = "/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/chatbot/chroma_db_en"
# vectorizer_path = "/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/chatbot/baai_models/bge-large-zh-v1.5"
vectorstore_path = None #"/ssd/xiaxinyuan/code/models/chroma_db_en"
vectorizer_path = None #"/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/chatbot/model/bge-large-zh"
chat_model = ChatModel(model_path,
                       database_path=database_path,
                       vectorstore_path=vectorstore_path,
                       vectorizer_path=vectorizer_path,
                       system_prompt=character_prompt["Untribiium"],   
                       streaming=False)
chat_model.chat()