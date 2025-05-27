
import os
import time
from langchain.tools import Tool, tool
from typing import Tuple, List
from typing_extensions import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_litellm import ChatLiteLLM
from IPython.display import Image, display
from tools import (search_tool,
                   download_tool,
                   get_web_page,
                   add,
                   subtract,
                   multiply,
                   divide,
                   power,
                   square_root,
                   get_information_from_wikipedia,
                   get_information_from_arxiv,
                   get_information_from_youtube,
                   python_tool,
                   get_information_from_json,
                   get_information_from_audio,
                   get_information_from_xml,
                   get_information_from_docx,
                   get_information_from_txt,
                   get_information_from_pdf,
                   get_information_from_csv,
                   get_information_from_excel,
                   get_information_from_pdb,
                   get_information_from_image,
                   get_information_from_pptx,
                   get_all_files_from_zip)

DELAY = 5
TIME_SLEEP = 60/15 + DELAY

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_1")

chat_model = ChatLiteLLM(model="gemini/gemini-2.0-flash",
                         temperature=0.1,
                         api_key=GEMINI_API_KEY,
                         max_retries=10,
                         verbose=True)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: Optional[str]
    file_path: Optional[str]
    task_id: Optional[str]
    new_messages: Optional[int]

class MyAgent:
    def __init__(self, web_tools=None):
        print("MyAgent initialized.")

        self.chat = chat_model

        self.tools = [search_tool,
                    download_tool,
                    get_web_page,
                    add,
                    subtract,
                    multiply,
                    divide,
                    power,
                    square_root,
                    get_information_from_wikipedia,
                    get_information_from_arxiv,
                    get_information_from_youtube,
                    python_tool,
                    get_information_from_json,
                    get_information_from_audio,
                    get_information_from_xml,
                    get_information_from_docx,
                    get_information_from_txt,
                    get_information_from_pdf,
                    get_information_from_csv,
                    get_information_from_excel,
                    get_information_from_pdb,
                    get_information_from_image,
                    get_information_from_pptx,
                    get_all_files_from_zip
                    ] + web_tools

        self.chat_with_tools = self.chat.bind_tools(self.tools, verbose=True)

        self.builder = StateGraph(AgentState)
        self.builder.add_node("assistant", self.assistant)
        self.builder.add_node("tools", ToolNode(self.tools))
        self.builder.add_node("extract_data_from_file", self.extract_data_from_file)
        self.builder.add_node("postprocess", self.postprocess)

        self.builder.add_edge(START, "extract_data_from_file")
        self.builder.add_edge("extract_data_from_file", "assistant")
        self.builder.add_conditional_edges(
            "assistant",
            self.assistant_router,
            {
                "tools": "tools",
                "postprocess": "postprocess"
            }
        )
        self.builder.add_edge("tools", "assistant")

        self.agent = self.builder.compile()

    async def __call__(self, question: str, file_path: str, task_id: str) -> str:
        print("\033[1m\033[93m"+"="*150+"\033[0m")
        print(f"QUESTION: {question}")
        print(f"File: {file_path}")
        prompt = f"""You are a general AI assistant. I will ask you a question and I give you the extracted data from associate files to the user question.

You must follow this process:
1. Analyze the user question to identify the required output type (e.g., number, string, list) and key concepts.
2. BEFORE planning or using any tool, determine if the answer can be obtained directly from your own knowledge or reasoning. You can't guest the answer.
   - If yes, answer directly without using any tools.
   - If not, proceed with tool-based planning as described below.
3. If tools are needed, generate a plan that includes:
   - The approach to solve the question.
   - Which tools to use and in what order.
   - How to reformulate the query if needed for web search.
   - Ensure that search queries do not contain punctuation marks, commas, quotes, or special characters.
4. Do not execute any tool until the plan is complete.
5. Follow the plan exactly. If a tool fails (e.g., due to network or no results), pause, replan, and retry with:
   - A reformulated query (more specific or with synonyms/context).
   - A fallback tool if available.
6. Evaluate tool results for relevance. If multiple source links are available, explore each one using `navigate_browser` to gather all relevant information.

Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

        user_message = f"Question: {question}\nFilepath: {file_path}"

        messages = [SystemMessage(content=prompt, name="SYSTEM"),
                    HumanMessage(content=user_message, name="USER")]

        response = await self.agent.ainvoke({"messages": messages,
                                             "question": question,
                                             "file_path": file_path,
                                             "task_id": task_id,
                                             "new_messages": 1},
                                              {"recursion_limit": 100})

        print("\033[1m\033[93m"+"="*150+"\033[0m")
        return response['messages'][-1].content

    async def assistant(self, state: AgentState):
        new_messages = state["new_messages"]

        for i in reversed(range(1, new_messages+1)):
            print("\033[1m\033[92m"+"+"*150+"\033[0m")
            name = state["messages"][-i].name
            content = state["messages"][-i].content
            print(f'\033[1m\033[96m{name}\033[0m: {content if len(content) < 5000 else content[:5000]}')

        result = await self.chat_with_tools.ainvoke(state["messages"])
        result.name="ASSISTANT"
        time.sleep(TIME_SLEEP)

        print("\033[1m\033[92m"+"+"*150+"\033[0m")
        content = result.content[:-2] if result.content[-2:] == '\n\n' else result.content
        print(f'\033[1m\033[96m{result.name}\033[0m: {content}')
        new_messages = 1

        return {"messages": result, "new_messages": new_messages}

    def extract_data_from_file(self, state: AgentState) -> str:
        path = state["file_path"]
        new_messages = state["new_messages"]
        prompt = ""
        messages = []

        if path and "." in path:
            ext = path.strip().split(".")[-1].lower()
            print(f"Extension detected: {ext}")

            if ext == "zip":
                files, prompt = get_all_files_from_zip(path)
                name = "get_all_file_from_zip"
                messages.append(AIMessage(content=prompt, name=name))
            else:
                files = [path]

            for file_path in files:
                ext = file_path.strip().split(".")[-1].lower()
                print(f"Extension detected: {ext}")

                prompt = f"Information extracted from {file_path}.\n\n"
                match ext:
                    case "csv":
                        content = get_information_from_csv.invoke(file_path)
                        name = "get_information_from_csv"
                    case "txt":
                        content = get_information_from_txt.invoke(file_path)
                        name = "get_information_from_txt"
                    case "pdf":
                        content = get_information_from_pdf.invoke(file_path)
                        name = "get_information_from_pdf"
                    case "json":
                        content = get_information_from_json.invoke(file_path)
                        name = "get_information_from_json"
                    case "jsonld":
                        content = get_information_from_json.invoke(file_path)
                        name = "get_information_from_json"
                    case "xml":
                        content = get_information_from_xml.invoke(file_path)
                        name = "get_information_from_xml"
                    case "pdb":
                        content = get_information_from_pdb.invoke(file_path)
                        name = "get_information_from_pdb"
                    case "mp3":
                        content = get_information_from_audio.invoke(file_path)
                        name = "get_information_from_audio"
                    case "m4a":
                        content = get_information_from_audio.invoke(file_path)
                        name = "get_information_from_audio"
                    case "docx":
                        content = get_information_from_docx.invoke(file_path)
                        name = "get_information_from_docx"
                    case "xlsx":
                        content = get_information_from_excel.invoke(file_path)
                        name = "get_information_from_excel"
                    case "xls":
                        content = get_information_from_excel.invoke(file_path)
                        name = "get_information_from_excel"
                    case "png":
                        content = get_information_from_image.invoke({"file_path": file_path, "question": state["question"]})
                        name = "get_information_from_image"
                    case "jpg":
                        content = get_information_from_image.invoke({"file_path": file_path, "question": state["question"]})
                        name = "get_information_from_image"
                    case "py":
                        content = get_information_from_python.invoke(file_path)
                        name = "get_information_from_python"
                    case "pptx":
                        content = get_information_from_pptx.invoke(file_path)
                        name = "get_information_from_pptx"
                    case _:
                        content = "Try to use some available tool to answer the user question."
                        name = "handle_no_file"
                prompt += f"{content}"
                messages.append(AIMessage(content=prompt, name=name))
                new_messages += 1
        else:
            prompt = "The question doesn't have an attached file."
            name = "handle_no_file"

        return {"messages": messages, "new_messages": new_messages}

    def assistant_router(self, state: AgentState) -> str:
        tool_decision = tools_condition(state)
        if tool_decision == "tools":
            return "tools"
        else:
            return "postprocess"

    def postprocess(self, state: AgentState) -> AgentState:
        last_msg = state["messages"][-1]
        content = last_msg.content
        index = content.find("FINAL ANSWER: ")
        if index != -1:
            content = content[index+len("FINAL ANSWER: "):].replace("\n", "")
        else:
            content = "Unable to find the answer"

        state["messages"][-1].content = content
        return state

    def draw_graph(self):
        display(Image(self.agent.get_graph().draw_mermaid_png()))
        return
