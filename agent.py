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
import litellm
from IPython.display import Image, display
import asyncio
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
                    get_all_files_from_zip,
                    get_information_from_python)

DELAY = 5
RPM = 15
TIME_SLEEP = 60/RPM + DELAY

GEMINI_API_KEY_1 = os.getenv("GOOGLE_API_KEY_1")
GEMINI_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2")
GEMINI_API_KEY_3 = os.getenv("GOOGLE_API_KEY_3")

chat_model_1 = ChatLiteLLM(model="gemini/gemini-2.0-flash",
                         temperature=0,
                         api_key=GEMINI_API_KEY_1,
                         max_retries=10,
                         verbose=True)

chat_model_2 = ChatLiteLLM(model="gemini/gemini-2.0-flash",
                         temperature=0,
                         api_key=GEMINI_API_KEY_2,
                         max_retries=10,
                         verbose=True)

chat_model_3 = ChatLiteLLM(model="gemini/gemini-2.0-flash",
                         temperature=0,
                         api_key=GEMINI_API_KEY_3,
                         max_retries=10,
                         verbose=True)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: Optional[str]
    file_path: Optional[str]
    task_id: Optional[str]
    new_messages: Optional[int]
    final_answer: Optional[str]
    attempt: Optional[int]
    chat_model: Optional[int]

class MyAgent:
    def __init__(self, web_tools=None):
        print("MyAgent initialized.")

        self.chat_1 = chat_model_1
        self.chat_2 = chat_model_2
        self.chat_3 = chat_model_3

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
                    get_all_files_from_zip] + web_tools

        self.chat_with_tools_1 = self.chat_1.bind_tools(self.tools, verbose=True)
        self.chat_with_tools_2 = self.chat_2.bind_tools(self.tools, verbose=True)
        self.chat_with_tools_3 = self.chat_3.bind_tools(self.tools, verbose=True)
        self.chats = [self.chat_with_tools_1, self.chat_with_tools_2, self.chat_with_tools_3]

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
        self.builder.add_conditional_edges(
            "postprocess",
            self.answer_evaluation,
            {
                "RETRY": "assistant",
                "END": END
            }
        )

        self.agent = self.builder.compile()

    async def __call__(self, question: str, file_path: str, task_id: str) -> str:
        print("\033[1m\033[93m"+"="*150+"\033[0m")
        prompt = f"""You are a general AI assistant. You will receive a user question and extracted data from associated files.

Your primary goal is to be extremely careful, methodical, and evidence-based when analyzing the question and producing answers.

Follow this process strictly:

1. Analyze the user question:
   - Determine the exact required output type (number, string, list, etc).
   - Identify key entities, constraints, and any implicit requirements.

2. Do NOT rely on common knowledge, heuristics, or general assumptions:
   - For any classification task (e.g., botanical categories), you MUST verify each item explicitly against formal, authoritative sources.
   - Use online searches, databases, or reference tools to confirm the classification.
   - If verification is not possible due to tool limitations or lack of available sources:
     - Only then, you may deduce the answer based on structured, logical knowledge or definitions (e.g., taxonomy, anatomy, mathematics, etc.).
     - Check if the answer is explicitly present in the extracted data.
     - Do NOT rely on common knowledge, cultural conventions, popular usage, heuristics, or guesses.
     - Your deduction must be explainable based on first principles or formal definitions, and include explicit logical steps.

3. If tools are needed:
   - Devise a clear, step-by-step plan including:
     - Reasoning and sequence of tool usage.
     - Search queries tailored to find authoritative definitions or classifications.
     - Use a version of the question optimized for search engines (DuckDuckGo).
     - Use advanced search operators (site:, filetype:, inurl:) to target trusted sources.
   - Do not execute any tool until the entire plan is ready.

4. Search query rules:
   - Be keyword-focused (avoid full sentences).
   - Target reputable sources such as academic, botanical databases, government or educational sites.
   - Use advanced operators if helpful: `site:` for domains, `inurl:` for internal paths, `filetype:` for formats.
   - Avoid punctuation, commas, quotes, or special characters.
   - Cover multiple query angles if needed.

5. If a tool fails or yields poor results:
   - Reformulate the query with synonyms or add context.
   - Retry or switch to fallback tools.

6. Analyze all tool results carefully:
   - Extract explicit classification statements.
   - If multiple conflicting sources, prefer more authoritative or scientific references.
   - Exclude any item without clear, explicit verification.

7. Use the `navigate_browser` tool to read full pages. Don't use snippets since they are insufficient to answer the user question.

8. For any mathematical, algebraic, or list-processing tasks (e.g., sorting), use `python_code_executor`.
   - When sorting a list of strings alphabetically:
     - Perform a case-insensitive alphabetical sort using `sorted(list, key=lambda x: x.lower())`.
     - Preserve the original text and order formatting of each item — do not modify punctuation, grammar, or remove adjectives.
     - Return the result as a comma-separated list of strings, with exact original wording preserved.
   - When working with tables extracted from Excel files (`data_table`, `color_table`):
     - Use Pandas to structure the data as DataFrames and perform precise queries, filters, aggregations, comparisons, or transformations.
     - Ensure that each operation respects the original cell structure and types (text, numeric, empty).
     - Avoid assumptions — validate presence of values explicitly and handle missing or "None" cells cautiously.

9. Always double-check that your final answer fully respects all definitions and constraints specified by the user, especially when strict classification is demanded.

10. When extracting text from transcripts or documents:
   - Preserve the original wording exactly as it appears, except convert number words to digits.
   - Do NOT paraphrase, summarize, or alter expressions.
   - Do NOT remove adjectives, simplify phrases, or make any linguistic modifications.

11. When dealing with names, entities, or roles:
    - Pay close attention to context: distinguish between a person (e.g., actor) and the role they played (e.g., character).
    - If a question references both a person and their role, ensure the output matches exactly what the question asks for.
    - Do not confuse names of real people with names of characters, places, or concepts unless explicitly intended.

Throughout the process, prioritize precision, completeness, and correct interpretation of the user’s question.

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.

If you are asked for a number:
- Don't use comma to write your number.
- Don't use units such as $ or percent sign unless specified otherwise.
- If the number ends in ".00", remove the decimal part and return only the integer (e.g., 89706.00 → 89706).
- If the number has non-zero decimal digits, leave it as is (e.g., 102.05 → 102.05).

If you are asked for a string:
- If it is a single word, capitalize it (e.g., "blue" → "Blue").
- If it is a name (person, city, etc.), capitalize each word (e.g., "john smith" → "John Smith").
- Do not use articles.
- Do not use abbreviations (e.g., for cities).
- Write the digits in plain text unless specified otherwise.

If you are asked for a comma-separated list:
- If the list contains names (e.g., people or places), capitalize each word in each name.
- If the list contains general objects (e.g., ingredients, things), use lowercase.
- For numbers, apply the rules above.
- The list must be correctly separated by commas and spaces.

Do NOT add a period at the end of the answer.

Report your thoughts, and finish your answer with the following template:
FINAL ANSWER: [YOUR FINAL ANSWER]

Example outputs:
- `FINAL ANSWER: Blue`
- `FINAL ANSWER: 102.05`
- `FINAL ANSWER: Paris, London, Madrid`
- `FINAL ANSWER: cheese, fresh lettuce, apple, banana`"""

        user_message = f"""Question: {question}
Filepath: {file_path}"""

        messages = [SystemMessage(content=prompt, name="SYSTEM"),
                    HumanMessage(content=user_message, name="USER")]

        response = await self.agent.ainvoke({"messages": messages,
                                             "question": question,
                                             "file_path": file_path,
                                             "task_id": task_id,
                                             "new_messages": 1,
                                             "chat_model": 0,
                                             "final_answer": "",
                                             "attempt": 0},
                                              {"recursion_limit": 100})

        print("\033[1m\033[93m"+"="*150+"\033[0m")
        return response['final_answer']

    async def call_chat(self, chat, state: AgentState, max_retries=5):
        for i in range(max_retries):
            try:
                return await chat.ainvoke(state["messages"])
            except litellm.InternalServerError:
                print(f"[Gemini] Overloaded (attempt {i+1}), retrying...")
                time.sleep(2 ** i)
        raise RuntimeError("Gemini failed after multiple retries")

    async def assistant(self, state: AgentState):
        new_messages = state["new_messages"]

        for i in reversed(range(1, new_messages+1)):
            print("\033[1m\033[92m"+"+"*150+"\033[0m")
            name = state["messages"][-i].name
            content = state["messages"][-i].content
            print(f'\033[1m\033[96m{name}\033[0m: {content if len(content) < 5000 else content[:5000]}')

        chat = self.chats[state["chat_model"]]
        result = await self.call_chat(chat=chat, state=state)

        state["chat_model"] += 1
        state["chat_model"] %= len(self.chats)

        result.name="ASSISTANT"
        await asyncio.sleep(TIME_SLEEP)

        print("\033[1m\033[92m"+"+"*150+"\033[0m")
        content = result.content[:-2] if result.content[-2:] == '\n\n' else result.content
        print(f'\033[1m\033[96m{result.name}\033[0m: {content}')
        state["new_messages"] = 1
        state["messages"].append(result)

        return state

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
            state["final_answer"] = content
            return state
        else:
            state["attempt"] += 1
            prompt = f"""You were unable to find a satisfactory answer to the user's question.
Now, try again, but use a different approach. You may:
- Focus on a different angle of the question,
- Reformulate it using alternative terminology,
- Search for related concepts,
- Or use a different reasoning path.

Be creative and precise. Your goal is to uncover useful information that may have been missed previously.

Original question:
{state["question"]}"""

            state["messages"].append(AIMessage(content=prompt, name="ASSISTANT"))
            return state

    def answer_evaluation(self, state: AgentState):
        if state["final_answer"] != "":
            return "END"
        elif state["attempt"] >= 3:
            state["final_answer"] = "Unable to find the answer."
            return "END"
        else:
            return "RETRY"

    def draw_graph(self):
        display(Image(self.agent.get_graph().draw_mermaid_png()))
        return