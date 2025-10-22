# 🧠 LangChain Tools (2025)

A practical **LangChain project** demonstrating how to integrate and use various tools — including **DuckDuckGo Search**, **Shell commands**, and **custom-built tools** — within **LLM workflows**.  

This project also shows how to:
- Create custom tools using the `@tool` decorator  
- Inspect tool metadata  
- Invoke tools programmatically  
- Build intelligent, automated AI pipelines  

---

## 🚀 Features
- 🧩 Integration with built-in LangChain tools  
- ⚙️ Custom tool creation using decorators  
- 🤖 Model–tool binding for dynamic responses  
- 🌦️ Real-world API integration example (Weather API)  
- 🔁 Execution of model-triggered tool calls  

---

## 🧩 Most Popular Inbuilt LangChain Tools (2025)

| Tool Name | Description | Use Case |
|------------|-------------|----------|
| **DuckDuckGoSearchRun** | Searches the web using DuckDuckGo | Real-time search retrieval |
| **SerpAPITool** | Uses SerpAPI for web search | Enhanced Google-like search |
| **WikipediaQueryRun** | Fetches summaries from Wikipedia | General knowledge lookup |
| **RequestsGetTool** | Performs HTTP GET requests | Fetch public APIs or REST data |
| **RequestsPostTool** | Performs HTTP POST requests | Submit data or API forms |
| **PythonREPLTool** | Executes Python code safely | Run calculations or logic |
| **TerminalTool** | Executes shell commands (sandboxed) | System-level operations |
| **SQLDatabaseToolkit** | Interacts with SQL databases | Query structured data |
| **VectorStoreRetrieverTool** | Retrieves documents from embeddings | RAG (Retrieval-Augmented Generation) |
| **GoogleSearchTool** | Uses Google Custom Search | Business or market insights |

---

## 💻 Example: Custom Tool Integration

Below is a complete example demonstrating how to create a **custom weather tool**, bind it to an **LLM**, and invoke it intelligently.

```python
from langchain.chat_models import ChatOpenAI
from langchain_core.tools import tool
import requests

# 1️⃣ Define the tool
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city (via weather API)."""
    api_key = "YOUR_API_KEY"  # Replace with your actual API key
    url = "https://api.weatherapi.com/v1/current.json"
    params = {"key": api_key, "q": city}

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    temp_c = data["current"]["temp_c"]
    cond = data["current"]["condition"]["text"]
    return f"The current temperature in {city} is {temp_c}°C and the condition is {cond}."

# 2️⃣ Initialize model & bind tool
model = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
model_with_tools = model.bind_tools([get_weather])

# 3️⃣ Ask the model — it may call the tool
user_input = "What's the weather in London right now?"
response = model_with_tools.invoke({"messages": [{"role": "user", "content": user_input}]})

# 4️⃣ Execute tool if model requested it
if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_args = tool_call["args"]
    result = get_weather.invoke(tool_args)
    print("🔧 Tool Result:", result)
else:
    print("💬 Model Answer:", response.content)
