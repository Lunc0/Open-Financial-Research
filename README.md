# 🔬 Open Financial Research

<img width="1388" height="298" alt="full_diagram" src="https://github.com/user-attachments/assets/12a2371b-8be2-4219-9b48-90503eb43c69" />

_The diagram above illustrates the inherited Open Deep Research graph architecture. This repository extends its state and control flow rather than replacing the core graph._

Open Financial Research is a finance-oriented extension of [LangChain's Open Deep Research](https://github.com/langchain-ai/open_deep_research). It keeps the original configurable LangGraph research workflow while adding runtime user guidance, explicit research planning, structured traceability, finance-specific research behavior, and optional SEC EDGAR integration.

The project is designed for long-running research where requirements may change after execution has started. Instead of treating a follow-up message as unstructured chat history, it turns the message into tracked guidance, maps it to an explicit plan change, binds researcher execution to plan items, and exposes the resulting state transitions to clients and UIs.

<img width="817" height="666" alt="Screenshot 2025-07-13 at 11 21 12 PM" src="https://github.com/user-attachments/assets/052f2ed3-c664-4a4f-8ec2-074349dcaa3f" />

_LangGraph Studio can be used to inspect graph state, including the added guidance, plan, trace-event, and node-status fields._

## ✨ Key Extensions

| Extension | What it adds |
| --- | --- |
| **Runtime guidance loop** | Users can add, revise, narrow, or redirect requirements while a research thread is running. Guidance remains `pending` until it is reflected in the research plan. |
| **Explicit research plan** | Research work is represented by structured plan items with stable IDs, priorities, and lifecycle states. Every `ConductResearch` call must reference an executable plan item. |
| **Structured traceability** | Append-only `trace_events` record what happened, while `node_status` exposes the latest status of each graph node for progress UIs and diagnostics. |
| **Finance research profile** | Planning and report generation emphasize primary financial sources, metric/date/currency definitions, assumptions, risk scenarios, and auditability. |
| **Search and MCP coordination** | General web search remains available alongside MCP tools, with conflict-safe tool loading and graceful search fallback. |
| **SEC EDGAR integration** | Finance runs can optionally attach the free EdgarTools MCP server through `uvx`, without requiring an SEC API key. |

The three central control mechanisms are documented in [Runtime Guidance, Explicit Planning, and Structured Traceability](docs/runtime_guidance_plan_traceability.md).

## 🔄 How the Control Loop Works

```text
User follow-up
      │
      ▼
state.user_guidance (pending)
      │
      ▼ UpdateResearchPlan
state.research_plan
      │
      ▼ ConductResearch(plan_item_id=...)
Research execution
      │
      ├── trace_events: append-only history
      └── node_status: latest progress snapshot
```

The LLM interprets the request and proposes plan changes. Code-controlled reducers and validation apply those changes, reject invalid task delegation, prevent completion while guidance is still pending, and record the resulting events.

## 🚀 Quickstart

1. Clone the repository and activate a virtual environment:
```bash
git clone https://github.com/Lunc0/Open-Financial-Research.git
cd Open-Financial-Research
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
uv sync
# or
uv pip install -r pyproject.toml
```

3. Set up your `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):
```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

4. Launch agent with the LangGraph server locally:

```bash
# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

This will open the LangGraph Studio UI in your browser.

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Ask a question in the `messages` input field and click `Submit`. Select different configuration in the "Manage Assistants" tab.

## ⚙️ Configurations

#### LLM :brain:

Open Financial Research supports the model providers available through LangChain's [init_chat_model() API](https://python.langchain.com/docs/how_to/chat_models_universal_init/). It uses separate models for several stages. See [configuration.py](src/open_deep_research/configuration.py) for the complete configuration. These settings can also be changed through LangGraph Studio.

- **Summarization** (default: `openai:gpt-4.1-mini`): Summarizes search API results
- **Research** (default: `openai:gpt-4.1`): Power the search agent
- **Compression** (default: `openai:gpt-4.1`): Compresses research findings
- **Final Report Model** (default: `openai:gpt-4.1`): Write the final report

> Note: the selected model will need to support [structured outputs](https://python.langchain.com/docs/integrations/chat/) and [tool calling](https://python.langchain.com/docs/how_to/tool_calling/).

> Note: For OpenRouter: Follow [this guide](https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408) and for local models via Ollama  see [setup instructions](https://github.com/langchain-ai/open_deep_research/issues/65#issuecomment-2743586318).

#### Search API :mag:

Open Financial Research supports Tavily, provider-native web search, and MCP tools. The `search_api` and `mcp_config` fields in [configuration.py](src/open_deep_research/configuration.py) control tool selection. In the finance profile, general web search and MCP tools can be exposed together so a specialized data source does not remove broad discovery and fallback search.

`mcp_config` supports both a simple single-server form and a pass-through multi-server form for `MultiServerMCPClient`.

Single streamable HTTP server:

```json
{
  "configurable": {
    "mcp_config": {
      "url": "https://your-mcp.example.com/mcp",
      "transport": "streamable_http",
      "tools": ["company_filings", "macro_data"],
      "access_token_env": "FINANCE_MCP_TOKEN"
    }
  }
}
```

Multi-server / stdio form:

```json
{
  "configurable": {
    "mcp_config": {
      "tool_name_prefix": true,
      "servers": {
        "filings": {
          "url": "https://your-mcp.example.com/mcp",
          "transport": "streamable_http"
        },
        "local_macro": {
          "transport": "stdio",
          "command": "python",
          "args": ["servers/macro_mcp.py"]
        }
      }
    }
  }
}
```

If `tools` is omitted, all non-conflicting MCP tools are exposed. MCP tool names never override existing graph tools such as `web_search`.

For finance research, the project can auto-attach the free EdgarTools SEC EDGAR MCP server. It does not require an API key, but SEC requires automated clients to identify themselves:

```dotenv
ENABLE_SEC_EDGAR_MCP=true
EDGAR_IDENTITY=Your Name your.email@example.com
```

When `research_profile=finance` and this flag is enabled, the graph launches:

```json
{
  "transport": "stdio",
  "command": "uvx",
  "args": ["--from", "edgartools[ai]", "edgartools-mcp"],
  "env": {
    "EDGAR_IDENTITY": "Your Name your.email@example.com"
  }
}
```

The first run may download the EdgarTools MCP package through `uvx`. No SEC API key is used.

#### Other 

See [configuration.py](src/open_deep_research/configuration.py) for concurrency, model, search, MCP, finance profile, traceability, and report-generation settings.

#### Domain profile, runtime guidance, explicit planning, and traceability

This project adds the following workflow controls for finance-oriented research and more transparent runs:

- `research_profile`: choose `general` or `finance`. The finance profile biases planning, source selection, evidence handling, and report writing toward primary financial sources, explicit metric/date/currency definitions, risk scenarios, assumptions, and auditability.
- `state.user_guidance`: runtime guidance injected through graph state or checkpoint updates. Clients can append structured guidance while research is running to add or modify research direction before the next supervisor or researcher step.
- `append_runtime_guidance`: service-layer helper for chat backends. When a user sends a message while a research thread is active, call this helper with the graph, thread config, and message content to append pending guidance into `state.user_guidance`.
- LangGraph Studio Chat: after a research thread already has a `research_brief`, new user messages in the same thread are converted into pending `state.user_guidance` and routed back to the supervisor instead of rewriting the original brief.
- `state.research_plan`: structured plan items track task identity, priority, origin, and status. `UpdateResearchPlan` applies explicit changes, and `ConductResearch` must reference a valid `pending` or `running` plan item.
- `enable_traceability`: when enabled, the graph emits `trace_events` and `node_status` in state. UIs can stream or inspect these fields to show what each node is doing, which tools were called, which subtopics were delegated, and why a phase stopped.

See the [mechanism design document](docs/runtime_guidance_plan_traceability.md) for reducers, state transitions, completion guards, failure handling, and an end-to-end example.

## 📊 Evaluation

The inherited evaluation tooling supports [Deep Research Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard). The benchmark contains 100 research tasks across multiple fields and uses LLM-based evaluation against reference reports. The results below are upstream Open Deep Research results; they are retained as a baseline and should not be interpreted as measurements of this finance-oriented fork.

#### Usage

> Warning: Running across the 100 examples can cost ~$20-$100 depending on the model selection.

The dataset is available on [LangSmith via this link](https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d). To kick off evaluation, run the following command:

```bash
# Run comprehensive evaluation on LangSmith datasets
python tests/run_evaluate.py
```

This will provide a link to a LangSmith experiment, which will have a name `YOUR_EXPERIMENT_NAME`. Once this is done, extract the results to a JSONL file that can be submitted to the Deep Research Bench.

```bash
python tests/extract_langsmith_data.py --project-name "YOUR_EXPERIMENT_NAME" --model-name "you-model-name" --dataset-name "deep_research_bench"
```

This creates `tests/expt_results/deep_research_bench_model-name.jsonl` with the required format. Move the generated JSONL file to a local clone of the Deep Research Bench repository and follow their [Quick Start guide](https://github.com/Ayanami0730/deep_research_bench?tab=readme-ov-file#quick-start) for evaluation submission.

#### Upstream baseline results

| Name | Commit | Summarization | Research | Compression | Total Cost | Total Tokens | RACE Score | Experiment |
|------|--------|---------------|----------|-------------|------------|--------------|------------|------------|
| GPT-5 | [ca3951d](https://github.com/langchain-ai/open_deep_research/pull/168/commits) | openai:gpt-4.1-mini | openai:gpt-5 | openai:gpt-4.1 |  | 204,640,896 | 0.4943 | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-613c-4bda-8bde-f64f0422bbf3/compare?selectedSessions=4d5941c8-69ce-4f3d-8b3e-e3c99dfbd4cc&baseline=undefined) |
| Defaults | [6532a41](https://github.com/langchain-ai/open_deep_research/commit/6532a4176a93cc9bb2102b3d825dcefa560c85d9) | openai:gpt-4.1-mini | openai:gpt-4.1 | openai:gpt-4.1 | $45.98 | 58,015,332 | 0.4309 | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=cf4355d7-6347-47e2-a774-484f290e79bc&baseline=undefined) |
| Claude Sonnet 4 | [f877ea9](https://github.com/langchain-ai/open_deep_research/pull/163/commits/f877ea93641680879c420ea991e998b47aab9bcc) | openai:gpt-4.1-mini | anthropic:claude-sonnet-4-20250514 | openai:gpt-4.1 | $187.09 | 138,917,050 | 0.4401 | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=04f6002d-6080-4759-bcf5-9a52e57449ea&baseline=undefined) |
| Deep Research Bench Submission | [c0a160b](https://github.com/langchain-ai/open_deep_research/commit/c0a160b57a9b5ecd4b8217c3811a14d8eff97f72) | openai:gpt-4.1-nano | openai:gpt-4.1 | openai:gpt-4.1 | $87.83 | 207,005,549 | 0.4344 | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=e6647f74-ad2f-4cb9-887e-acb38b5f73c0&baseline=undefined) |

## 🚀 Deployments and Usage

#### LangGraph Studio

Follow the [quickstart](#-quickstart) to start LangGraph server locally and test the agent out on LangGraph Studio.

#### Hosted deployment
 
You can easily deploy to [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 

#### Open Agent Platform

Open Agent Platform (OAP) is a UI from which non-technical users can build and configure their own agents. OAP is great for allowing users to configure the Deep Researcher with different MCP tools and search APIs that are best suited to their needs and the problems that they want to solve.

The upstream Open Deep Research project is available in the public OAP demo. This repository can also be deployed to your own compatible LangGraph or OAP environment.

You can also deploy your own instance of OAP, and make your own custom agents (like Deep Researcher) available on it to your users.
1. [Deploy Open Agent Platform](https://docs.oap.langchain.com/quickstart)
2. [Add Deep Researcher to OAP](https://docs.oap.langchain.com/setup/agents)

## Legacy Implementations 🏛️

The `src/legacy/` folder contains two earlier implementations that provide alternative approaches to automated research. They are less performant than the current implementation, but provide alternative ideas understanding the different approaches to deep research.

#### 1. Workflow Implementation (`legacy/graph.py`)
- **Plan-and-Execute**: Structured workflow with human-in-the-loop planning
- **Sequential Processing**: Creates sections one by one with reflection
- **Interactive Control**: Allows feedback and approval of report plans
- **Quality Focused**: Emphasizes accuracy through iterative refinement

#### 2. Multi-Agent Implementation (`legacy/multi_agent.py`)  
- **Supervisor-Researcher Architecture**: Coordinated multi-agent system
- **Parallel Processing**: Multiple researchers work simultaneously
- **Speed Optimized**: Faster report generation through concurrency
- **MCP Support**: Extensive Model Context Protocol integration

## 🙏 Upstream and License

Open Financial Research is based on [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research). The original graph architecture, provider integrations, evaluation tooling, and legacy implementations come from that project. This repository adds the finance-oriented behavior and control mechanisms described above.

The project remains available under the [MIT License](LICENSE). Please retain the existing license and attribution when redistributing derived work.
