# 🚀 AI Financial Analyst: Multi-Agent Research Engine

Detailed documentation of our specialized AI Financial Analyst, built to synthesize complex financial data from live web sources and local SEC filings into professional investment memos.

---

## 💡 The Pivot: From Research to High-Finance
Originally a general research engine, we have pivoted this system into a specialized **AI Financial Analyst**. It is designed to think like a Wall Street Hedge Fund Analyst—critically evaluating management claims, identifying market discrepancies, and calculating the "Bear Case" for major tech investments.

### Key Use Cases:
- **Earnings Call Synthesis**: Upload a transcript and verify claims against live market news.
- **M&A Due Diligence**: Research rumors, competitive positioning, and financial health.
- **Market Sentiment Analysis**: Cross-reference private documents with live "on the ground" web reporting.

---

## 🏗️ System Architecture: The "Four-Agent" Stack

The engine operates on a multi-agent orchestration pattern where each agent has a distinct financial "personality."

### 1. 📄 The Doc Agent (Financial Data Extractor)
- **Role**: Heavy-lifting data extraction from local documents (10-Ks, 10-Qs, Transcripts).
- **Capabilities**: Uses `PyMuPDF` and `unstructured` to parse PDFs, TXTs, and MDs.
- **Goal**: Extract internal metrics (Revenue, EPS, Guidance) and store them in the **Pinecone Vector Memory**.

### 2. 🌐 The Web Agent (FireCrawl Powered)
- **Role**: Live market researcher.
- **Tooling**: Uses the **FireCrawl Search API**.
- **Capability**: Instead of just getting links, FireCrawl returns full **LLM-ready Markdown** of the top search results in one go.
- **Goal**: Find "The Reality"—what is happening on the news right now that might contradict what the company is saying in their filings.

### 3. 🔍 The Fact Checker (Skeptical Auditor)
- **Role**: The "Brake" on the system.
- **Capability**: Aggressively cross-references claims from the Doc Agent and Web Agent.
- **Goal**: Determine if a claim is `VERIFIED`, `DISPUTED`, or `NUANCED`. If it finds a conflict (e.g., Nvidia reports record profits but the stock is dropping), it triggers a **Cross-Question Loop** to find out why.

### 4. ✍️ The Writer Agent (Wall Street Analyst)
- **Role**: Synthesis and Delivery.
- **Capability**: Generates professional **Investment Memos**.
- **Special Feature**: Creates **ASCII Visual Charts** and tables directly in the markdown report for quick data digestion.

---

## 🔥 Key Technological Upgrades

### FireCrawl Integration
We moved away from traditional scraping to **FireCrawl**. This allows the engine to:
- Bypass heavy Javascript and paywalls.
- Retrieve clean markdown content directly into the prompt.
- Perform high-speed searches that combine Google results with instant content extraction.

### Report Persistence
All final research reports are preserved in a dedicated `/reports` folder with timestamped filenames.
`e:/Projects/ai-multiagent-content-engine/reports/Report_Topic_Timestamp.md`

---

## � The Anatomy of a Fact-Check: Nvidia Case Study

One of the most powerful features of this system is the **Cross-Question Loop**. This is how the engine handles contradictions between "Internal Claims" and "External Realities."

### 🚩 The Contradiction:
- **Doc Agent Claim**: Nvidia reports record-shattering Q4 FY26 revenue of **$68.1 Billion** (73% YoY growth) and massive profitability.
- **Web Agent Reality**: Live FireCrawl data shows Nvidia stock (NVDA) plummeted by **5.5% ($260B in market value lost)** immediately following the earnings announcement.

### 🛡️ The Fact-Check Proof:
The **Fact Checker** didn't just report both numbers. It performed a secondary search and a vector-memory (RAG) lookup to resolve *why* this was happening.

**Verification Process:**
1. **Initial Claim**: "Nvidia is at an all-time financial high."
2. **Conflict Detected**: Web Agent found "Panic Selling" news articles.
3. **Deep Search (FireCrawl)**: Fact Checker ran a target query: `"Why did NVDA stock drop after record Q4 earnings?"`
4. **Resolution**: The engine discovered that despite the records, the market was reacting to **valuation concerns** and the long-term threat of **Google and Amazon custom silicon** (which were verified as cheaper alternatives).

**Final Output**:
> *"While Nvidia's financial health is verified as record-breaking, the 5.5% stock drop represents a DISPUTED sentiment regarding valuation sustainability and the emergence of competitive threats from Amazon Trainium and Google TPUs."*

---

## �📊 Metrics & Performance

A typical "Medium Depth" research run on a topic like **Nvidia Q4 Earnings** results in:

| Metric | Average Value |
| :--- | :--- |
| **Total Sources Read** | 4-6 Sources (Web + Docs) |
| **Verified Claims** | 3-5 Significant Facts |
| **Execution Time** | ~45-60 Seconds |
| **Total Cost (USD)** | **~$0.03 - $0.05** |

*Note: By using GPT-4o-Mini for workers and GPT-4o for the final Orchestrator/Writer pass, we achieve high-fidelity output at a fraction of the cost of raw GPT-4o usage.*

---

## 🛠️ Usage Instructions

### 1. Simple Web Analysis
```bash
venv\Scripts\python engine.py --query "Investment Memo on Microsoft AI growth" --depth medium
```

### 2. Deep Document Verification
```bash
venv\Scripts\python engine.py --query "Verify Blackwell demand claims" --docs docs/Nvidia_Transcript.pdf --depth deep
```

### 3. Output Management
Final reports are rendered in the terminal using **Rich Markdown** for a beautiful UI and saved to the `reports/` folder for permanent storage.

---

**Built with:** Python, OpenAI, FireCrawl, Pinecone, Redis, and Rich.
