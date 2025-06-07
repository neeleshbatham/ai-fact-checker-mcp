# AI-Powered Fact-Checking Prototype with MCP Integration

## Overview
This project develops a proof-of-concept AI-powered fact-checking system that verifies textual claims using natural language processing (NLP) and a Model Context Protocol (MCP) layer. The MCP layer connects to multiple knowledge sources, including Wikipedia API, X (Twitter) API, and Google Fact Check Tools.

The system extracts claims from input text, retrieves evidence through the MCP layer, and provides verdicts with explanations.

## Project Information
- **Developer**: Neelesh Batham (241562502)
- **Program**: eMasters AI/ML '24
- **Course**: EE964 - Project
- **Timeline**: April 28, 2025 - June 15, 2025 (6 weeks)

## System Architecture

```
┌─────────────────┐     ┌───────────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                       │     │                 │     │                 │
│ Claim           │────▶│ Evidence Retrieval    │────▶│ Fact            │────▶│ Output          │
│ Extraction      │     │ (MCP Layer)           │     │ Verification    │     │ Generation      │
│                 │     │                       │     │                 │     │                 │
└─────────────────┘     └───────────────────────┘     └─────────────────┘     └─────────────────┘
                              │         │
                              ▼         ▼
             ┌─────────────────────┐  ┌─────────────────────┐
             │                     │  │                     │
             │ Wikipedia MCP       │  │ X (Twitter) MCP     │
             │ Server              │  │ Server              │
             │                     │  │                     │
             └─────────────────────┘  └─────────────────────┘
```

### Core Modules
1. **Claim Extraction**: Uses spaCy to identify verifiable claims from text inputs.
2. **Evidence Retrieval (MCP Layer)**: MCP client connecting to MCP servers for multiple knowledge sources:
   - Wikipedia MCP Server: Queries the Wikipedia API
   - X (Twitter) MCP Server: Retrieves relevant posts and trends
3. **Fact Verification**: Pre-trained RoBERTa model for textual entailment to compare claims against evidence.
4. **Output Generation**: BART model to generate human-readable explanations with source citations.

## Technologies
- **Backend**: Python
- **NLP Libraries**: Hugging Face Transformers (RoBERTa, BART), spaCy
- **MCP Framework**: MCP Python SDK
- **APIs**: Wikipedia API, X API, Google Fact Check Tools API
- **Frontend**: Streamlit
- **Hosting**: Local or free-tier cloud (GCP or IITK servers)

## Data Sources
- **Training/Validation**: FEVER dataset
- **Knowledge Bases**: Wikipedia, X posts, Google Fact Check Tools, and other extensible sources via MCP servers

## Development Timeline
- **Phase 1 (Weeks 1-2)**: Setup and Research
- **Phase 2 (Week 3)**: Core Development
- **Phase 3 (Week 4)**: Interface and Testing
- **Phase 4 (Week 5)**: Documentation and Presentation

## Ethical Considerations
- **Transparency**: All sources cited in outputs
- **Bias Mitigation**: Neutral datasets, source ranking by credibility
- **Privacy**: No user data collection in prototype

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- X API credentials (for X integration)

### Setup
1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/ai-fact-checker-mcp.git
   cd ai-fact-checker-mcp
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up API credentials
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

## Running the Application

### Start the MCP Servers
1. Start the Wikipedia MCP server
   ```bash
   python mcp_servers/wikipedia_server.py
   ```

2. Start the X MCP server (in a new terminal)
   ```bash
   python mcp_servers/x_server.py
   ```

### Launch the Web Interface
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 in your web browser.

### Using the Fact Checker
1. Enter a claim or paste text containing claims in the text box
2. Click "Check Facts" to process
3. View the results showing verdicts (True, False, Misleading) with explanations and sources

## Demo Examples
- "The Eiffel Tower is located in London."
- "Water boils at 100 degrees Celsius at sea level."
- "COVID-19 vaccines contain microchips."

## Contact
Neelesh Batham - neeleshb24@iitk.ac.in

## References
- FEVER dataset: http://fever.ai
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- MCP Framework: https://modelcontext.org
- Streamlit documentation: https://docs.streamlit.io
