# AI Agents Blockchain Payment System

## Overview
This project demonstrates the integration of AI agents with blockchain technology, specifically focusing on facilitating autonomous payments using USDC. By creating a network of specialized AI agents using the AutoGen framework and managing payments via Circle's Programmable Wallets, this system autonomously executes research tasks and compensates agents based on their contributions.

## Features
- **Multi-Agent System**: Deploys multiple AI agents, each with specific roles like Admin, Engineer, Scientist, Planner, Executor, and Critic to collaboratively handle research tasks.
- **Blockchain Payments**: Implements Programmable Wallets to execute secure payments in USDC using Circle’s Web3 Services.
- **Environment Setup**: Detailed instructions on setting up the development environment for the project using Python and Node.js.

## Structure
```
ai_agents_project/
├── ai_research.py         # Handles AI research tasks and agent orchestration
├── circle_payment.js      # Manages blockchain payments via Circle's Programmable Wallets
├── main.py                # Main script to execute the research and payment workflow
├── .env                   # Environment variables for API keys
└── OAI_CONFIG_LIST        # Configuration for OpenAI models
```

## Prerequisites
- **Python** (3.9 or later)
- **Node.js** (v20.17.0 or later)
- **API Keys from Circle’s Developer Console**: Obtain your API keys by visiting [Circle’s Developer Console](https://console.circle.com/).

## Environment Setup

### Python Setup
1. Install Python and create a virtual environment:
   ```bash
   mkdir ai_agents_project
   cd ai_agents_project
   python -m venv ai_agents_env
   source ai_agents_env/bin/activate  # On Windows use: ai_agents_env\Scripts\activate
   ```
2. Install necessary Python packages:
   ```bash
   pip install autogen python-dotenv
   ```

### Node.js Setup
1. Create a new Node.js project:
   ```bash
   npm init -y
   ```
2. Install Circle's Developer-Controlled Wallets SDK:
   ```bash
   npm install @circle-fin/developer-controlled-wallets --save
   ```

### Configuration
1. Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   CIRCLE_API_KEY=your_circle_api_key
   CIRCLE_ENTITY_SECRET=your_circle_entity_secret
   ```
2. Create an `OAI_CONFIG_LIST` file containing necessary OpenAI configurations:
   ```json
   [
       {
           "model": "gpt-4o",
           "api_key": "${OPENAI_API_KEY}"
       }
   ]
   ```

## Usage
To run the entire project, execute the following command:
```bash
python3 main.py
```

### Key Scripts
- **`ai_research.py`**: Contains logic to create and manage the AI agents that contribute to research tasks.
- **`circle_payment.js`**: Implements blockchain payment processing, including wallet creation, transaction processing, and validation of payments received by AI agents.
- **`main.py`**: Orchestrates the execution of the research tasks and the payment processing.

## Key Functions
- **AI Research**: Managed by the function defined in `ai_research.py`, which initiates conversations among AI agents and collaborates on tasked workflows.
- **Blockchain Payments**: Handled by the functions in `circle_payment.js`, which encompass creating wallets, checking balances, and processing transactions.

## Running the Project
1. Execute the project by running `main.py`. It will perform AI research on the topic defined in the script and handle all wallet interactions and payments through the Circle API.
2. Monitor the logs in the terminal for real-time feedback regarding agent contributions and payment confirmations.

## Conclusion
This project shows the synergy between AI and blockchain, providing a foundational framework for building autonomous payment systems. By leveraging the power of AI agents and blockchain technology, this system opens pathways for automating workflows and transactions in diverse applications.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---