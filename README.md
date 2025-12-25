# AI Installer CLI & GUI
Install software with AI support!

Notice: ai_installer_cli.py ai_installer_gui.py and ai_installer_config.py work 
together and must be present in same directory.

# Steps your AI Agent will make
1. Analyze the target system state
2. Plan the installation steps based on requirements and user support.
3. Execute commands with proper error handling
4. Validate the installation through tests
5. Troubleshoot failures autonomously

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                    AI Installation Agent                                     │
├───────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                         │
│  │   System     │  │     LLM      │  │   Command    │                         │
│  │   Analyzer   │──│   Reasoner   │──│   Executor   │                         │
│  └──────────────┘  └──────────────┘  └──────────────┘                         │
│         │                 │                 │                                 │
│         ▼                 ▼                 ▼                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                         │
│  │  Requirement │  │    Test      │  │ Troubleshoot │                         │
│  │   Resolver   │  │   Runner     │  │    Engine    │                         │
│  └──────────────┘  └──────────────┘  └──────────────┘                         │
└───────────────────────────────────────────────────────────────────────────────┘
```

# 🏗️ Architecture Overview

The script is organized into 5 core components:

1. SystemAnalyzer

    Gathers comprehensive system information:

    - OS type, version, kernel
    - Port in use
    - WSL detection (version 1 or 2)
    - GPU hardware & driver versions
    - Installed packages (apt, pip, docker)
    - Disk space & memory

2. LLMInterface + InstallationReasoner

    The AI brain that:

    - Analyzes system state vs software requirements
    - Gathers user's configuration requests
    - Plans installation steps with proper ordering
    - Generates validation tests
    - Troubleshoots failures with targeted fixes

3. CommandExecutor

    Safely executes commands with:

    - Sudo handling
    - Timeout protection
    - Dry-run mode for testing

4. TestRunner

    Validates installations by:

    - Running post-install verification commands
    - Checking for expected outputs
    - Reporting pass/fail status

5. AIInstallationAgent

    Orchestrates the full workflow:

    - Analyze → System scan
    - Plan → AI generates steps
    - Execute → Run commands with confirmation
    - Validate → Test installation
    - Troubleshoot → Auto-fix failures

# VERSION 2.0 **NEW**

# 🚀 Usage Examples for CLI

## Use built-in gguf model
```sh
python ai_installer_cli.py "Install MySql Docker Container"
```

## Up-to-date installation using custom local models
```sh
python ai_installer_cli.py "Install Portainer" --provider local --local-present text-generation-webui
or
python ai_installer_cli.py "install portainer" --provider local --local-preset lm-studio
```

## Installation with OpenAI
```sh
python ai_installer_cli.py "Install CUDA 12.0" --provider openai --api-key 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
```

## Install with a dry run (Test to see steps without installing anything)
```sh
python ai_installer_cli.py "Install NextCloud AIO" --dry-run
```

## No confirmation prompts (automated)
```sh
python ai_installer_cli.py "Install Docker" --no-confirm
```

## Using custom servers/hosts
```sh
python ai_installer_cli.py "Install Docker" --provider local --local-preset lm-studio --host 127.0.40.3 --port 4000
python ai_installer_cli.py "Install Docker" --provider local --local-preset ollama --model llama3.1 --base-url http://myserver.xyz/v1
```

## Install with Gemini Model
```sh
python ai_installer_cli.py "Install Docker" --provider gemini --model gemini-3-pro-preview --api-key AIzaSyC-UbeuNs-4YQ9Ak_h00f5LO1c0Gq7d9f2
```

## Advanced options
```sh
python ai_installer_cli.py "Install Docker" --provider local \ 
    --local-preset lm-studio \ 
    --temperature 0.5 \ 
    --max-tokens 8192 \ 
    --timeout 180
```

## Test connection before running
```sh
python ai_installer_cli.py --provider local --local-preset lm-studio --test-connection
```

## List available models
```sh
python ai_installer_cli.py --provider local --local-preset ollama --list-models
```

# Compatible with the following AI Models
- openai
- gemini
- anthropic
- lm-studio
- ollama
- localai
- text-generation-webui
- vllm

# Features
- 📋 System Analyze & Scanning
- ⚙️ Configure Using Plain Language
- 🌐 Web Search (Built-in DuckDuckGo live searches)
- 🛡️ Dangerous Command Safety Filter
- 📦 Docker Image Support (Pulls directly from Docker API)
- 🤖 Automatic Troubleshooter & Repair

# Prerequisites for CLI
```sh
python3 -m venv venv
source venv/bin/activate
pip install anthropic  # or openai
export ANTHROPIC_API_KEY="your-key"  # or OPENAI_API_KEY or GEMINI_API_KEY
```

# Prerequisites for GUI
```sh
python3 -m venv venv
source venv/bin/activate
# using claude/anthropic
pip install anthropic
# using openai
pip install openai
# using Google Gemini
pip install google-genai
pip install PySide6
python ai_installer_gui.py
```

# Mission Statement
My goal is to make installing software completely automated, no longer should a software designer have to deal with customers/users unable to use their software due to installation issues. No longer will a user be barred from utilizing a peice of software because he/she doesn't fully understand it. If your on the operating system it's designed for, AI Installer will make sure installation will be a success. 
This will always be opensourced.
Anyone can add to it as long as: 
- #1: Bug fix
- #2: Stays to the design theme
- #3: Communicate any new features
