# AI Installer CLI & GUI
Install software with AI support!

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

# 🚀 Usage Examples for CLI

## Dry run (test without executing)
```sh
python ai_installer_cli.py "Install AMD ROCm 6.1 for WSL2" --dry-run
```

## Up-to-date installation using web search
```sh
python ai_installer_cli.py "Install Portainer" --provider local --local-present text-generation-webui --web-search
or
python ai_installer_cli.py "install portainer" --provider local --local-preset lm-studio -w
```

## Actual installation with OpenAI
```sh
python ai_installer_cli.py "Install CUDA 12.0" --provider openai
```

## Run the built-in ROCm example
```sh
python ai_installer_cli.py --example
```

## No confirmation prompts (automated)
```sh
python ai_installer_cli.py "Install Docker" --no-confirm
```

## Using presets
```sh
python ai_installer_cli.py "Install Docker" --provider local --local-preset lm-studio
python ai_installer_cli.py "Install Docker" --provider local --local-preset ollama --model llama3.1
```

## Custom server URL
```sh
python ai_installer_cli.py "Install Docker" --provider local \ 
    --base-url http://192.168.1.100:8080/v1 \ 
    --model my-model
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
- 🌐 Web Search
- 🛡️ Dangerous Command Safety Filter
- 📦 Docker Image Support
- 🤖 Automatic Troubleshooter & Repair

# Prerequisites for CLI
```sh
python3 -m venv venv
source venv/bin/activate
pip install anthropic  # or openai
export ANTHROPIC_API_KEY="your-key"  # or OPENAI_API_KEY
```

# Prerequisites for GUI
```sh
python3 -m venv venv
source venv/bin/activate
pip install anthropic  # or openai
if using Google Gemini
pip install google-generativeai
pip install PySide6
python ai_installer_gui.py
```
