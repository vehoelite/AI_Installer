# AI_Installer
Install software with AI support!

# Steps your AI Agent will make
1. Analyze the target system state
2. Plan the installation steps based on requirements
3. Execute commands with proper error handling
4. Validate the installation through tests
5. Troubleshoot failures autonomously

┌─────────────────────────────────────────────────────────────────┐
│                    AI Installation Agent                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   System     │  │     LLM      │  │   Command    │          │
│  │   Analyzer   │──│   Reasoner   │──│   Executor   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Requirement │  │    Test      │  │ Troubleshoot │          │
│  │   Resolver   │  │   Runner     │  │    Engine    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘

# 🏗️ Architecture Overview
The script is organized into 5 core components:

1. SystemAnalyzer
Gathers comprehensive system information:

OS type, version, kernel
WSL detection (version 1 or 2)
GPU hardware & driver versions
Installed packages (apt, pip)
Disk space & memory
2. LLMInterface + InstallationReasoner
The AI brain that:

Analyzes system state vs software requirements
Plans installation steps with proper ordering
Generates validation tests
Troubleshoots failures with targeted fixes
3. CommandExecutor
Safely executes commands with:

Sudo handling
Timeout protection
Dry-run mode for testing
Command history for rollback
4. TestRunner
Validates installations by:

Running post-install verification commands
Checking for expected outputs
Reporting pass/fail status
5. AIInstallationAgent
Orchestrates the full workflow:

Analyze → System scan
Plan → AI generates steps
Execute → Run commands with confirmation
Validate → Test installation
Troubleshoot → Auto-fix failures


# 🚀 Usage Examples
# Dry run (test without executing)
python ai_installer.py "Install AMD ROCm 6.1 for WSL2" --dry-run

# Actual installation with OpenAI
python ai_installer.py "Install CUDA 12.0" --provider openai

# Run the built-in ROCm example
python ai_installer.py --example

# No confirmation prompts (automated)
python ai_installer.py "Install Docker" --no-confirm

# Using presets
python ai_installer.py "Install Docker" --provider local --local-preset lm-studio
python ai_installer.py "Install Docker" --provider local --local-preset ollama --model llama3.1

# Custom server URL
python ai_installer.py "Install Docker" --provider local \
    --base-url http://192.168.1.100:8080/v1 \
    --model my-model

# Override port
python ai_installer.py "Install Docker" --provider local \
    --local-preset lm-studio --port 5000

# Override host (for remote servers)
python ai_installer.py "Install Docker" --provider local \
    --local-preset ollama --host 192.168.1.50 --port 11434

# Advanced options
python ai_installer.py "Install Docker" --provider local \
    --local-preset lm-studio \
    --temperature 0.5 \
    --max-tokens 8192 \
    --timeout 180

# Test connection before running
python ai_installer.py --provider local --local-preset lm-studio --test-connection

# List available models
python ai_installer.py --provider local --local-preset ollama --list-models

# Show configuration examples
python ai_installer.py --example-local

# Compatible with the following AI Models
[PRESETS]
openai
anthropic
lm-studio
ollama
localai
text-generation-webui
vllm

# Features
📋 System Analyze & Scanning
🛡️ Dangerous Command Safety Filter
🤖 Automatic Troubleshooter & Repair

# Prerequisites
python3 -m venv venv
source venv/bin/activate
pip install anthropic  # or openai
export ANTHROPIC_API_KEY="your-key"  # or OPENAI_API_KEY


