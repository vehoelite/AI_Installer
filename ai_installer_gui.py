#!/usr/bin/env python3
"""
AI Installer - PySide6 GUI Application
=======================================
A user-friendly graphical interface for the AI-powered software installation agent.

Requirements:
    pip install PySide6

Usage:
    python ai_installer_gui.py
"""

import sys
import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Check for PySide6
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QLineEdit, QTextEdit, QPlainTextEdit,
        QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
        QTabWidget, QStackedWidget, QSplitter, QFrame, QScrollArea,
        QDialog, QDialogButtonBox, QMessageBox, QProgressBar,
        QStatusBar, QToolBar, QMenu, QMenuBar, QSystemTrayIcon,
        QSizePolicy, QSpacerItem, QFormLayout, QGridLayout
    )
    from PySide6.QtCore import (
        Qt, QThread, Signal, Slot, QTimer, QSize, QSettings, QUrl
    )
    from PySide6.QtGui import (
        QFont, QFontDatabase, QIcon, QAction, QPalette, QColor,
        QTextCursor, QDesktopServices, QPixmap
    )
except ImportError:
    print("=" * 60)
    print("  PySide6 is required for the GUI version")
    print("=" * 60)
    print()
    print("  Install it with:")
    print("    pip install PySide6")
    print()
    print("  Or use the CLI version:")
    print("    python ai_installer.py --help")
    print("=" * 60)
    sys.exit(1)

# Import our configuration and core modules
from ai_installer_config import (
    Config, ConfigManager, get_config_manager, get_config, save_config,
    LLMProviderSettings, AppSettings
)

# We'll import the core installer classes
try:
    from ai_installer import (
        AgentConfig, LLMProvider, OpenAICompatibleConfig,
        SystemAnalyzer, SystemInfo, InstallationReasoner,
        CommandExecutor, TestRunner, PlanEnhancer, WebSearcher,
        AnthropicLLM, OpenAILLM, OpenAICompatibleLLM
    )
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    CORE_AVAILABLE = False


# =============================================================================
# Dark Theme Stylesheet (Catppuccin Mocha inspired)
# =============================================================================

DARK_THEME = """
QMainWindow, QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'SF Pro Display', 'Helvetica Neue', sans-serif;
    font-size: 14px;
}

QLabel {
    color: #cdd6f4;
    background-color: transparent;
}

QLabel#title {
    font-size: 24px;
    font-weight: bold;
    color: #89b4fa;
}

QLabel#subtitle {
    font-size: 16px;
    color: #a6adc8;
}

QLabel#section {
    font-size: 16px;
    font-weight: bold;
    color: #f5c2e7;
    padding-top: 10px;
}

QLabel#success {
    color: #a6e3a1;
}

QLabel#error {
    color: #f38ba8;
}

QLabel#warning {
    color: #f9e2af;
}

QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #585b70;
}

QPushButton:pressed {
    background-color: #313244;
}

QPushButton:disabled {
    background-color: #313244;
    color: #6c7086;
}

QPushButton#primary {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
}

QPushButton#primary:hover {
    background-color: #b4befe;
}

QPushButton#primary:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QPushButton#success {
    background-color: #a6e3a1;
    color: #1e1e2e;
    font-weight: bold;
}

QPushButton#success:hover {
    background-color: #94e2d5;
}

QPushButton#success:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QPushButton#danger {
    background-color: #f38ba8;
    color: #1e1e2e;
}

QPushButton#danger:hover {
    background-color: #eba0ac;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 2px solid #45475a;
    border-radius: 6px;
    padding: 8px 12px;
    selection-background-color: #89b4fa;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #89b4fa;
}

QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {
    background-color: #181825;
    color: #6c7086;
}

QComboBox::drop-down {
    border: none;
    padding-right: 10px;
}

QComboBox::down-arrow {
    width: 12px;
    height: 12px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #45475a;
    border: 1px solid #45475a;
}

QTextEdit, QPlainTextEdit {
    background-color: #11111b;
    color: #cdd6f4;
    border: 2px solid #313244;
    border-radius: 6px;
    padding: 10px;
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace;
    font-size: 13px;
}

QGroupBox {
    background-color: #181825;
    border: 2px solid #313244;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
}

QGroupBox::title {
    color: #89b4fa;
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 5px;
}

QCheckBox {
    color: #cdd6f4;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 2px solid #45475a;
    background-color: #313244;
}

QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 6px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 6px;
}

QTabWidget::pane {
    background-color: #181825;
    border: 2px solid #313244;
    border-radius: 8px;
}

QTabBar::tab {
    background-color: #313244;
    color: #a6adc8;
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #181825;
    color: #89b4fa;
}

QTabBar::tab:hover:!selected {
    background-color: #45475a;
}

QScrollBar:vertical {
    background-color: #181825;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QStatusBar {
    background-color: #181825;
    color: #a6adc8;
}

QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
}

QMenuBar::item:selected {
    background-color: #313244;
}

QMenu {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
}

QMenu::item:selected {
    background-color: #45475a;
}

QSplitter::handle {
    background-color: #313244;
}

QFrame#card {
    background-color: #181825;
    border: 2px solid #313244;
    border-radius: 12px;
    padding: 20px;
}

QFrame#output_panel {
    background-color: #11111b;
    border: 2px solid #313244;
    border-radius: 8px;
}
"""


# =============================================================================
# Worker Thread for Background Operations
# =============================================================================

class WorkerThread(QThread):
    """Background worker thread for LLM and system operations"""
    output = Signal(str)  # Emit output text
    progress = Signal(int)  # Emit progress percentage
    finished_signal = Signal(dict)  # Emit result when done
    error = Signal(str)  # Emit error message

    def __init__(self, task_type: str, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if self.task_type == "analyze_system":
                self._analyze_system()
            elif self.task_type == "create_plan":
                self._create_plan()
            elif self.task_type == "execute_plan":
                self._execute_plan()
            elif self.task_type == "test_connection":
                self._test_connection()
            elif self.task_type == "answer_question":
                self._answer_question()
            elif self.task_type == "modify_plan":
                self._modify_plan()
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")

    def _analyze_system(self):
        """Analyze system configuration"""
        self.output.emit("Analyzing system configuration...")
        self.progress.emit(10)

        if not CORE_AVAILABLE:
            self.error.emit("Core modules not available. Please ensure ai_installer.py is in the same directory.")
            return

        analyzer = SystemAnalyzer(verbose=False)
        self.progress.emit(30)

        self.output.emit("  Detecting OS...")
        self.progress.emit(40)
        self.output.emit("  Checking GPU hardware...")
        self.progress.emit(60)
        self.output.emit("  Scanning installed packages...")
        self.progress.emit(80)

        system_info = analyzer.analyze()
        self.progress.emit(100)

        self.finished_signal.emit({"system_info": system_info})

    def _test_connection(self):
        """Test LLM connection"""
        config = self.kwargs.get("config")
        if not config:
            self.error.emit("No configuration provided")
            return

        self.output.emit("Testing connection to LLM...")

        try:
            if config.llm.provider == "local":
                base_url = get_config_manager().get_llm_base_url()
                local_config = OpenAICompatibleConfig(
                    base_url=base_url,
                    model_name=config.llm.local_model or "local-model",
                    api_key=config.llm.local_api_key
                )
                llm = OpenAICompatibleLLM(local_config)
                success, msg = llm.test_connection()
                if success:
                    models = llm.list_models()
                    self.finished_signal.emit({"success": True, "message": msg, "models": models})
                else:
                    self.finished_signal.emit({"success": False, "message": msg})
            elif config.llm.provider == "openai":
                # Test OpenAI connection by creating client
                import openai
                client = openai.OpenAI(api_key=config.llm.openai_api_key)
                # Simple test - list models
                models = client.models.list()
                self.finished_signal.emit({"success": True, "message": "OpenAI API connected"})
            elif config.llm.provider == "anthropic":
                # Test Anthropic connection
                import anthropic
                client = anthropic.Anthropic(api_key=config.llm.anthropic_api_key)
                self.finished_signal.emit({"success": True, "message": "Anthropic API configured"})
            else:
                self.finished_signal.emit({"success": True, "message": "API key configured"})
        except Exception as e:
            self.error.emit(str(e))

    def _get_llm(self):
        """Get the configured LLM instance"""
        config = self.kwargs.get("gui_config")
        if not config:
            raise ValueError("No configuration provided")

        if config.llm.provider == "local":
            base_url = get_config_manager().get_llm_base_url()
            local_config = OpenAICompatibleConfig(
                base_url=base_url,
                model_name=config.llm.local_model or "local-model",
                api_key=config.llm.local_api_key,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                timeout_seconds=config.llm.timeout_seconds
            )
            return OpenAICompatibleLLM(local_config)
        elif config.llm.provider == "openai":
            return OpenAILLM(config.llm.openai_api_key, config.llm.openai_model)
        elif config.llm.provider == "anthropic":
            return AnthropicLLM(config.llm.anthropic_api_key, config.llm.anthropic_model)
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm.provider}")

    def _create_plan(self):
        """Create installation plan using LLM"""
        software = self.kwargs.get("software", "")
        system_info = self.kwargs.get("system_info")
        web_search = self.kwargs.get("web_search", False)
        gui_config = self.kwargs.get("gui_config")

        if not software:
            self.error.emit("No software specified")
            return

        if not CORE_AVAILABLE:
            self.error.emit("Core modules not available")
            return

        self.output.emit(f"Creating installation plan for: {software}")
        self.progress.emit(10)

        # Analyze system if not provided
        if not system_info:
            self.output.emit("Analyzing system first...")
            analyzer = SystemAnalyzer(verbose=False)
            system_info = analyzer.analyze()
            self.progress.emit(25)

        # Web search for docs if enabled
        web_docs = None
        if web_search:
            self.output.emit("Searching web for up-to-date installation docs...")
            self.progress.emit(35)
            try:
                searcher = WebSearcher(verbose=False)
                os_info = f"{system_info.distro_name} {system_info.distro_version}"
                web_docs = searcher.search_installation_docs(software, os_info)
                if web_docs and web_docs.get("fetched_docs"):
                    self.output.emit(f"  Found {len(web_docs['fetched_docs'])} relevant docs")
            except Exception as e:
                self.output.emit(f"  Web search failed: {e}")
        self.progress.emit(45)

        # Get LLM and create plan
        self.output.emit("Consulting AI for installation plan...")
        try:
            llm = self._get_llm()
            reasoner = InstallationReasoner(llm)
            self.progress.emit(55)

            plan = reasoner.analyze_requirements(system_info, software, web_docs)
            self.progress.emit(75)

            # Enhance plan with real metrics
            self.output.emit("Calculating real package sizes and time estimates...")
            enhancer = PlanEnhancer(verbose=False)
            enhanced_plan, metrics = enhancer.enhance_plan(plan)
            self.progress.emit(100)

            self.finished_signal.emit({
                "plan": enhanced_plan,
                "metrics": metrics,
                "system_info": system_info,
                "web_docs": web_docs
            })

        except Exception as e:
            self.error.emit(f"Failed to create plan: {str(e)}")

    def _execute_plan(self):
        """Execute installation plan with auto-repair on failures"""
        plan = self.kwargs.get("plan")
        dry_run = self.kwargs.get("dry_run", False)
        system_info = self.kwargs.get("system_info")

        if not plan:
            self.error.emit("No plan provided")
            return

        if not CORE_AVAILABLE:
            self.error.emit("Core modules not available")
            return

        executor = CommandExecutor(dry_run=dry_run, verbose=False)
        steps = plan.get("installation_steps", [])
        total_steps = len(steps)

        self.output.emit(f"Starting installation ({total_steps} steps)...")
        if dry_run:
            self.output.emit("  [DRY RUN MODE - Commands will not be executed]")

        # Pre-authenticate sudo to avoid password prompts during execution
        if not dry_run:
            self.output.emit("\n[*] Checking sudo access...")
            self.output.emit("    If prompted, please enter your password in the terminal.")
            sudo_check = executor.execute("sudo -v", timeout=60)
            if sudo_check.success:
                self.output.emit("    Sudo access confirmed.")
            else:
                self.output.emit("    Warning: Could not verify sudo access. Some commands may fail.")

        results = []
        failed_steps = []

        for i, step in enumerate(steps):
            if self._is_cancelled:
                self.output.emit("\n[!] Installation cancelled by user")
                break

            step_num = step.get("step", i + 1)
            description = step.get("description", f"Step {step_num}")
            commands = step.get("commands", [])
            requires_sudo = step.get("requires_sudo", False)

            progress = int(((i + 1) / total_steps) * 100)
            self.progress.emit(progress)

            self.output.emit(f"\n[Step {step_num}] {description}")

            # Join multi-line commands (commands with trailing \)
            joined_commands = self._join_multiline_commands(commands)

            for cmd in joined_commands:
                if self._is_cancelled:
                    break

                # Display command (truncate if too long)
                display_cmd = cmd if len(cmd) < 100 else cmd[:97] + "..."
                self.output.emit(f"   $ {display_cmd}")

                result = executor.execute(cmd, requires_sudo=requires_sudo)
                results.append(result)

                if result.skipped:
                    self.output.emit(f"   [SKIP] {result.skip_reason[:80]}")
                elif result.success:
                    self.output.emit(f"   [OK] Completed ({result.duration_seconds:.1f}s)")
                    if result.stdout and len(result.stdout) < 200:
                        for line in result.stdout.split('\n')[:3]:
                            self.output.emit(f"   {line}")
                else:
                    self.output.emit(f"   [FAIL] Exit code {result.return_code}")
                    if result.stderr:
                        self.output.emit(f"   Error: {result.stderr[:200]}")
                    failed_steps.append({
                        "step": step_num,
                        "command": cmd,
                        "error": result.stderr or result.stdout,
                        "return_code": result.return_code
                    })

        # Auto-repair: Try to fix failed steps
        if failed_steps and not self._is_cancelled and not dry_run:
            self.output.emit("\n" + "=" * 50)
            self.output.emit("[*] Some steps failed. Attempting auto-repair...")
            self._attempt_auto_repair(executor, failed_steps, system_info, results)

        # Run post-install tests
        if not self._is_cancelled and plan.get("post_install_tests"):
            self.output.emit("\n[*] Running post-installation tests...")
            test_runner = TestRunner(executor, verbose=False)
            test_results = test_runner.run_tests(plan["post_install_tests"])
            self.output.emit(f"   Tests: {test_results['passed']}/{test_results['total']} passed")

        # Show pending user actions
        if executor.pending_user_actions:
            self.output.emit("\n[!] Actions required after installation:")
            for action in executor.pending_user_actions:
                self.output.emit(f"   - {action}")

        # Determine overall success
        final_success = not self._is_cancelled and len(failed_steps) == 0
        if not final_success and failed_steps:
            # Check if repairs fixed the issues
            final_success = all(r.success for r in results[-len(failed_steps):]) if len(results) >= len(failed_steps) else False

        self.finished_signal.emit({
            "success": final_success,
            "results": results,
            "pending_actions": executor.pending_user_actions,
            "failed_steps": failed_steps
        })

    def _join_multiline_commands(self, commands: list) -> list:
        """Join commands that are split across multiple lines with backslash continuation"""
        if not commands:
            return commands

        joined = []
        current_cmd = ""

        for cmd in commands:
            if isinstance(cmd, str):
                cmd = cmd.strip()
                if current_cmd:
                    # Continue building multi-line command
                    current_cmd += " " + cmd
                else:
                    current_cmd = cmd

                # Check if this command continues on next line
                if current_cmd.endswith("\\"):
                    # Remove trailing backslash and continue
                    current_cmd = current_cmd[:-1].strip()
                else:
                    # Command is complete
                    if current_cmd:
                        joined.append(current_cmd)
                    current_cmd = ""

        # Don't forget the last command if it didn't have a trailing newline
        if current_cmd:
            joined.append(current_cmd)

        return joined

    def _attempt_auto_repair(self, executor, failed_steps, system_info, results):
        """Attempt to auto-repair failed installation steps"""
        try:
            llm = self._get_llm()
            reasoner = InstallationReasoner(llm)

            for failure in failed_steps[:3]:  # Limit to first 3 failures
                self.output.emit(f"\n[*] Analyzing failure in step {failure['step']}...")

                # Get troubleshooting advice from LLM
                fix_plan = reasoner.troubleshoot(
                    system_info,
                    failure['error'],
                    failure['command'],
                    f"Step {failure['step']} failed with exit code {failure['return_code']}"
                )

                if fix_plan.get("_parse_error"):
                    self.output.emit("   Could not parse repair suggestions")
                    continue

                self.output.emit(f"   Analysis: {fix_plan.get('root_cause', 'Unknown')[:100]}")

                # Execute fix steps
                fix_steps = fix_plan.get("fix_steps", [])
                if fix_steps:
                    self.output.emit(f"   Attempting {len(fix_steps)} repair step(s)...")
                    for fix_step in fix_steps:
                        fix_commands = fix_step.get("commands", [])
                        for fix_cmd in fix_commands:
                            self.output.emit(f"   $ {fix_cmd[:80]}...")
                            fix_result = executor.execute(
                                fix_cmd,
                                requires_sudo=fix_step.get("requires_sudo", False)
                            )
                            results.append(fix_result)
                            if fix_result.success:
                                self.output.emit("   [OK] Repair step completed")
                            else:
                                self.output.emit(f"   [FAIL] Repair failed: {fix_result.stderr[:100]}")

                    # Retry the original command if suggested
                    if fix_plan.get("should_retry_original", False):
                        self.output.emit(f"   Retrying original command...")
                        retry_result = executor.execute(failure['command'])
                        results.append(retry_result)
                        if retry_result.success:
                            self.output.emit("   [OK] Original command now succeeded!")
                        else:
                            self.output.emit("   [FAIL] Original command still failing")
                else:
                    self.output.emit("   No automatic fix available")
                    if fix_plan.get("alternative_approaches"):
                        self.output.emit("   Alternatives:")
                        for alt in fix_plan["alternative_approaches"][:2]:
                            self.output.emit(f"     - {alt[:80]}")

        except Exception as e:
            self.output.emit(f"   Auto-repair error: {str(e)[:100]}")

    def _answer_question(self):
        """Answer a question about the installation plan"""
        question = self.kwargs.get("question", "")
        plan = self.kwargs.get("plan")
        system_info = self.kwargs.get("system_info")
        web_docs = self.kwargs.get("web_docs")

        if not question:
            self.error.emit("No question provided")
            return

        if not CORE_AVAILABLE:
            self.error.emit("Core modules not available")
            return

        self.output.emit("Processing your question...")

        try:
            llm = self._get_llm()
            reasoner = InstallationReasoner(llm)
            answer = reasoner.answer_question(question, plan or {}, system_info, web_docs)
            self.finished_signal.emit({"answer": answer})
        except Exception as e:
            self.error.emit(f"Failed to answer question: {str(e)}")

    def _modify_plan(self):
        """Modify the installation plan based on user request"""
        request = self.kwargs.get("request", "")
        plan = self.kwargs.get("plan")
        system_info = self.kwargs.get("system_info")

        if not request or not plan:
            self.error.emit("Missing request or plan")
            return

        if not CORE_AVAILABLE:
            self.error.emit("Core modules not available")
            return

        self.output.emit("Modifying installation plan...")
        self.progress.emit(30)

        try:
            llm = self._get_llm()
            reasoner = InstallationReasoner(llm)
            modified_plan = reasoner.modify_plan(plan, request, system_info)
            self.progress.emit(70)

            # Re-enhance with real metrics
            enhancer = PlanEnhancer(verbose=False)
            enhanced_plan, metrics = enhancer.enhance_plan(modified_plan)
            self.progress.emit(100)

            self.finished_signal.emit({
                "plan": enhanced_plan,
                "metrics": metrics
            })
        except Exception as e:
            self.error.emit(f"Failed to modify plan: {str(e)}")


# =============================================================================
# Sudo Password Dialog
# =============================================================================

class SudoAuthDialog(QDialog):
    """Dialog to notify user about sudo password requirement"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Authentication Required")
        self.setMinimumWidth(450)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Warning icon and message
        msg_layout = QHBoxLayout()
        icon_label = QLabel("[!]")
        icon_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #f9e2af;")
        msg_layout.addWidget(icon_label)

        msg = QLabel(
            "<b>Administrator Privileges Required</b><br><br>"
            "This installation requires sudo access. When you proceed, "
            "you may be prompted for your password <b>in the terminal window</b> "
            "where you launched this application.<br><br>"
            "<b>Please keep the terminal window visible.</b>"
        )
        msg.setWordWrap(True)
        msg_layout.addWidget(msg, 1)
        layout.addLayout(msg_layout)

        # Tip
        tip = QLabel(
            "Tip: You can run 'sudo -v' in the terminal before starting "
            "to pre-authenticate and avoid prompts during installation."
        )
        tip.setObjectName("subtitle")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        layout.addSpacing(10)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        proceed_btn = QPushButton("I Understand, Proceed")
        proceed_btn.setObjectName("primary")
        proceed_btn.clicked.connect(self.accept)
        button_layout.addWidget(proceed_btn)

        layout.addLayout(button_layout)


# =============================================================================
# Setup Wizard Dialog
# =============================================================================

class SetupWizard(QDialog):
    """First-run setup wizard for configuring the AI Installer"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Installer Setup")
        self.setMinimumSize(600, 550)
        self.config_manager = get_config_manager()
        self.config = self.config_manager.config
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header = QLabel("Welcome to AI Installer")
        header.setObjectName("title")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        subtitle = QLabel("Let's configure your LLM provider to get started")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Provider selection
        provider_group = QGroupBox("LLM Provider")
        provider_layout = QFormLayout(provider_group)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Local LLM (LM Studio, Ollama, etc.)", "OpenAI API", "Anthropic API"])
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        provider_layout.addRow("Provider:", self.provider_combo)

        layout.addWidget(provider_group)

        # Stacked widget for provider-specific settings
        self.settings_stack = QStackedWidget()

        # Local LLM settings
        local_widget = QWidget()
        local_layout = QFormLayout(local_widget)

        self.local_preset = QComboBox()
        self.local_preset.addItems(["LM Studio", "Ollama", "LocalAI", "Custom"])
        self.local_preset.currentIndexChanged.connect(self._on_preset_changed)
        local_layout.addRow("Preset:", self.local_preset)

        self.local_host = QLineEdit("127.0.0.1")
        local_layout.addRow("Host:", self.local_host)

        self.local_port = QSpinBox()
        self.local_port.setRange(1, 65535)
        self.local_port.setValue(1234)
        local_layout.addRow("Port:", self.local_port)

        self.local_model = QLineEdit()
        self.local_model.setPlaceholderText("Leave blank to auto-detect")
        local_layout.addRow("Model:", self.local_model)

        self.test_btn = QPushButton("Test Connection")
        self.test_btn.setObjectName("primary")
        self.test_btn.clicked.connect(self._test_connection)
        local_layout.addRow("", self.test_btn)

        self.test_status = QLabel("")
        local_layout.addRow("", self.test_status)

        self.settings_stack.addWidget(local_widget)

        # OpenAI settings
        openai_widget = QWidget()
        openai_layout = QFormLayout(openai_widget)

        self.openai_key = QLineEdit()
        self.openai_key.setPlaceholderText("sk-...")
        self.openai_key.setEchoMode(QLineEdit.Password)
        openai_layout.addRow("API Key:", self.openai_key)

        self.openai_model = QComboBox()
        self.openai_model.addItems(["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
        self.openai_model.setEditable(True)
        openai_layout.addRow("Model:", self.openai_model)

        self.settings_stack.addWidget(openai_widget)

        # Anthropic settings
        anthropic_widget = QWidget()
        anthropic_layout = QFormLayout(anthropic_widget)

        self.anthropic_key = QLineEdit()
        self.anthropic_key.setPlaceholderText("sk-ant-...")
        self.anthropic_key.setEchoMode(QLineEdit.Password)
        anthropic_layout.addRow("API Key:", self.anthropic_key)

        self.anthropic_model = QComboBox()
        self.anthropic_model.addItems(["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"])
        self.anthropic_model.setEditable(True)
        anthropic_layout.addRow("Model:", self.anthropic_model)

        self.settings_stack.addWidget(anthropic_widget)

        layout.addWidget(self.settings_stack)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        self.web_search_check = QCheckBox("Enable web search for up-to-date installation docs")
        self.web_search_check.setChecked(True)
        options_layout.addWidget(self.web_search_check)

        self.verbose_check = QCheckBox("Show detailed output during installation")
        self.verbose_check.setChecked(True)
        options_layout.addWidget(self.verbose_check)

        self.confirm_check = QCheckBox("Confirm before executing commands")
        self.confirm_check.setChecked(True)
        options_layout.addWidget(self.confirm_check)

        layout.addWidget(options_group)

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save & Continue")
        save_btn.setObjectName("success")
        save_btn.clicked.connect(self._save_and_continue)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

        # Load existing config
        self._load_config()

    def _load_config(self):
        """Load existing configuration into UI"""
        llm = self.config.llm

        # Set provider
        provider_map = {"local": 0, "openai": 1, "anthropic": 2}
        self.provider_combo.setCurrentIndex(provider_map.get(llm.provider, 0))

        # Local settings
        preset_map = {"lm-studio": 0, "ollama": 1, "localai": 2, "custom": 3}
        self.local_preset.setCurrentIndex(preset_map.get(llm.local_preset, 0))
        self.local_host.setText(llm.local_host)
        self.local_port.setValue(llm.local_port)
        self.local_model.setText(llm.local_model)

        # API keys
        self.openai_key.setText(llm.openai_api_key)
        self.anthropic_key.setText(llm.anthropic_api_key)

        # Options
        self.web_search_check.setChecked(self.config.app.web_search_enabled)
        self.verbose_check.setChecked(self.config.app.verbose_output)
        self.confirm_check.setChecked(self.config.app.confirm_before_execute)

    def _on_provider_changed(self, index):
        self.settings_stack.setCurrentIndex(index)

    def _on_preset_changed(self, index):
        ports = {0: 1234, 1: 11434, 2: 8080, 3: 1234}
        self.local_port.setValue(ports.get(index, 1234))

    def _test_connection(self):
        self.test_status.setText("Testing...")
        self.test_status.setStyleSheet("color: #f9e2af;")

        # Save current settings temporarily
        self._save_to_config()

        # Run test in background
        self.worker = WorkerThread("test_connection", config=self.config)
        self.worker.finished_signal.connect(self._on_test_complete)
        self.worker.error.connect(self._on_test_error)
        self.worker.start()

    def _on_test_complete(self, result):
        if result.get("success"):
            models = result.get("models", [])
            model_text = f" ({len(models)} models available)" if models else ""
            self.test_status.setText(f"✓ Connected{model_text}")
            self.test_status.setStyleSheet("color: #a6e3a1;")

            # Auto-fill model if available
            if models and not self.local_model.text():
                self.local_model.setText(models[0])
        else:
            self.test_status.setText(f"✗ {result.get('message', 'Connection failed')}")
            self.test_status.setStyleSheet("color: #f38ba8;")

    def _on_test_error(self, error):
        self.test_status.setText(f"✗ Error: {error[:50]}...")
        self.test_status.setStyleSheet("color: #f38ba8;")

    def _save_to_config(self):
        """Save UI values to config object"""
        providers = ["local", "openai", "anthropic"]
        self.config.llm.provider = providers[self.provider_combo.currentIndex()]

        presets = ["lm-studio", "ollama", "localai", "custom"]
        self.config.llm.local_preset = presets[self.local_preset.currentIndex()]
        self.config.llm.local_host = self.local_host.text()
        self.config.llm.local_port = self.local_port.value()
        self.config.llm.local_model = self.local_model.text()

        self.config.llm.openai_api_key = self.openai_key.text()
        self.config.llm.openai_model = self.openai_model.currentText()

        self.config.llm.anthropic_api_key = self.anthropic_key.text()
        self.config.llm.anthropic_model = self.anthropic_model.currentText()

        self.config.app.web_search_enabled = self.web_search_check.isChecked()
        self.config.app.verbose_output = self.verbose_check.isChecked()
        self.config.app.confirm_before_execute = self.confirm_check.isChecked()

    def _save_and_continue(self):
        self._save_to_config()
        self.config.app.first_run_complete = True
        self.config_manager.save()
        self.accept()


# =============================================================================
# Main Window
# =============================================================================

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Installer")
        self.setMinimumSize(1100, 750)

        self.config_manager = get_config_manager()
        self.config = self.config_manager.config
        self.system_info = None
        self.current_plan = None
        self.current_metrics = None
        self.web_docs = None
        self.worker = None

        self._init_ui()
        self._restore_window_state()

        # Auto-analyze system on startup
        QTimer.singleShot(500, self._analyze_system)

    def _init_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Input
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 10, 20)

        # Header
        header_layout = QHBoxLayout()
        header = QLabel("AI Install Agent")
        header.setObjectName("title")
        header_layout.addWidget(header)
        header_layout.addStretch()

        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._show_settings)
        header_layout.addWidget(settings_btn)
        left_layout.addLayout(header_layout)

        # Software input
        input_group = QGroupBox("What would you like to install?")
        input_layout = QVBoxLayout(input_group)

        self.software_input = QLineEdit()
        self.software_input.setPlaceholderText("e.g., Docker web UI, PostgreSQL database, Node.js, Portainer...")
        self.software_input.returnPressed.connect(self._start_installation)
        input_layout.addWidget(self.software_input)

        # Options row
        options_layout = QHBoxLayout()

        self.web_search_check = QCheckBox("Web Search")
        self.web_search_check.setChecked(self.config.app.web_search_enabled)
        self.web_search_check.setToolTip("Search the web for up-to-date installation instructions")
        options_layout.addWidget(self.web_search_check)

        self.dry_run_check = QCheckBox("Dry Run")
        self.dry_run_check.setToolTip("Preview commands without executing them")
        options_layout.addWidget(self.dry_run_check)

        options_layout.addStretch()

        self.start_btn = QPushButton("Create Plan")
        self.start_btn.setObjectName("primary")
        self.start_btn.setMinimumWidth(120)
        self.start_btn.clicked.connect(self._start_installation)
        options_layout.addWidget(self.start_btn)

        input_layout.addLayout(options_layout)
        left_layout.addWidget(input_group)

        # System info card
        self.system_card = QGroupBox("System Information")
        system_layout = QVBoxLayout(self.system_card)
        self.system_label = QLabel("Analyzing system...")
        self.system_label.setWordWrap(True)
        system_layout.addWidget(self.system_label)
        left_layout.addWidget(self.system_card)

        # Plan display
        self.plan_group = QGroupBox("Installation Plan")
        plan_layout = QVBoxLayout(self.plan_group)
        self.plan_display = QTextEdit()
        self.plan_display.setReadOnly(True)
        self.plan_display.setPlaceholderText("Enter what you want to install and click 'Create Plan'...")
        self.plan_display.setMinimumHeight(200)
        plan_layout.addWidget(self.plan_display)
        left_layout.addWidget(self.plan_group)

        # Chat/Q&A input
        chat_group = QGroupBox("Ask Questions or Request Changes")
        chat_layout = QVBoxLayout(chat_group)

        hint_label = QLabel("Examples: 'What config options are available?' | 'Change port to 8080' | 'How large is this?'")
        hint_label.setObjectName("subtitle")
        hint_label.setWordWrap(True)
        chat_layout.addWidget(hint_label)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your question or modification request...")
        self.chat_input.returnPressed.connect(self._send_chat)
        self.chat_input.setEnabled(False)
        chat_layout.addWidget(self.chat_input)

        chat_buttons = QHBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send_chat)
        self.send_btn.setEnabled(False)
        chat_buttons.addWidget(self.send_btn)

        chat_buttons.addStretch()

        self.install_btn = QPushButton("Install Now")
        self.install_btn.setObjectName("success")
        self.install_btn.setMinimumWidth(150)
        self.install_btn.clicked.connect(self._execute_installation)
        self.install_btn.setEnabled(False)
        chat_buttons.addWidget(self.install_btn)

        chat_layout.addLayout(chat_buttons)
        left_layout.addWidget(chat_group)

        splitter.addWidget(left_panel)

        # Right panel - Output
        right_panel = QFrame()
        right_panel.setObjectName("output_panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 20, 20, 20)

        output_header = QLabel("Output")
        output_header.setObjectName("section")
        right_layout.addWidget(output_header)

        self.output_display = QPlainTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setPlaceholderText("Output will appear here...")
        right_layout.addWidget(self.output_display)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Cancel button (hidden by default)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.clicked.connect(self._cancel_operation)
        self.cancel_btn.setVisible(False)
        right_layout.addWidget(self.cancel_btn)

        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([550, 450])
        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Menu bar
        self._create_menu_bar()

    def _create_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self._show_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        github_action = QAction("GitHub Repository", self)
        github_action.triggered.connect(self._open_github)
        help_menu.addAction(github_action)

    def _restore_window_state(self):
        """Restore window position and size"""
        if self.config.app.window_x >= 0:
            self.move(self.config.app.window_x, self.config.app.window_y)
        self.resize(self.config.app.window_width, self.config.app.window_height)

    def _save_window_state(self):
        """Save window position and size"""
        self.config.app.window_width = self.width()
        self.config.app.window_height = self.height()
        self.config.app.window_x = self.x()
        self.config.app.window_y = self.y()
        self.config_manager.save()

    def closeEvent(self, event):
        self._save_window_state()
        super().closeEvent(event)

    def _append_output(self, text: str):
        """Append text to output display"""
        self.output_display.appendPlainText(text)
        # Scroll to bottom
        cursor = self.output_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.output_display.setTextCursor(cursor)

    def _set_busy(self, busy: bool, message: str = ""):
        """Set UI to busy/idle state"""
        self.start_btn.setEnabled(not busy)
        self.send_btn.setEnabled(not busy and self.current_plan is not None)
        self.install_btn.setEnabled(not busy and self.current_plan is not None)
        self.chat_input.setEnabled(not busy and self.current_plan is not None)
        self.software_input.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        self.cancel_btn.setVisible(busy)
        if message:
            self.status_bar.showMessage(message)

    def _cancel_operation(self):
        """Cancel the current operation"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._append_output("\n[!] Cancellation requested...")

    def _analyze_system(self):
        self.status_bar.showMessage("Analyzing system...")
        self.system_label.setText("Analyzing system configuration...")

        if not CORE_AVAILABLE:
            self.system_label.setText("[!] Core modules not available. Check ai_installer.py is present.")
            self.status_bar.showMessage("Error: Core modules missing")
            return

        self.worker = WorkerThread("analyze_system")
        self.worker.output.connect(self._append_output)
        self.worker.finished_signal.connect(self._on_system_analyzed)
        self.worker.error.connect(lambda e: self.system_label.setText(f"[!] Error: {e}"))
        self.worker.start()

    def _on_system_analyzed(self, result):
        self.system_info = result.get("system_info")
        if self.system_info:
            info = self.system_info
            wsl_text = f"Yes (v{info.wsl_version})" if info.is_wsl else "No"

            # Get GPU info - prefer detailed gpus list, fall back to gpu_info
            gpu_text = "None detected"
            if info.gpus:
                # Use detailed GPU info from nvidia-smi/rocm-smi
                gpu_names = []
                for gpu in info.gpus[:2]:
                    if hasattr(gpu, 'name') and gpu.name:
                        gpu_names.append(gpu.name)
                    elif isinstance(gpu, dict) and gpu.get('name'):
                        gpu_names.append(gpu['name'])
                if gpu_names:
                    gpu_text = ", ".join(gpu_names)
                    if len(info.gpus) > 2:
                        gpu_text += f" (+{len(info.gpus)-2} more)"
            elif info.gpu_info:
                # Fall back to lspci output
                gpu_text = info.gpu_info[0][:60] + "..." if len(info.gpu_info[0]) > 60 else info.gpu_info[0]

            # Add driver info if available
            driver_text = ""
            if info.gpu_driver_version:
                driver_text = f" (Driver: {info.gpu_driver_version})"
            if info.cuda_version:
                driver_text += f" CUDA {info.cuda_version}"
            if info.rocm_version:
                driver_text += f" ROCm {info.rocm_version}"

            text = f"""<b>OS:</b> {info.distro_name} {info.distro_version}<br>
<b>Kernel:</b> {info.kernel_version}<br>
<b>WSL:</b> {wsl_text}<br>
<b>GPU:</b> {gpu_text}{driver_text}<br>
<b>Memory:</b> {info.memory_info.get('total', 'Unknown')}<br>
<b>Disk:</b> {info.disk_space.get('available', 'Unknown')} available"""
            self.system_label.setText(text)
        self.status_bar.showMessage("Ready")

    def _start_installation(self):
        """Start creating an installation plan"""
        software = self.software_input.text().strip()
        if not software:
            self._show_message_box("Input Required", "Please enter the software you want to install.", "warning")
            return

        self._set_busy(True, "Creating installation plan...")
        self._append_output("\n" + "=" * 50)
        self._append_output(f"Creating installation plan for: {software}")
        self.progress_bar.setValue(0)

        self.worker = WorkerThread(
            "create_plan",
            software=software,
            system_info=self.system_info,
            web_search=self.web_search_check.isChecked(),
            gui_config=self.config
        )
        self.worker.output.connect(self._append_output)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self._on_plan_created)
        self.worker.error.connect(self._on_plan_error)
        self.worker.start()

    def _on_plan_created(self, result):
        """Handle plan creation completion"""
        self._set_busy(False, "Plan created")

        self.current_plan = result.get("plan")
        self.current_metrics = result.get("metrics")
        self.web_docs = result.get("web_docs")

        if result.get("system_info"):
            self.system_info = result["system_info"]

        if self.current_plan:
            self._display_plan()
            self.chat_input.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.install_btn.setEnabled(True)
            self._append_output("\n[OK] Plan created successfully!")
            self._append_output("You can now ask questions, request changes, or click 'Install Now'")

    def _on_plan_error(self, error):
        """Handle plan creation error"""
        self._set_busy(False, "Error")
        self._append_output(f"\n[ERROR] {error}")
        self._show_message_box("Error Creating Plan", error, "critical")

    def _display_plan(self):
        """Display the current plan in the plan display area"""
        if not self.current_plan:
            return

        plan = self.current_plan
        metrics = self.current_metrics

        # Build plan display text
        lines = []
        software_name = plan.get("software_name", "Unknown")
        lines.append(f"[PACKAGE] {software_name}")
        lines.append("=" * 40)

        # Compatibility
        if not plan.get("compatible", True):
            lines.append("\n[!] COMPATIBILITY ISSUES:")
            for issue in plan.get("compatibility_issues", []):
                lines.append(f"  - {issue}")

        # Configuration
        config = plan.get("configuration", {})
        if config:
            lines.append("\n[CONFIG] Configuration:")
            if config.get("ports"):
                lines.append(f"  Ports: {', '.join(map(str, config['ports']))}")
            if config.get("data_directory"):
                lines.append(f"  Data: {config['data_directory']}")

        # Port conflict detection
        if self.system_info and config.get("ports"):
            listening_ports = getattr(self.system_info, 'listening_ports', [])
            conflicting_ports = []
            for port in config["ports"]:
                if port in listening_ports:
                    conflicting_ports.append(port)
            if conflicting_ports:
                lines.append(f"\n[!!] PORT CONFLICTS DETECTED:")
                for port in conflicting_ports:
                    lines.append(f"  - Port {port} is already in use!")
                lines.append("  Consider changing ports or stopping the conflicting service.")

        # Resource requirements
        if metrics:
            lines.append("\n[RESOURCES] Requirements:")
            total_download = metrics.total_download_size_mb + metrics.docker_download_size_mb
            if total_download > 0:
                lines.append(f"  Download: {total_download:.1f} MB")
            if metrics.total_disk_space_mb > 0:
                lines.append(f"  Disk space: {metrics.total_disk_space_mb:.1f} MB")
            lines.append(f"  Est. time: ~{metrics.estimated_total_time_minutes:.1f} min")

        # Packages to install
        if metrics and metrics.packages_to_install:
            lines.append(f"\n[PACKAGES] ({len(metrics.packages_to_install)}):")
            for pkg in metrics.packages_to_install[:5]:
                lines.append(f"  - {pkg}")
            if len(metrics.packages_to_install) > 5:
                lines.append(f"  ... and {len(metrics.packages_to_install) - 5} more")

        # Docker images
        if metrics and metrics.docker_images_to_pull:
            lines.append(f"\n[DOCKER] Images ({len(metrics.docker_images_to_pull)}):")
            for img in metrics.docker_images_to_pull[:3]:
                lines.append(f"  - {img}")
            if len(metrics.docker_images_to_pull) > 3:
                lines.append(f"  ... and {len(metrics.docker_images_to_pull) - 3} more")

        # Installation steps summary
        steps = plan.get("installation_steps", [])
        if steps:
            lines.append(f"\n[STEPS] ({len(steps)}):")
            for step in steps[:5]:
                desc = step.get("description", "Unknown step")[:50]
                lines.append(f"  {step.get('step', '?')}. {desc}")
            if len(steps) > 5:
                lines.append(f"  ... and {len(steps) - 5} more steps")

        # Warnings
        if plan.get("warnings"):
            lines.append("\n[!] Warnings:")
            for warning in plan["warnings"]:
                lines.append(f"  - {warning}")

        # Modifications
        if plan.get("modifications_made"):
            lines.append("\n[MODIFIED]:")
            for mod in plan["modifications_made"]:
                lines.append(f"  - {mod}")

        self.plan_display.setText("\n".join(lines))

    def _send_chat(self):
        """Handle chat input"""
        message = self.chat_input.text().strip()
        if not message:
            return

        self._append_output(f"\n[YOU] {message}")
        self.chat_input.clear()

        # Detect if this is a question or modification request
        is_question = self._is_question(message)

        self._set_busy(True, "Processing...")
        self.progress_bar.setValue(0)

        if is_question:
            self._append_output("[AI] Thinking...")
            self.worker = WorkerThread(
                "answer_question",
                question=message,
                plan=self.current_plan,
                system_info=self.system_info,
                web_docs=self.web_docs,
                gui_config=self.config
            )
            self.worker.finished_signal.connect(self._on_question_answered)
        else:
            self._append_output("[AI] Modifying plan...")
            self.worker = WorkerThread(
                "modify_plan",
                request=message,
                plan=self.current_plan,
                system_info=self.system_info,
                gui_config=self.config
            )
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished_signal.connect(self._on_plan_modified)

        self.worker.output.connect(self._append_output)
        self.worker.error.connect(self._on_chat_error)
        self.worker.start()

    def _is_question(self, text: str) -> bool:
        """Detect if user input is a question vs modification request"""
        text_lower = text.lower().strip()

        # Ends with question mark
        if text.endswith("?"):
            return True

        # Starts with question words
        question_starters = (
            "how ", "what ", "why ", "when ", "where ", "which ",
            "who ", "can ", "could ", "would ", "is ", "are ", "do ", "does ",
            "tell me", "explain", "describe"
        )
        if text_lower.startswith(question_starters):
            return True

        return False

    def _on_question_answered(self, result):
        """Handle question answered"""
        self._set_busy(False, "Ready")
        answer = result.get("answer", "No answer received")
        self._append_output(f"\n[AI] {answer}")

    def _on_plan_modified(self, result):
        """Handle plan modification"""
        self._set_busy(False, "Plan updated")
        self.current_plan = result.get("plan")
        self.current_metrics = result.get("metrics")
        self._display_plan()
        self._append_output("\n[OK] Plan updated!")

    def _on_chat_error(self, error):
        """Handle chat error"""
        self._set_busy(False, "Error")
        self._append_output(f"\n[ERROR] {error}")

    def _execute_installation(self):
        """Execute the installation plan"""
        if not self.current_plan:
            self._show_message_box("No Plan", "Please create an installation plan first.", "warning")
            return

        # Confirm before execution
        if self.config.app.confirm_before_execute:
            reply = self._show_confirm_dialog(
                "Confirm Installation",
                "Are you ready to proceed with the installation?\n\n"
                f"Dry Run: {'Yes' if self.dry_run_check.isChecked() else 'No'}"
            )
            if not reply:
                return

        self._set_busy(True, "Installing...")
        self._append_output("\n" + "=" * 50)
        self._append_output("Starting installation...")
        self.progress_bar.setValue(0)

        self.worker = WorkerThread(
            "execute_plan",
            plan=self.current_plan,
            dry_run=self.dry_run_check.isChecked(),
            gui_config=self.config,
            system_info=self.system_info
        )
        self.worker.output.connect(self._append_output)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self._on_installation_complete)
        self.worker.error.connect(self._on_installation_error)
        self.worker.start()

    def _on_installation_complete(self, result):
        """Handle installation completion"""
        self._set_busy(False, "Complete")

        if result.get("success"):
            self._append_output("\n" + "=" * 50)
            self._append_output("[SUCCESS] Installation completed successfully!")

            if result.get("pending_actions"):
                self._append_output("\n[NOTE] Post-installation actions required:")
                for action in result["pending_actions"]:
                    self._append_output(f"  - {action}")

            self._show_message_box("Success", "Installation completed successfully!", "info")
        else:
            self._append_output("\n[FAILED] Installation failed or was cancelled")
            self._show_message_box("Installation Failed", "Some steps may have failed. Check the output for details.", "warning")

    def _on_installation_error(self, error):
        """Handle installation error"""
        self._set_busy(False, "Error")
        self._append_output(f"\n[ERROR] Installation error: {error}")
        self._show_message_box("Installation Error", error, "critical")

    def _show_settings(self):
        wizard = SetupWizard(self)
        if wizard.exec() == QDialog.Accepted:
            # Reload config
            self.config = self.config_manager.config
            self.web_search_check.setChecked(self.config.app.web_search_enabled)
            self.status_bar.showMessage("Settings saved!")

    def _show_about(self):
        QMessageBox.about(
            self, "About AI Installer",
            "<h2>AI Installer</h2>"
            "<p>An AI-powered software installation agent.</p>"
            "<p>Uses LLMs to analyze, plan, and execute software installations "
            "with intelligent troubleshooting.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Supports local LLMs (LM Studio, Ollama, etc.)</li>"
            "<li>Supports cloud APIs (OpenAI, Anthropic)</li>"
            "<li>Real-time web search for docs</li>"
            "<li>Real package size calculations</li>"
            "<li>Interactive Q&A during planning</li>"
            "</ul>"
            "<p><b>License:</b> LGPL v3 (PySide6)</p>"
        )

    def _show_message_box(self, title: str, message: str, level: str = "info"):
        """Show a message box with the given title and message"""
        if level == "critical":
            QMessageBox.critical(self, title, message)
        elif level == "warning":
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.information(self, title, message)

    def _show_confirm_dialog(self, title: str, message: str) -> bool:
        """Show a confirmation dialog and return True if confirmed"""
        reply = QMessageBox.question(
            self, title, message,
            QMessageBox.Yes | QMessageBox.No
        )
        return reply == QMessageBox.Yes

    def _open_github(self):
        """Open GitHub repository - with fallback for WSL/headless environments"""
        import subprocess
        url = "https://github.com/vehoelite/AI_Installer"

        # Try xdg-open first (works better in WSL2)
        try:
            subprocess.Popen(['xdg-open', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            pass

        # Try wslview for WSL2
        try:
            subprocess.Popen(['wslview', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            pass

        # Fallback to Qt's default method
        if not QDesktopServices.openUrl(QUrl(url)):
            # If all else fails, show the URL to the user
            QMessageBox.information(
                self, "GitHub Repository",
                f"Please open this URL in your browser:\n\n{url}"
            )


# =============================================================================
# Application Entry Point
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI Installer")
    app.setOrganizationName("AI Installer")

    # Apply dark theme
    app.setStyleSheet(DARK_THEME)

    # Check for first run
    config_manager = get_config_manager()
    if config_manager.is_first_run():
        wizard = SetupWizard()
        if wizard.exec() != QDialog.Accepted:
            sys.exit(0)

    # Show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
