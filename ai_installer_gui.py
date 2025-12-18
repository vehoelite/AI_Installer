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
        QSizePolicy, QSpacerItem, QFormLayout, QGridLayout,
        QFileDialog
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
    from ai_installer_cli import (
        AgentConfig, LLMProvider, OpenAICompatibleConfig,
        SystemAnalyzer, SystemInfo, InstallationReasoner,
        CommandExecutor, TestRunner, PlanEnhancer, WebSearcher,
        AnthropicLLM, OpenAILLM, OpenAICompatibleLLM, GeminiLLM
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
    subcontrol-origin: padding;
    subcontrol-position: right center;
    width: 30px;
    border-left: 1px solid #45475a;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
    background-color: #45475a;
}

QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #cdd6f4;
}

QComboBox:hover::drop-down {
    background-color: #585b70;
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

class GeminiWorkerThread(QThread):
    """Background worker thread for Gemini API operations"""
    finished_signal = Signal(dict)

    def __init__(self, operation: str, api_key: str, model: str):
        super().__init__()
        self.operation = operation  # "test" or "fetch"
        self.api_key = api_key
        self.model = model

    def run(self):
        try:
            llm = GeminiLLM(self.api_key, self.model)
            models = llm.list_models()
            self.finished_signal.emit({"success": True, "models": models})
        except Exception as e:
            self.finished_signal.emit({"success": False, "error": str(e)})


class WorkerThread(QThread):
    """Background worker thread for LLM and system operations"""
    output = Signal(str)  # Emit output text
    progress = Signal(int)  # Emit progress percentage
    finished_signal = Signal(dict)  # Emit result when done
    error = Signal(str)  # Emit error message
    step_confirmation_needed = Signal(str, int)  # Emit step description and step number
    credentials_needed = Signal(list, str)  # Emit list of required fields and software name

    def __init__(self, task_type: str, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs
        self._is_cancelled = False
        self._step_confirmed = False
        self._waiting_for_confirmation = False
        self._waiting_for_credentials = False
        self._credentials = {}  # Filled by GUI when credentials dialog completes

    def cancel(self):
        self._is_cancelled = True
        self._step_confirmed = False  # Also break out of confirmation wait

    def confirm_step(self):
        """Called by GUI to confirm current step"""
        self._step_confirmed = True

    def skip_step(self):
        """Called by GUI to skip current step"""
        self._step_confirmed = False
        self._waiting_for_confirmation = False

    def set_credentials(self, credentials: dict):
        """Called by GUI when credentials are provided"""
        self._credentials = credentials
        self._waiting_for_credentials = False

    def cancel_credentials(self):
        """Called by GUI when credentials dialog is cancelled"""
        self._credentials = {}
        self._waiting_for_credentials = False

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
            elif config.llm.provider == "gemini":
                # Test Gemini connection
                gemini_llm = GeminiLLM(config.llm.gemini_api_key, config.llm.gemini_model)
                models = gemini_llm.list_models()
                self.finished_signal.emit({
                    "success": True,
                    "message": f"Gemini API connected ({len(models)} models available)",
                    "models": models
                })
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
        elif config.llm.provider == "gemini":
            return GeminiLLM(config.llm.gemini_api_key, config.llm.gemini_model)
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
        """Execute installation plan with immediate auto-repair on failures"""
        plan = self.kwargs.get("plan")
        dry_run = self.kwargs.get("dry_run", False)
        system_info = self.kwargs.get("system_info")
        gui_config = self.kwargs.get("gui_config")

        # Get settings
        confirm_each_step = gui_config.app.confirm_each_step if gui_config else False
        auto_repair_enabled = gui_config.app.auto_repair_enabled if gui_config else True

        if not plan:
            self.error.emit("No plan provided")
            return

        if not CORE_AVAILABLE:
            self.error.emit("Core modules not available")
            return

        executor = CommandExecutor(dry_run=dry_run, verbose=True)
        steps = plan.get("installation_steps", [])
        total_steps = len(steps)

        self.output.emit(f"Starting installation ({total_steps} steps)...")
        if dry_run:
            self.output.emit("  [DRY RUN MODE - Commands will not be executed]")
        if confirm_each_step:
            self.output.emit("  [STEP-BY-STEP MODE - Will pause before each step]")

        # Check if credentials are needed and prompt for them
        required_credentials = self._detect_required_credentials(plan)
        if required_credentials and not dry_run:
            software_name = plan.get("software_name", "")
            self.output.emit(f"\n[*] This installation requires credentials...")
            self._waiting_for_credentials = True
            self.credentials_needed.emit(required_credentials, software_name)

            # Wait for credentials
            timeout_counter = 0
            while self._waiting_for_credentials and not self._is_cancelled:
                self.msleep(100)
                timeout_counter += 1
                if timeout_counter > 3000:  # 5 minute timeout
                    self.error.emit("Credentials input timed out")
                    return

            if self._is_cancelled or not self._credentials:
                self.output.emit("[!] Installation cancelled - no credentials provided")
                self.finished_signal.emit({"success": False, "results": [], "pending_actions": []})
                return

            # Replace placeholders in commands with actual credentials
            self.output.emit("[OK] Credentials received, updating configuration...")
            plan = self._apply_credentials_to_plan(plan, self._credentials)
            steps = plan.get("installation_steps", [])

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
        skipped_steps = []
        current_working_dir = None  # Track working directory for cd commands

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

            self.output.emit(f"\n[Step {step_num}/{total_steps}] {description}")

            # Step-by-step confirmation if enabled
            if confirm_each_step and not dry_run:
                self.output.emit("   Waiting for confirmation...")
                self._waiting_for_confirmation = True
                self._step_confirmed = False
                self.step_confirmation_needed.emit(description, step_num)

                # Wait for confirmation (polling with timeout)
                timeout_counter = 0
                while self._waiting_for_confirmation and not self._is_cancelled:
                    self.msleep(100)  # Sleep 100ms
                    timeout_counter += 1
                    if self._step_confirmed:
                        self._waiting_for_confirmation = False
                        break
                    if timeout_counter > 6000:  # 10 minute timeout
                        self.output.emit("   [TIMEOUT] Step skipped due to no response")
                        skipped_steps.append(step_num)
                        continue

                if self._is_cancelled:
                    break
                if not self._step_confirmed:
                    self.output.emit("   [SKIPPED] User skipped this step")
                    skipped_steps.append(step_num)
                    continue

            # Join multi-line commands (commands with trailing \)
            joined_commands = self._join_multiline_commands(commands)

            step_failed = False
            for cmd in joined_commands:
                if self._is_cancelled or step_failed:
                    break

                # Fix common shell compatibility issues
                cmd = self._fix_shell_command(cmd, current_working_dir)

                # Track cd commands for working directory
                if cmd.strip().startswith("cd "):
                    new_dir = cmd.strip()[3:].strip()
                    if new_dir.startswith("/"):
                        current_working_dir = new_dir
                    elif current_working_dir:
                        current_working_dir = f"{current_working_dir}/{new_dir}"
                    else:
                        current_working_dir = new_dir
                    self.output.emit(f"   $ {cmd}")
                    self.output.emit(f"   [OK] Working directory set to: {current_working_dir}")
                    continue  # cd is handled by tracking, not executing

                # Display command (truncate if too long)
                display_cmd = cmd if len(cmd) < 100 else cmd[:97] + "..."
                self.output.emit(f"   $ {display_cmd}")

                # Execute with current working directory (if supported by executor)
                try:
                    result = executor.execute(cmd, requires_sudo=requires_sudo, cwd=current_working_dir)
                except TypeError:
                    # Fallback for older CommandExecutor without cwd support
                    # Prepend cd to command if we have a working directory
                    if current_working_dir:
                        cmd = f"cd {current_working_dir} && {cmd}"
                    result = executor.execute(cmd, requires_sudo=requires_sudo)
                results.append(result)

                if result.skipped:
                    self.output.emit(f"   [SKIP] {result.skip_reason[:80]}")
                elif result.success:
                    self.output.emit(f"   [OK] Completed ({result.duration_seconds:.1f}s)")
                    if result.stdout and len(result.stdout) < 200:
                        for line in result.stdout.split('\n')[:3]:
                            self.output.emit(f"   {line}")
                elif self._is_acceptable_exit_code(cmd, result):
                    # Some commands return non-zero but aren't failures
                    self.output.emit(f"   [OK] Completed (exit code {result.return_code} is acceptable for this command)")
                    if result.stdout and len(result.stdout) < 200:
                        for line in result.stdout.split('\n')[:3]:
                            self.output.emit(f"   {line}")
                elif self._should_check_docker_container(cmd, result):
                    # Docker ps shows no containers - check what happened
                    container_name = self._extract_container_name_from_context(cmd, joined_commands)
                    if container_name:
                        self.output.emit(f"   [*] Checking container status: {container_name}")
                        is_running, status_msg = self._check_docker_container_status(container_name, executor)
                        if is_running:
                            self.output.emit(f"   [OK] {status_msg}")
                        else:
                            self.output.emit(f"   [!] {status_msg}")
                            # This is a real failure - container isn't running
                            failed_steps.append({
                                "step": step_num,
                                "command": cmd,
                                "error": status_msg,
                                "return_code": result.return_code
                            })
                            step_failed = True
                else:
                    self.output.emit(f"   [FAIL] Exit code {result.return_code}")
                    if result.stderr:
                        self.output.emit(f"   Error: {result.stderr[:200]}")

                    # IMMEDIATE auto-repair attempt
                    if auto_repair_enabled and not dry_run:
                        self.output.emit("\n   [*] Attempting immediate repair...")
                        repair_success = self._attempt_immediate_repair(
                            executor, cmd, result, system_info, current_working_dir, requires_sudo
                        )
                        if repair_success:
                            self.output.emit("   [OK] Repair successful, continuing...")
                        else:
                            self.output.emit("   [!] Repair failed. Pausing step execution.")
                            failed_steps.append({
                                "step": step_num,
                                "command": cmd,
                                "error": result.stderr or result.stdout,
                                "return_code": result.return_code
                            })
                            step_failed = True
                    else:
                        failed_steps.append({
                            "step": step_num,
                            "command": cmd,
                            "error": result.stderr or result.stdout,
                            "return_code": result.return_code
                        })
                        if not auto_repair_enabled:
                            self.output.emit("   [!] Auto-repair disabled. Continuing to next command...")
                        step_failed = True

        # Run post-install tests
        if not self._is_cancelled and plan.get("post_install_tests"):
            self.output.emit("\n[*] Running post-installation tests...")
            test_runner = TestRunner(executor, verbose=True)
            test_results = test_runner.run_tests(plan["post_install_tests"])
            self.output.emit(f"   Tests: {test_results['passed']}/{test_results['total']} passed")

            # Show detailed test failures
            if test_results.get('failed', 0) > 0:
                self.output.emit("\n   Failed tests:")
                for test in test_results.get('results', []):
                    if not test.get('passed', True):
                        test_name = test.get('name', test.get('command', 'Unknown test'))
                        self.output.emit(f"   [X] {test_name}")
                        if test.get('output'):
                            # Show first 100 chars of output
                            output = test['output'][:100].replace('\n', ' ')
                            self.output.emit(f"       {output}...")

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

    def _detect_required_credentials(self, plan: dict) -> list:
        """Detect if the installation plan requires user credentials"""
        import re
        required_fields = []
        seen_fields = set()

        # Common credential patterns in Docker/compose configurations
        credential_patterns = [
            # Username patterns
            (r'(?:ADMIN_USER|ADMIN_USERNAME|DB_USER|MYSQL_USER|POSTGRES_USER|USER)=(\$?\{?[A-Z_]+\}?|admin|root|user)',
             "admin_user", "Admin Username", "text", "admin"),
            # Password patterns - detect placeholders
            (r'(?:ADMIN_PASSWORD|ADMIN_PASS|DB_PASSWORD|MYSQL_PASSWORD|MYSQL_ROOT_PASSWORD|POSTGRES_PASSWORD|PASSWORD)=(\$?\{?[A-Z_]+\}?|password|changeme|secret)',
             "admin_password", "Admin Password", "password", ""),
            # Database name
            (r'(?:DB_NAME|MYSQL_DATABASE|POSTGRES_DB|DATABASE)=(\$?\{?[A-Z_]+\}?|database|mydb)',
             "db_name", "Database Name", "text", "nextcloud"),
            # Domain/hostname
            (r'(?:DOMAIN|HOSTNAME|VIRTUAL_HOST|NEXTCLOUD_TRUSTED_DOMAINS)=(\$?\{?[A-Z_]+\}?|localhost|example\.com)',
             "domain", "Domain/Hostname", "text", "localhost"),
        ]

        # Check all commands in all steps
        all_commands = []
        for step in plan.get("installation_steps", []):
            for cmd in step.get("commands", []):
                all_commands.append(cmd)

        all_text = "\n".join(all_commands)

        for pattern, field_name, label, field_type, default in credential_patterns:
            if re.search(pattern, all_text, re.IGNORECASE):
                if field_name not in seen_fields:
                    seen_fields.add(field_name)
                    required_fields.append({
                        "name": field_name,
                        "label": label,
                        "type": field_type,
                        "default": default,
                        "hint": f"Enter the {label.lower()}"
                    })

        # Also check for explicit placeholder markers
        placeholder_patterns = [
            (r'USERNAME', "admin_user", "Admin Username", "text", "admin"),
            (r'PASSWORD', "admin_password", "Admin Password", "password", ""),
            (r'YOUR_PASSWORD', "admin_password", "Admin Password", "password", ""),
            (r'REPLACE_WITH', "custom_value", "Required Value", "text", ""),
        ]

        for pattern, field_name, label, field_type, default in placeholder_patterns:
            if pattern in all_text and field_name not in seen_fields:
                seen_fields.add(field_name)
                required_fields.append({
                    "name": field_name,
                    "label": label,
                    "type": field_type,
                    "default": default,
                    "hint": f"Enter the {label.lower()}"
                })

        return required_fields

    def _apply_credentials_to_plan(self, plan: dict, credentials: dict) -> dict:
        """Replace credential placeholders in plan commands with actual values"""
        import copy
        import re

        plan = copy.deepcopy(plan)

        # Mapping of credential names to environment variable patterns
        replacements = {
            "admin_user": [
                (r'(ADMIN_USER(?:NAME)?=)[^\s"\']+', r'\g<1>' + credentials.get("admin_user", "admin")),
                (r'(NEXTCLOUD_ADMIN_USER=)[^\s"\']+', r'\g<1>' + credentials.get("admin_user", "admin")),
                (r'(MYSQL_USER=)[^\s"\']+', r'\g<1>' + credentials.get("admin_user", "admin")),
                (r'(POSTGRES_USER=)[^\s"\']+', r'\g<1>' + credentials.get("admin_user", "admin")),
            ],
            "admin_password": [
                (r'(ADMIN_PASS(?:WORD)?=)[^\s"\']+', r'\g<1>' + credentials.get("admin_password", "")),
                (r'(NEXTCLOUD_ADMIN_PASSWORD=)[^\s"\']+', r'\g<1>' + credentials.get("admin_password", "")),
                (r'(MYSQL_PASSWORD=)[^\s"\']+', r'\g<1>' + credentials.get("admin_password", "")),
                (r'(MYSQL_ROOT_PASSWORD=)[^\s"\']+', r'\g<1>' + credentials.get("admin_password", "")),
                (r'(POSTGRES_PASSWORD=)[^\s"\']+', r'\g<1>' + credentials.get("admin_password", "")),
            ],
            "db_name": [
                (r'(DB_NAME=)[^\s"\']+', r'\g<1>' + credentials.get("db_name", "database")),
                (r'(MYSQL_DATABASE=)[^\s"\']+', r'\g<1>' + credentials.get("db_name", "database")),
                (r'(POSTGRES_DB=)[^\s"\']+', r'\g<1>' + credentials.get("db_name", "database")),
            ],
            "domain": [
                (r'(DOMAIN=)[^\s"\']+', r'\g<1>' + credentials.get("domain", "localhost")),
                (r'(VIRTUAL_HOST=)[^\s"\']+', r'\g<1>' + credentials.get("domain", "localhost")),
                (r'(NEXTCLOUD_TRUSTED_DOMAINS=)[^\s"\']+', r'\g<1>' + credentials.get("domain", "localhost")),
            ],
        }

        for step in plan.get("installation_steps", []):
            new_commands = []
            for cmd in step.get("commands", []):
                modified_cmd = cmd
                for cred_name, patterns in replacements.items():
                    if cred_name in credentials:
                        for pattern, replacement in patterns:
                            modified_cmd = re.sub(pattern, replacement, modified_cmd, flags=re.IGNORECASE)
                new_commands.append(modified_cmd)
            step["commands"] = new_commands

        return plan

    def _is_acceptable_exit_code(self, cmd: str, result) -> bool:
        """Check if a non-zero exit code is actually acceptable for this command"""
        cmd_lower = cmd.lower().strip()

        # grep returns 1 when no matches found - this is often acceptable in checks
        if '| grep' in cmd_lower or cmd_lower.startswith('grep '):
            if result.return_code == 1:
                return True  # No matches is not a failure

        # diff returns 1 when files differ - often used for comparison checks
        if 'diff ' in cmd_lower and result.return_code == 1:
            return True

        # test/[ commands return 1 when condition is false
        if cmd_lower.startswith('test ') or cmd_lower.startswith('[ '):
            return True

        # which/command -v returns 1 when command not found (often used to check if something exists)
        if cmd_lower.startswith('which ') or 'command -v' in cmd_lower:
            return True

        # ss/netstat with grep - exit 1 means port not listening (might be early, not failure)
        if ('ss ' in cmd_lower or 'netstat ' in cmd_lower) and '| grep' in cmd_lower:
            if result.return_code == 1:
                self.output.emit("   [NOTE] Port not yet listening - container may still be starting")
                return True

        return False

    def _should_check_docker_container(self, cmd: str, result) -> bool:
        """Check if this is a docker ps command that shows no running containers"""
        cmd_lower = cmd.lower().strip()
        if 'docker ps' in cmd_lower and result.return_code == 0:
            # Check if output is just headers (no containers running)
            lines = [l for l in (result.stdout or '').strip().split('\n') if l.strip()]
            if len(lines) <= 1:  # Only header line or empty
                return True
        return False

    def _extract_container_name_from_context(self, current_cmd: str, all_commands: list) -> str:
        """Try to extract container name from docker commands"""
        import re

        # First check current command for --filter name=
        match = re.search(r'--filter\s+name=(\S+)', current_cmd)
        if match:
            return match.group(1)

        # Look through previous commands for docker run --name
        for cmd in all_commands:
            if 'docker run' in cmd:
                match = re.search(r'--name\s+(\S+)', cmd)
                if match:
                    return match.group(1)

        # Look for docker-compose project names
        for cmd in all_commands:
            if 'docker-compose' in cmd or 'docker compose' in cmd:
                # Try to find -p or --project-name
                match = re.search(r'(?:-p|--project-name)\s+(\S+)', cmd)
                if match:
                    return match.group(1)

        return None

    def _check_docker_container_status(self, container_name: str, executor) -> tuple:
        """Check if a docker container is running and get diagnostics if not"""
        # Check if container exists and its status
        try:
            result = executor.execute(f"docker ps -a --filter name={container_name} --format '{{{{.Status}}}}'")
        except TypeError:
            result = executor.execute(f"docker ps -a --filter name={container_name} --format '{{{{.Status}}}}'")

        status = (result.stdout or '').strip()

        if not status:
            return False, "Container does not exist"

        if 'Up' in status:
            return True, f"Container is running: {status}"

        if 'Exited' in status:
            # Get logs to see why it exited
            try:
                logs_result = executor.execute(f"docker logs --tail 20 {container_name}")
            except TypeError:
                logs_result = executor.execute(f"docker logs --tail 20 {container_name}")

            logs = logs_result.stdout or logs_result.stderr or "No logs available"
            return False, f"Container exited. Last logs:\n{logs[:500]}"

        return False, f"Container status: {status}"

    def _fix_shell_command(self, cmd: str, cwd: str = None) -> str:
        """Fix common shell compatibility issues in commands"""
        original_cmd = cmd.strip()

        # Fix venv activation - activation in subprocess doesn't persist
        # Instead of activating, we use the venv's binaries directly
        if "bin/activate" in original_cmd:
            if original_cmd.startswith("source ") or original_cmd.startswith(". "):
                # Determine venv path
                if ".venv/bin/activate" in original_cmd:
                    venv_name = ".venv"
                elif "venv/bin/activate" in original_cmd:
                    venv_name = "venv"
                else:
                    # Extract venv path from command
                    import re
                    match = re.search(r'source\s+(\S+)/bin/activate', original_cmd) or \
                            re.search(r'\.\s+(\S+)/bin/activate', original_cmd)
                    venv_name = match.group(1) if match else ".venv"

                self.output.emit(f"   [NOTE] Virtual env activation converted to direct binary usage")
                # Return a command that verifies the venv exists
                if cwd:
                    return f"test -d {cwd}/{venv_name}/bin && echo 'Virtual environment {venv_name} ready at {cwd}'"
                return f"test -d {venv_name}/bin && echo 'Virtual environment {venv_name} ready'"

        # Fix pip commands to use venv versions when in a project directory
        if cwd and (original_cmd.startswith("pip ") or original_cmd.startswith("pip3 ")):
            pip_args = original_cmd.split(None, 1)[1] if ' ' in original_cmd else ""
            # Prioritize venv pip
            venv_pip = f"{cwd}/.venv/bin/pip"
            venv_pip_alt = f"{cwd}/venv/bin/pip"
            cmd = f"if [ -f {venv_pip} ]; then {venv_pip} {pip_args}; " \
                  f"elif [ -f {venv_pip_alt} ]; then {venv_pip_alt} {pip_args}; " \
                  f"else {original_cmd}; fi"
            self.output.emit(f"   [FIX] Using venv pip if available")
            return cmd

        # Fix python commands to use venv versions
        if cwd and (original_cmd.startswith("python ") or original_cmd.startswith("python3 ")):
            python_args = original_cmd.split(None, 1)[1] if ' ' in original_cmd else ""
            venv_python = f"{cwd}/.venv/bin/python"
            venv_python_alt = f"{cwd}/venv/bin/python"
            cmd = f"if [ -f {venv_python} ]; then {venv_python} {python_args}; " \
                  f"elif [ -f {venv_python_alt} ]; then {venv_python_alt} {python_args}; " \
                  f"else {original_cmd}; fi"
            self.output.emit(f"   [FIX] Using venv python if available")
            return cmd

        return cmd

    def _attempt_immediate_repair(self, executor, failed_cmd: str, result, system_info, cwd: str, requires_sudo: bool) -> bool:
        """Attempt immediate repair of a failed command before continuing"""
        try:
            llm = self._get_llm()
            reasoner = InstallationReasoner(llm)

            error_msg = result.stderr or result.stdout or "Unknown error"

            self.output.emit(f"   [*] Analyzing failure: {error_msg[:100]}...")

            # Web search for solutions (Stack Overflow, GitHub issues, etc.)
            web_solutions = None
            try:
                self.output.emit("   [*] Searching web for solutions...")
                searcher = WebSearcher(verbose=False)

                # Create a focused search query from the error
                search_query = self._create_error_search_query(failed_cmd, error_msg, system_info)
                web_results = searcher.search_error_solutions(search_query, error_msg)

                if web_results and web_results.get("fetched_docs"):
                    num_results = len(web_results["fetched_docs"])
                    self.output.emit(f"   [OK] Found {num_results} potential solution(s) online")
                    web_solutions = web_results
                else:
                    self.output.emit("   [*] No web results found, using AI knowledge...")
            except Exception as e:
                self.output.emit(f"   [*] Web search unavailable: {str(e)[:50]}, using AI knowledge...")

            # Build context with web solutions if available
            context = f"Command failed with exit code {result.return_code}. Working directory: {cwd or 'default'}"
            if web_solutions and web_solutions.get("fetched_docs"):
                context += "\n\nWEB SEARCH RESULTS (Stack Overflow, GitHub, docs):\n"
                for i, doc in enumerate(web_solutions["fetched_docs"][:3], 1):
                    title = doc.get("title", "Unknown")
                    content = doc.get("content", "")[:1500]  # Limit content size
                    url = doc.get("url", "")
                    context += f"\n--- Solution {i}: {title} ---\n"
                    context += f"URL: {url}\n"
                    context += f"{content}\n"

            # Get troubleshooting advice from LLM (enhanced with web solutions)
            fix_plan = reasoner.troubleshoot(
                system_info,
                error_msg,
                failed_cmd,
                context
            )

            if fix_plan.get("_parse_error"):
                self.output.emit("   [!] Could not parse repair suggestions")
                return False

            root_cause = fix_plan.get('root_cause', 'Unknown')
            self.output.emit(f"   [*] Root cause: {root_cause[:80]}")

            # Execute fix steps
            fix_steps = fix_plan.get("fix_steps", [])
            if not fix_steps:
                self.output.emit("   [!] No automatic fix available")
                if fix_plan.get("alternative_approaches"):
                    self.output.emit("   [*] Suggestions:")
                    for alt in fix_plan["alternative_approaches"][:2]:
                        self.output.emit(f"       - {alt[:70]}")
                return False

            self.output.emit(f"   [*] Attempting {len(fix_steps)} repair action(s)...")

            for fix_step in fix_steps:
                fix_commands = fix_step.get("commands", [])
                for fix_cmd in fix_commands:
                    # Apply same shell fixes
                    fix_cmd = self._fix_shell_command(fix_cmd, cwd)

                    # Skip if it's just a cd command
                    if fix_cmd.strip().startswith("cd "):
                        continue

                    display_cmd = fix_cmd if len(fix_cmd) < 60 else fix_cmd[:57] + "..."
                    self.output.emit(f"   $ {display_cmd}")

                    try:
                        fix_result = executor.execute(
                            fix_cmd,
                            requires_sudo=fix_step.get("requires_sudo", requires_sudo),
                            cwd=cwd
                        )
                    except TypeError:
                        # Fallback for older CommandExecutor without cwd support
                        if cwd:
                            fix_cmd = f"cd {cwd} && {fix_cmd}"
                        fix_result = executor.execute(
                            fix_cmd,
                            requires_sudo=fix_step.get("requires_sudo", requires_sudo)
                        )

                    if fix_result.success:
                        self.output.emit("   [OK] Repair step completed")
                    else:
                        self.output.emit(f"   [FAIL] Repair step failed: {fix_result.stderr[:50] if fix_result.stderr else 'unknown'}")

            # Retry the original command if suggested
            if fix_plan.get("should_retry_original", True):
                self.output.emit(f"   [*] Retrying original command...")
                try:
                    retry_result = executor.execute(failed_cmd, requires_sudo=requires_sudo, cwd=cwd)
                except TypeError:
                    # Fallback for older CommandExecutor without cwd support
                    retry_cmd = f"cd {cwd} && {failed_cmd}" if cwd else failed_cmd
                    retry_result = executor.execute(retry_cmd, requires_sudo=requires_sudo)
                if retry_result.success:
                    self.output.emit("   [OK] Original command now succeeded!")
                    return True
                else:
                    self.output.emit(f"   [FAIL] Original command still failing: {retry_result.stderr[:50] if retry_result.stderr else ''}")
                    return False

            return False

        except Exception as e:
            self.output.emit(f"   [!] Repair error: {str(e)[:80]}")
            return False

    def _create_error_search_query(self, failed_cmd: str, error_msg: str, system_info) -> str:
        """
        Create a focused search query from a failed command and error message.
        Optimized for finding solutions on Stack Overflow, GitHub, and forums.
        """
        import re

        # Extract the command name/tool (first word of command)
        cmd_parts = failed_cmd.strip().split()
        primary_tool = cmd_parts[0] if cmd_parts else "command"

        # Handle sudo prefix
        if primary_tool == "sudo" and len(cmd_parts) > 1:
            primary_tool = cmd_parts[1]

        # Clean up error message - extract the most relevant part
        error_clean = error_msg.strip()

        # Remove ANSI escape codes
        error_clean = re.sub(r'\x1b\[[0-9;]*m', '', error_clean)

        # Extract key error patterns
        error_patterns = [
            r'(?:error|Error|ERROR)[:\s]+(.+?)(?:\n|$)',
            r'(?:failed|Failed|FAILED)[:\s]+(.+?)(?:\n|$)',
            r'(?:cannot|Cannot|CANNOT)\s+(.+?)(?:\n|$)',
            r'(?:No such file or directory)[:\s]*(.+?)(?:\n|$)',
            r'(?:Permission denied)[:\s]*(.+?)(?:\n|$)',
            r'(?:not found)[:\s]*(.+?)(?:\n|$)',
            r'(?:command not found)[:\s]*(.+?)(?:\n|$)',
            r'(?:ModuleNotFoundError|ImportError)[:\s]+(.+?)(?:\n|$)',
            r'(?:E:|E\s+)[:\s]+(.+?)(?:\n|$)',  # apt/pip errors
        ]

        key_error = ""
        for pattern in error_patterns:
            match = re.search(pattern, error_clean, re.IGNORECASE)
            if match:
                key_error = match.group(0).strip()[:100]
                break

        # If no pattern matched, take first meaningful line
        if not key_error:
            lines = [l.strip() for l in error_clean.split('\n') if l.strip() and not l.strip().startswith('#')]
            if lines:
                # Skip lines that are just progress or status
                for line in lines:
                    if any(x in line.lower() for x in ['error', 'fail', 'cannot', 'denied', 'not found']):
                        key_error = line[:100]
                        break
                if not key_error:
                    key_error = lines[-1][:100]  # Often the last line has the error

        # Get distro info for more relevant results
        distro = ""
        if hasattr(system_info, 'distro_name'):
            distro = system_info.distro_name.lower()
        elif hasattr(system_info, 'os_name'):
            distro = system_info.os_name.lower()

        # Build search query - prioritize Stack Overflow style
        query_parts = []

        # Add distro for context if it's Linux
        if distro and 'ubuntu' in distro:
            query_parts.append("ubuntu")
        elif distro and 'debian' in distro:
            query_parts.append("debian")
        elif distro and any(x in distro for x in ['fedora', 'rhel', 'centos', 'rocky']):
            query_parts.append("linux")

        # Add the tool/command name
        query_parts.append(primary_tool)

        # Add cleaned error message
        if key_error:
            # Remove common noise from error
            key_error = re.sub(r'^\d+[\.:]\s*', '', key_error)  # Remove line numbers
            key_error = re.sub(r'^[\s\-\*]+', '', key_error)  # Remove leading dashes/bullets
            query_parts.append(f'"{key_error}"')  # Quote for exact match

        # Construct final query
        search_query = ' '.join(query_parts)

        # Ensure reasonable length
        if len(search_query) > 150:
            search_query = search_query[:150]

        return search_query

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
# Step Confirmation Dialog
# =============================================================================

class CredentialsDialog(QDialog):
    """Dialog to collect required credentials for Docker installations"""

    def __init__(self, required_fields: list, software_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Credentials Required")
        self.setMinimumWidth(450)
        self.credentials = {}
        self.field_widgets = {}
        self._init_ui(required_fields, software_name)

    def _init_ui(self, required_fields: list, software_name: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel(f"<b>Credentials Required{' for ' + software_name if software_name else ''}</b>")
        header.setObjectName("section")
        layout.addWidget(header)

        info = QLabel(
            "This installation requires you to set credentials. "
            "Please provide the following values:"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Form for credentials
        form_layout = QFormLayout()

        for field in required_fields:
            field_name = field.get("name", "unknown")
            field_label = field.get("label", field_name)
            field_type = field.get("type", "text")  # "text" or "password"
            field_default = field.get("default", "")
            field_hint = field.get("hint", "")

            if field_type == "password":
                widget = QLineEdit()
                widget.setEchoMode(QLineEdit.Password)
                widget.setPlaceholderText(field_hint or "Enter a secure password")
            else:
                widget = QLineEdit()
                widget.setText(field_default)
                widget.setPlaceholderText(field_hint or f"Enter {field_label}")

            self.field_widgets[field_name] = widget
            form_layout.addRow(f"{field_label}:", widget)

        layout.addLayout(form_layout)

        # Security note
        note = QLabel(
            "<i>Note: These values will be used in the Docker configuration. "
            "Choose strong passwords for production use.</i>"
        )
        note.setWordWrap(True)
        note.setObjectName("subtitle")
        layout.addWidget(note)

        layout.addSpacing(10)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        continue_btn = QPushButton("Continue")
        continue_btn.setObjectName("primary")
        continue_btn.clicked.connect(self._on_continue)
        button_layout.addWidget(continue_btn)

        layout.addLayout(button_layout)

    def _on_continue(self):
        # Collect all values
        for field_name, widget in self.field_widgets.items():
            value = widget.text().strip()
            if not value:
                self._show_error(f"Please fill in all required fields")
                return
            self.credentials[field_name] = value
        self.accept()

    def _show_error(self, message: str):
        QMessageBox.warning(self, "Missing Information", message)

    def get_credentials(self) -> dict:
        return self.credentials


class StepConfirmDialog(QDialog):
    """Dialog to confirm individual installation steps"""

    def __init__(self, step_description: str, step_num: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Confirm Step {step_num}")
        self.setMinimumWidth(450)
        self.result_action = None  # "execute", "skip", or "cancel"
        self._init_ui(step_description, step_num)

    def _init_ui(self, description: str, step_num: int):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Step info
        header = QLabel(f"<b>Step {step_num}</b>")
        header.setObjectName("section")
        layout.addWidget(header)

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        layout.addSpacing(10)

        # Buttons
        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel Installation")
        cancel_btn.setObjectName("danger")
        cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(cancel_btn)

        button_layout.addStretch()

        skip_btn = QPushButton("Skip This Step")
        skip_btn.clicked.connect(self._on_skip)
        button_layout.addWidget(skip_btn)

        execute_btn = QPushButton("Execute Step")
        execute_btn.setObjectName("success")
        execute_btn.clicked.connect(self._on_execute)
        execute_btn.setDefault(True)
        button_layout.addWidget(execute_btn)

        layout.addLayout(button_layout)

    def _on_execute(self):
        self.result_action = "execute"
        self.accept()

    def _on_skip(self):
        self.result_action = "skip"
        self.accept()

    def _on_cancel(self):
        self.result_action = "cancel"
        self.reject()


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
        self.provider_combo.addItems(["Local LLM (LM Studio, Ollama, etc.)", "OpenAI API", "Anthropic API", "Google Gemini API"])
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
        self.openai_model.addItems(["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"])
        self.openai_model.setEditable(False)  # Dropdown only - prevents typos
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
        self.anthropic_model.addItems([
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307"
        ])
        self.anthropic_model.setEditable(False)  # Dropdown only - prevents typos
        anthropic_layout.addRow("Model:", self.anthropic_model)

        self.settings_stack.addWidget(anthropic_widget)

        # Gemini settings
        gemini_widget = QWidget()
        gemini_layout = QFormLayout(gemini_widget)

        self.gemini_key = QLineEdit()
        self.gemini_key.setPlaceholderText("AIza...")
        self.gemini_key.setEchoMode(QLineEdit.Password)
        gemini_layout.addRow("API Key:", self.gemini_key)

        gemini_key_link = QLabel('<a href="https://aistudio.google.com/apikey">Get API Key from Google AI Studio</a>')
        gemini_key_link.setOpenExternalLinks(True)
        gemini_layout.addRow("", gemini_key_link)

        # Model dropdown - proper dropdown, not editable text
        self.gemini_model = QComboBox()
        self.gemini_model.setMinimumWidth(300)
        self.gemini_model.addItems([
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ])
        # NOT editable - users must select from dropdown to avoid typos/errors
        self.gemini_model.setEditable(False)
        gemini_layout.addRow("Model:", self.gemini_model)

        # Test connection and fetch models buttons
        gemini_btn_layout = QHBoxLayout()
        self.gemini_test_btn = QPushButton("Test Connection")
        self.gemini_test_btn.setObjectName("primary")
        self.gemini_test_btn.clicked.connect(self._test_gemini_connection)
        gemini_btn_layout.addWidget(self.gemini_test_btn)

        self.gemini_fetch_btn = QPushButton("Fetch Models")
        self.gemini_fetch_btn.clicked.connect(self._fetch_gemini_models)
        gemini_btn_layout.addWidget(self.gemini_fetch_btn)
        gemini_layout.addRow("", gemini_btn_layout)

        self.gemini_status = QLabel("")
        gemini_layout.addRow("", self.gemini_status)

        self.settings_stack.addWidget(gemini_widget)

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

        self.confirm_check = QCheckBox("Confirm before starting installation")
        self.confirm_check.setChecked(True)
        options_layout.addWidget(self.confirm_check)

        self.confirm_each_step_check = QCheckBox("Confirm each step before execution (safer, more control)")
        self.confirm_each_step_check.setChecked(False)
        options_layout.addWidget(self.confirm_each_step_check)

        self.auto_repair_check = QCheckBox("Auto-repair: Attempt to fix failed steps using AI")
        self.auto_repair_check.setChecked(True)
        options_layout.addWidget(self.auto_repair_check)

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
        provider_map = {"local": 0, "openai": 1, "anthropic": 2, "gemini": 3}
        self.provider_combo.setCurrentIndex(provider_map.get(llm.provider, 0))

        # Local settings
        preset_map = {"lm-studio": 0, "ollama": 1, "localai": 2, "custom": 3}
        self.local_preset.setCurrentIndex(preset_map.get(llm.local_preset, 0))
        self.local_host.setText(llm.local_host)
        self.local_port.setValue(llm.local_port)
        self.local_model.setText(llm.local_model)

        # API keys and models
        self.openai_key.setText(llm.openai_api_key)
        self.anthropic_key.setText(llm.anthropic_api_key)
        self.gemini_key.setText(llm.gemini_api_key)

        # Set models if they differ from defaults
        if llm.openai_model:
            self.openai_model.setCurrentText(llm.openai_model)
        if llm.anthropic_model:
            self.anthropic_model.setCurrentText(llm.anthropic_model)
        if llm.gemini_model:
            self.gemini_model.setCurrentText(llm.gemini_model)

        # Options
        self.web_search_check.setChecked(self.config.app.web_search_enabled)
        self.verbose_check.setChecked(self.config.app.verbose_output)
        self.confirm_check.setChecked(self.config.app.confirm_before_execute)
        self.confirm_each_step_check.setChecked(self.config.app.confirm_each_step)
        self.auto_repair_check.setChecked(self.config.app.auto_repair_enabled)

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

    def _test_gemini_connection(self):
        """Test Gemini API connection"""
        api_key = self.gemini_key.text().strip()
        if not api_key:
            self.gemini_status.setText("✗ Please enter an API key first")
            self.gemini_status.setStyleSheet("color: #f38ba8;")
            return

        self.gemini_status.setText("Testing connection...")
        self.gemini_status.setStyleSheet("color: #f9e2af;")
        self.gemini_test_btn.setEnabled(False)
        self.gemini_fetch_btn.setEnabled(False)

        # Use QThread for proper Qt threading
        self._gemini_worker = GeminiWorkerThread("test", api_key, self.gemini_model.currentText())
        self._gemini_worker.finished_signal.connect(self._on_gemini_test_complete)
        self._gemini_worker.start()

    def _on_gemini_test_complete(self, result):
        """Handle Gemini test result"""
        self.gemini_test_btn.setEnabled(True)
        self.gemini_fetch_btn.setEnabled(True)

        if result.get("success"):
            models = result.get("models", [])
            model_count = len(models) if models else 0
            self.gemini_status.setText(f"✓ Connected! {model_count} models available")
            self.gemini_status.setStyleSheet("color: #a6e3a1;")
        else:
            error = result.get("error", "Connection failed")[:60]
            self.gemini_status.setText(f"✗ {error}")
            self.gemini_status.setStyleSheet("color: #f38ba8;")

    def _fetch_gemini_models(self):
        """Fetch available Gemini models and populate dropdown"""
        api_key = self.gemini_key.text().strip()
        if not api_key:
            self.gemini_status.setText("✗ Please enter an API key first")
            self.gemini_status.setStyleSheet("color: #f38ba8;")
            return

        self.gemini_status.setText("Fetching models...")
        self.gemini_status.setStyleSheet("color: #f9e2af;")
        self.gemini_test_btn.setEnabled(False)
        self.gemini_fetch_btn.setEnabled(False)

        # Use QThread for proper Qt threading
        self._gemini_worker = GeminiWorkerThread("fetch", api_key, "gemini-2.0-flash")
        self._gemini_worker.finished_signal.connect(self._on_gemini_fetch_complete)
        self._gemini_worker.start()

    def _on_gemini_fetch_complete(self, result):
        """Handle Gemini model fetch result"""
        self.gemini_test_btn.setEnabled(True)
        self.gemini_fetch_btn.setEnabled(True)

        if result.get("success"):
            models = result.get("models", [])
            if models:
                # Save current selection
                current = self.gemini_model.currentText()

                # Clear and repopulate
                self.gemini_model.clear()
                self.gemini_model.addItems(models)

                # Restore selection if it exists
                index = self.gemini_model.findText(current)
                if index >= 0:
                    self.gemini_model.setCurrentIndex(index)
                else:
                    self.gemini_model.setCurrentIndex(0)

                self.gemini_status.setText(f"✓ Loaded {len(models)} text generation models")
                self.gemini_status.setStyleSheet("color: #a6e3a1;")
            else:
                self.gemini_status.setText("✗ No models found")
                self.gemini_status.setStyleSheet("color: #f38ba8;")
        else:
            error = result.get("error", "Fetch failed")[:60]
            self.gemini_status.setText(f"✗ {error}")
            self.gemini_status.setStyleSheet("color: #f38ba8;")

    def _save_to_config(self):
        """Save UI values to config object"""
        providers = ["local", "openai", "anthropic", "gemini"]
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

        self.config.llm.gemini_api_key = self.gemini_key.text()
        self.config.llm.gemini_model = self.gemini_model.currentText()

        self.config.app.web_search_enabled = self.web_search_check.isChecked()
        self.config.app.verbose_output = self.verbose_check.isChecked()
        self.config.app.confirm_before_execute = self.confirm_check.isChecked()
        self.config.app.confirm_each_step = self.confirm_each_step_check.isChecked()
        self.config.app.auto_repair_enabled = self.auto_repair_check.isChecked()

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
        self.custom_compose_file = None  # Path to user's custom docker-compose.yml
        self.custom_compose_content = None  # Content of the compose file

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

        # Custom compose file button (for Docker installations)
        self.compose_btn = QPushButton("Compose File...")
        self.compose_btn.setToolTip("Use a custom docker-compose.yml file (right-click to clear)")
        self.compose_btn.clicked.connect(self._select_compose_file)
        self.compose_btn.setContextMenuPolicy(Qt.CustomContextMenu)
        self.compose_btn.customContextMenuRequested.connect(self._show_compose_context_menu)
        options_layout.addWidget(self.compose_btn)

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

        # If user has a custom compose file, mention it in the request
        software_with_compose = software
        if self.custom_compose_content:
            self._append_output(f"Using custom compose file: {self.custom_compose_file}")
            software_with_compose = f"{software}\n\n[USER PROVIDED DOCKER COMPOSE FILE - USE THIS EXACTLY]:\n```yaml\n{self.custom_compose_content}\n```"

        self.worker = WorkerThread(
            "create_plan",
            software=software_with_compose,
            system_info=self.system_info,
            web_search=self.web_search_check.isChecked(),
            gui_config=self.config,
            custom_compose_content=self.custom_compose_content
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

        # Note if custom compose was used
        if self.custom_compose_content:
            self._append_output(f"[*] Plan uses your custom compose file: {Path(self.custom_compose_file).name}")
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
        self.worker.step_confirmation_needed.connect(self._on_step_confirmation_needed)
        self.worker.credentials_needed.connect(self._on_credentials_needed)
        self.worker.start()

    def _on_credentials_needed(self, required_fields: list, software_name: str):
        """Handle credentials request from worker thread"""
        dialog = CredentialsDialog(required_fields, software_name, self)
        result = dialog.exec()

        if result == QDialog.Accepted:
            credentials = dialog.get_credentials()
            self._append_output(f"[OK] Credentials provided for: {', '.join(credentials.keys())}")
            self.worker.set_credentials(credentials)
        else:
            self._append_output("[!] Credentials dialog cancelled")
            self.worker.cancel_credentials()

    def _on_step_confirmation_needed(self, description: str, step_num: int):
        """Handle step confirmation request from worker thread"""
        dialog = StepConfirmDialog(description, step_num, self)
        result = dialog.exec()

        if dialog.result_action == "execute":
            self.worker.confirm_step()
        elif dialog.result_action == "skip":
            self.worker.skip_step()
        else:  # cancel
            self.worker.cancel()

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

    def _select_compose_file(self):
        """Let user select a custom docker-compose.yml file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Docker Compose File",
            "",
            "YAML files (*.yml *.yaml);;All files (*.*)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.custom_compose_content = f.read()
                self.custom_compose_file = file_path
                self.compose_btn.setText(f"Compose: {Path(file_path).name}")
                self.compose_btn.setToolTip(f"Using: {file_path}\nClick to change or clear")
                self._append_output(f"\n[*] Custom compose file loaded: {file_path}")
                self._append_output(f"    This will be used instead of AI-generated compose configuration.")

                # If we already have a plan, offer to regenerate with custom compose
                if self.current_plan:
                    self._append_output("    [TIP] Click 'Create Plan' again to incorporate your compose file.")

            except Exception as e:
                self._show_message_box("Error Loading File", f"Could not read compose file: {e}", "critical")
                self.custom_compose_file = None
                self.custom_compose_content = None

    def _clear_compose_file(self):
        """Clear the custom compose file"""
        self.custom_compose_file = None
        self.custom_compose_content = None
        self.compose_btn.setText("Compose File...")
        self.compose_btn.setToolTip("Use a custom docker-compose.yml file (right-click to clear)")
        self._append_output("\n[*] Custom compose file cleared. Will use AI-generated configuration.")

    def _show_compose_context_menu(self, pos):
        """Show context menu for compose button"""
        menu = QMenu(self)
        if self.custom_compose_file:
            view_action = menu.addAction("View Compose Content")
            view_action.triggered.connect(self._view_compose_content)
            clear_action = menu.addAction("Clear Compose File")
            clear_action.triggered.connect(self._clear_compose_file)
        else:
            select_action = menu.addAction("Select Compose File...")
            select_action.triggered.connect(self._select_compose_file)
        menu.exec(self.compose_btn.mapToGlobal(pos))

    def _view_compose_content(self):
        """Show the compose file content in a dialog"""
        if not self.custom_compose_content:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Compose File: {Path(self.custom_compose_file).name}")
        dialog.setMinimumSize(600, 400)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit()
        text.setPlainText(self.custom_compose_content)
        text.setReadOnly(True)
        layout.addWidget(text)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()

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
