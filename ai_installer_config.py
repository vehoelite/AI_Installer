#!/usr/bin/env python3
"""
AI Installer - Configuration Management
========================================
Handles persistent storage of user preferences and LLM provider settings.
Settings are stored in ~/.config/ai-installer/config.json
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class LLMProviderSettings:
    """Settings for LLM provider configuration"""
    provider: str = "builtin"  # "builtin", "local", "openai", "anthropic", "gemini"

    # Built-in model uses static path ./models/ (no configuration needed)

    # Local LLM settings
    local_preset: str = "lm-studio"  # "lm-studio", "ollama", "localai", "custom"
    local_host: str = "127.0.0.1"
    local_port: int = 1234
    local_model: str = ""
    local_api_key: str = "not-needed"

    # Cloud API settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Common settings
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: int = 120


@dataclass
class AppSettings:
    """General application settings"""
    theme: str = "dark"  # "dark", "light"
    web_search_enabled: bool = True
    verbose_output: bool = True
    confirm_before_execute: bool = True
    show_command_details: bool = True
    first_run_complete: bool = False

    # Execution settings
    confirm_each_step: bool = False  # Pause before each step for confirmation
    auto_repair_enabled: bool = True  # Automatically attempt to fix failed steps
    max_repair_attempts: int = 3  # Maximum number of auto-repair attempts

    # Window settings
    window_width: int = 1200
    window_height: int = 800
    window_x: int = -1  # -1 means center
    window_y: int = -1


@dataclass
class Config:
    """Main configuration container"""
    llm: LLMProviderSettings = field(default_factory=LLMProviderSettings)
    app: AppSettings = field(default_factory=AppSettings)
    version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "llm": asdict(self.llm),
            "app": asdict(self.app),
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        config = cls()

        if "llm" in data:
            for key, value in data["llm"].items():
                if hasattr(config.llm, key):
                    setattr(config.llm, key, value)

        if "app" in data:
            for key, value in data["app"].items():
                if hasattr(config.app, key):
                    setattr(config.app, key, value)

        if "version" in data:
            config.version = data["version"]

        return config


class ConfigManager:
    """Manages loading and saving configuration"""

    DEFAULT_CONFIG_DIR = Path.home() / ".config" / "ai-installer"
    DEFAULT_CONFIG_FILE = "config.json"

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self.config_file = self.config_dir / self.DEFAULT_CONFIG_FILE
        self._config: Optional[Config] = None

    @property
    def config(self) -> Config:
        """Get current configuration, loading from disk if needed"""
        if self._config is None:
            self._config = self.load()
        return self._config

    def ensure_config_dir(self):
        """Create config directory if it doesn't exist"""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Config:
        """Load configuration from disk, or return defaults"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                return Config.from_dict(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file: {e}")
                return Config()
        return Config()

    def save(self, config: Optional[Config] = None):
        """Save configuration to disk"""
        if config is not None:
            self._config = config

        if self._config is None:
            self._config = Config()

        self.ensure_config_dir()

        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config.to_dict(), f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save config file: {e}")

    def reset(self):
        """Reset to default configuration"""
        self._config = Config()
        self.save()

    def is_first_run(self) -> bool:
        """Check if this is the first run"""
        return not self.config.app.first_run_complete

    def mark_first_run_complete(self):
        """Mark first run as complete"""
        self.config.app.first_run_complete = True
        self.save()

    def get_llm_base_url(self) -> str:
        """Get the base URL for local LLM based on settings"""
        llm = self.config.llm

        if llm.provider != "local":
            return ""

        # Preset URLs
        presets = {
            "lm-studio": f"http://{llm.local_host}:{llm.local_port}/v1",
            "ollama": f"http://{llm.local_host}:11434/v1",
            "localai": f"http://{llm.local_host}:8080/v1",
            "custom": f"http://{llm.local_host}:{llm.local_port}/v1"
        }

        return presets.get(llm.local_preset, presets["custom"])


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> Config:
    """Convenience function to get current config"""
    return get_config_manager().config


def save_config():
    """Convenience function to save current config"""
    get_config_manager().save()


if __name__ == "__main__":
    # Test the configuration system
    print("Testing AI Installer Configuration System")
    print("=" * 50)

    manager = ConfigManager()
    config = manager.config

    print(f"Config directory: {manager.config_dir}")
    print(f"Config file: {manager.config_file}")
    print(f"First run: {manager.is_first_run()}")
    print()
    print("Current settings:")
    print(json.dumps(config.to_dict(), indent=2))

    # Test saving
    config.app.theme = "dark"
    config.llm.provider = "local"
    config.llm.local_preset = "lm-studio"
    manager.save()
    print("\nConfiguration saved!")
