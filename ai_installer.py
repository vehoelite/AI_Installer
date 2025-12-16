#!/usr/bin/env python3
"""
AI-Powered Software Installation Agent
=======================================
An intelligent installation system that uses LLMs to analyze, install,
test, and troubleshoot complex software packages.

Example Use Case: Installing AMD ROCm on WSL2
"""

import subprocess
import platform
import json
import os
import sys
import re
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
from abc import ABC, abstractmethod

# ============================================================================
# Configuration
# ============================================================================

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"  # For LM Studio, Ollama, LocalAI, etc.


@dataclass
class OpenAICompatibleConfig:
    """Configuration for OpenAI-compatible API endpoints (LM Studio, Ollama, LocalAI, etc.)"""
    base_url: str = "http://localhost:1234/v1"  # Default: LM Studio
    api_key: str = "not-needed"  # Most local servers don't require a key
    model_name: str = "local-model"  # Model identifier used by the server

    # Optional endpoint overrides (usually not needed as they follow OpenAI convention)
    chat_endpoint: str = "/chat/completions"
    models_endpoint: str = "/models"

    # Connection settings
    timeout_seconds: int = 120
    max_tokens: int = 4096
    temperature: float = 0.7

    # Provider-specific presets
    # Note: Using 127.0.0.1 instead of "localhost" for better compatibility
    # Some systems (especially WSL) have issues resolving "localhost" properly

    @classmethod
    def lm_studio(cls, model: str = "local-model", host: str = "127.0.0.1", port: int = 1234) -> "OpenAICompatibleConfig":
        """Preset for LM Studio"""
        return cls(
            base_url=f"http://{host}:{port}/v1",
            model_name=model,
            api_key="lm-studio"
        )

    @classmethod
    def ollama(cls, model: str = "llama3.1", host: str = "127.0.0.1", port: int = 11434) -> "OpenAICompatibleConfig":
        """Preset for Ollama"""
        return cls(
            base_url=f"http://{host}:{port}/v1",
            model_name=model,
            api_key="ollama"
        )

    @classmethod
    def localai(cls, model: str = "gpt-4", host: str = "127.0.0.1", port: int = 8080) -> "OpenAICompatibleConfig":
        """Preset for LocalAI"""
        return cls(
            base_url=f"http://{host}:{port}/v1",
            model_name=model,
            api_key="localai"
        )

    @classmethod
    def text_gen_webui(cls, model: str = "default", host: str = "127.0.0.1", port: int = 5000) -> "OpenAICompatibleConfig":
        """Preset for text-generation-webui (oobabooga)"""
        return cls(
            base_url=f"http://{host}:{port}/v1",
            model_name=model,
            api_key="text-gen-webui"
        )

    @classmethod
    def vllm(cls, model: str = "meta-llama/Llama-2-7b-hf", host: str = "127.0.0.1", port: int = 8000) -> "OpenAICompatibleConfig":
        """Preset for vLLM server"""
        return cls(
            base_url=f"http://{host}:{port}/v1",
            model_name=model,
            api_key="vllm"
        )

    @classmethod
    def custom(cls, base_url: str, model: str, api_key: str = "not-needed", **kwargs) -> "OpenAICompatibleConfig":
        """Create a fully custom configuration"""
        return cls(base_url=base_url, model_name=model, api_key=api_key, **kwargs)

    @classmethod
    def from_url(cls, url: str, model: str = "local-model", api_key: str = "not-needed") -> "OpenAICompatibleConfig":
        """Create config from a full URL (auto-detects host and port)"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.path and parsed.path != "/":
            base += parsed.path.rstrip("/")
        return cls(base_url=base, model_name=model, api_key=api_key)


@dataclass
class AgentConfig:
    """Configuration for the AI Installation Agent"""
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    model_name: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    max_retries: int = 3
    dry_run: bool = False  # If True, commands are printed but not executed
    verbose: bool = True

    # OpenAI-compatible API configuration (for local LLMs)
    openai_compatible_config: Optional[OpenAICompatibleConfig] = None

    def __post_init__(self):
        if not self.api_key:
            if self.llm_provider == LLMProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.llm_provider == LLMProvider.ANTHROPIC:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.llm_provider == LLMProvider.OPENAI_COMPATIBLE:
                # Local servers typically don't need a real API key
                self.api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed")


# ============================================================================
# System Analyzer - Gathers comprehensive system information
# ============================================================================

@dataclass
class GPUInfo:
    """Detailed GPU information"""
    vendor: str = ""  # "nvidia", "amd", "intel", "unknown"
    name: str = ""
    driver_version: Optional[str] = None
    vram_total: Optional[str] = None
    vram_used: Optional[str] = None
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    compute_capability: Optional[str] = None
    rocm_version: Optional[str] = None
    pci_bus_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "vendor": self.vendor,
            "name": self.name,
            "driver_version": self.driver_version,
            "vram_total": self.vram_total,
            "vram_used": self.vram_used,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "compute_capability": self.compute_capability,
            "rocm_version": self.rocm_version,
            "pci_bus_id": self.pci_bus_id,
        }.items() if v is not None and v != ""}


@dataclass
class SystemInfo:
    """Comprehensive system information"""
    os_type: str = ""
    os_version: str = ""
    kernel_version: str = ""
    architecture: str = ""
    is_wsl: bool = False
    wsl_version: Optional[str] = None
    distro_name: str = ""
    distro_version: str = ""

    # GPU Information (legacy list for backward compatibility)
    gpu_info: list = field(default_factory=list)
    gpu_driver_version: Optional[str] = None

    # Enhanced GPU Information
    gpus: list = field(default_factory=list)  # List of GPUInfo objects
    nvidia_detected: bool = False
    amd_detected: bool = False
    intel_gpu_detected: bool = False

    # CUDA/ROCm specific
    cuda_version: Optional[str] = None
    cuda_toolkit_path: Optional[str] = None
    cudnn_version: Optional[str] = None
    nvcc_version: Optional[str] = None
    rocm_version: Optional[str] = None

    # Other
    installed_packages: list = field(default_factory=list)
    environment_vars: dict = field(default_factory=dict)
    disk_space: dict = field(default_factory=dict)
    memory_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "os_type": self.os_type,
            "os_version": self.os_version,
            "kernel_version": self.kernel_version,
            "architecture": self.architecture,
            "is_wsl": self.is_wsl,
            "wsl_version": self.wsl_version,
            "distro_name": self.distro_name,
            "distro_version": self.distro_version,
            "gpu_info": self.gpu_info,
            "gpu_driver_version": self.gpu_driver_version,
            "gpus": [g.to_dict() if hasattr(g, 'to_dict') else g for g in self.gpus],
            "nvidia_detected": self.nvidia_detected,
            "amd_detected": self.amd_detected,
            "intel_gpu_detected": self.intel_gpu_detected,
            "cuda_version": self.cuda_version,
            "cuda_toolkit_path": self.cuda_toolkit_path,
            "cudnn_version": self.cudnn_version,
            "nvcc_version": self.nvcc_version,
            "rocm_version": self.rocm_version,
            "installed_packages": self.installed_packages[:50],  # Limit for context
            "disk_space": self.disk_space,
            "memory_info": self.memory_info,
        }


class SystemAnalyzer:
    """Analyzes the target system to gather installation-relevant information"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _run_cmd(self, cmd: str, shell: bool = True) -> tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)"""
        try:
            result = subprocess.run(
                cmd, shell=shell, capture_output=True, text=True, timeout=30
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Analyzer] {msg}")

    def analyze(self) -> SystemInfo:
        """Perform comprehensive system analysis"""
        info = SystemInfo()

        self._log("Detecting OS type...")
        info.os_type = platform.system()
        info.os_version = platform.version()
        info.architecture = platform.machine()

        # Linux-specific detection
        if info.os_type == "Linux":
            self._analyze_linux(info)

        # WSL Detection
        self._detect_wsl(info)

        # GPU Detection
        self._detect_gpu(info)

        # Package Detection
        self._detect_packages(info)

        # System Resources
        self._detect_resources(info)

        return info

    def _analyze_linux(self, info: SystemInfo):
        """Analyze Linux-specific information"""
        self._log("Analyzing Linux distribution...")

        # Kernel version
        code, out, _ = self._run_cmd("uname -r")
        if code == 0:
            info.kernel_version = out

        # Distribution info from /etc/os-release
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("ID="):
                        info.distro_name = line.split("=")[1].strip().strip('"')
                    elif line.startswith("VERSION_ID="):
                        info.distro_version = line.split("=")[1].strip().strip('"')

    def _detect_wsl(self, info: SystemInfo):
        """Detect if running in WSL and which version"""
        self._log("Checking for WSL environment...")

        # Check for WSL indicators
        code, out, _ = self._run_cmd("cat /proc/version 2>/dev/null")
        if code == 0 and "microsoft" in out.lower():
            info.is_wsl = True

            # Determine WSL version
            if "wsl2" in out.lower():
                info.wsl_version = "2"
            else:
                # Check via interop
                code, out, _ = self._run_cmd("cat /proc/sys/fs/binfmt_misc/WSLInterop 2>/dev/null")
                if code == 0:
                    info.wsl_version = "2"  # WSL2 typically has this
                else:
                    info.wsl_version = "1"

    def _detect_gpu(self, info: SystemInfo):
        """Detect GPU information including NVIDIA/CUDA, AMD/ROCm, and Intel"""
        self._log("Detecting GPU hardware...")

        # Try lspci for GPU detection (legacy list for backward compatibility)
        code, out, _ = self._run_cmd("lspci 2>/dev/null | grep -iE 'vga|3d|display'")
        if code == 0 and out:
            info.gpu_info = [line.strip() for line in out.split('\n') if line.strip()]

            # Detect vendor from lspci output
            for line in info.gpu_info:
                line_lower = line.lower()
                if 'nvidia' in line_lower:
                    info.nvidia_detected = True
                if 'amd' in line_lower or 'radeon' in line_lower:
                    info.amd_detected = True
                if 'intel' in line_lower:
                    info.intel_gpu_detected = True

        # =====================================================================
        # NVIDIA / CUDA Detection
        # =====================================================================
        self._detect_nvidia_cuda(info)

        # =====================================================================
        # AMD / ROCm Detection
        # =====================================================================
        self._detect_amd_rocm(info)

    def _detect_nvidia_cuda(self, info: SystemInfo):
        """Detect NVIDIA GPU and CUDA toolkit"""
        self._log("Checking for NVIDIA/CUDA...")

        # Try nvidia-smi for detailed GPU info
        code, out, _ = self._run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,pci.bus_id,compute_cap --format=csv,noheader,nounits 2>/dev/null")
        if code == 0 and out:
            info.nvidia_detected = True
            for line in out.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpu = GPUInfo(
                            vendor="nvidia",
                            name=parts[0] if len(parts) > 0 else "",
                            driver_version=parts[1] if len(parts) > 1 else None,
                            vram_total=f"{parts[2]} MiB" if len(parts) > 2 else None,
                            vram_used=f"{parts[3]} MiB" if len(parts) > 3 else None,
                            pci_bus_id=parts[4] if len(parts) > 4 else None,
                            compute_capability=parts[5] if len(parts) > 5 else None,
                        )
                        info.gpus.append(gpu)
                        # Set driver version from first GPU
                        if not info.gpu_driver_version:
                            info.gpu_driver_version = gpu.driver_version

        # Get CUDA version from nvidia-smi
        code, out, _ = self._run_cmd("nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \\K[0-9.]+'")
        if code == 0 and out:
            info.cuda_version = out.strip()
            info.nvidia_detected = True

        # Get nvcc version (CUDA toolkit compiler)
        code, out, _ = self._run_cmd("nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+'")
        if code == 0 and out:
            info.nvcc_version = out.strip()

        # Find CUDA toolkit path
        cuda_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-12",
            "/usr/local/cuda-11",
            "/opt/cuda",
        ]
        for path in cuda_paths:
            if os.path.exists(path):
                info.cuda_toolkit_path = path
                break

        # Also check CUDA_HOME environment variable
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home and os.path.exists(cuda_home):
            info.cuda_toolkit_path = cuda_home

        # Detect cuDNN version
        self._detect_cudnn(info)

        # If we have CUDA but no GPU info yet from nvidia-smi, try basic detection
        if info.nvidia_detected and not info.gpus:
            code, out, _ = self._run_cmd("nvidia-smi -L 2>/dev/null")
            if code == 0 and out:
                for line in out.strip().split('\n'):
                    if line.strip():
                        # Parse lines like "GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-...)"
                        match = re.search(r'GPU \d+: (.+?)(?:\s*\(UUID|$)', line)
                        if match:
                            gpu = GPUInfo(vendor="nvidia", name=match.group(1).strip())
                            info.gpus.append(gpu)

    def _detect_cudnn(self, info: SystemInfo):
        """Detect cuDNN version"""
        cudnn_paths = []

        # Check common cuDNN header locations
        if info.cuda_toolkit_path:
            cudnn_paths.append(f"{info.cuda_toolkit_path}/include/cudnn_version.h")
            cudnn_paths.append(f"{info.cuda_toolkit_path}/include/cudnn.h")

        cudnn_paths.extend([
            "/usr/include/cudnn_version.h",
            "/usr/include/cudnn.h",
            "/usr/local/include/cudnn_version.h",
            "/usr/include/x86_64-linux-gnu/cudnn_version.h",
        ])

        for path in cudnn_paths:
            if os.path.exists(path):
                code, out, _ = self._run_cmd(f"grep -E 'CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL' {path} 2>/dev/null | head -3")
                if code == 0 and out:
                    major = minor = patch = ""
                    for line in out.split('\n'):
                        if 'CUDNN_MAJOR' in line and 'VERSION' not in line:
                            match = re.search(r'(\d+)', line.split('CUDNN_MAJOR')[-1])
                            if match:
                                major = match.group(1)
                        elif 'CUDNN_MINOR' in line:
                            match = re.search(r'(\d+)', line.split('CUDNN_MINOR')[-1])
                            if match:
                                minor = match.group(1)
                        elif 'CUDNN_PATCHLEVEL' in line:
                            match = re.search(r'(\d+)', line.split('CUDNN_PATCHLEVEL')[-1])
                            if match:
                                patch = match.group(1)
                    if major:
                        info.cudnn_version = f"{major}.{minor}.{patch}".rstrip('.')
                        break

        # Alternative: try to get cuDNN version from ldconfig
        if not info.cudnn_version:
            code, out, _ = self._run_cmd("ldconfig -p 2>/dev/null | grep libcudnn | head -1")
            if code == 0 and out:
                match = re.search(r'libcudnn\.so\.(\d+)', out)
                if match:
                    info.cudnn_version = f"{match.group(1)}.x"

    def _detect_amd_rocm(self, info: SystemInfo):
        """Detect AMD GPU and ROCm"""
        self._log("Checking for AMD/ROCm...")

        # AMD driver version from kernel module
        code, out, _ = self._run_cmd("cat /sys/module/amdgpu/version 2>/dev/null")
        if code == 0 and out:
            info.amd_detected = True
            if not info.gpu_driver_version:
                info.gpu_driver_version = out.strip()

        # ROCm version detection
        rocm_paths = [
            "/opt/rocm/.info/version",
            "/opt/rocm/include/rocm-core/rocm_version.h",
            "/opt/rocm/.info/version-dev",
        ]

        for path in rocm_paths:
            if os.path.exists(path):
                code, out, _ = self._run_cmd(f"cat {path} 2>/dev/null | head -1")
                if code == 0 and out:
                    # Clean up version string
                    version = out.strip()
                    match = re.search(r'[\d.]+', version)
                    if match:
                        info.rocm_version = match.group(0)
                        break

        # Try rocm-smi for GPU info
        code, out, _ = self._run_cmd("rocm-smi --showproductname 2>/dev/null")
        if code == 0 and out:
            info.amd_detected = True
            for line in out.split('\n'):
                if 'GPU' in line or 'Card' in line:
                    # Parse GPU name
                    match = re.search(r'(?:GPU\[\d+\]|Card\s*\d*).*?:\s*(.+)', line)
                    if match:
                        gpu = GPUInfo(vendor="amd", name=match.group(1).strip())
                        info.gpus.append(gpu)

        # Get driver version from rocm-smi
        code, out, _ = self._run_cmd("rocm-smi --showdriverversion 2>/dev/null")
        if code == 0 and out:
            match = re.search(r'[\d.]+', out)
            if match:
                info.gpu_driver_version = match.group(0)

        # Get VRAM info from rocm-smi
        code, out, _ = self._run_cmd("rocm-smi --showmeminfo vram 2>/dev/null")
        if code == 0 and out and info.gpus:
            for i, gpu in enumerate(info.gpus):
                if gpu.vendor == "amd":
                    # Parse VRAM info
                    total_match = re.search(r'Total.*?(\d+)', out)
                    used_match = re.search(r'Used.*?(\d+)', out)
                    if total_match:
                        gpu.vram_total = f"{int(total_match.group(1)) // (1024*1024)} MiB"
                    if used_match:
                        gpu.vram_used = f"{int(used_match.group(1)) // (1024*1024)} MiB"

    def _detect_packages(self, info: SystemInfo):
        """Detect installed packages relevant to the installation"""
        self._log("Scanning installed packages...")

        # Common packages that are frequently needed for installations
        common_packages = [
            "curl", "wget", "git", "gnupg", "gnupg2", "gpg",
            "ca-certificates", "apt-transport-https", "software-properties-common",
            "build-essential", "gcc", "g++", "make", "cmake",
            "python3", "python3-pip", "python3-venv",
            "docker", "docker-ce", "docker.io", "containerd.io",
            "lsb-release", "dirmngr",
        ]

        # Check which common packages are installed
        for pkg in common_packages:
            code, _, _ = self._run_cmd(f"dpkg -s {pkg} 2>/dev/null | grep -q 'Status: install ok installed'")
            if code == 0:
                info.installed_packages.append(pkg)

        # Debian/Ubuntu - GPU related packages
        code, out, _ = self._run_cmd("dpkg -l 2>/dev/null | grep -iE 'rocm|amdgpu|hip|opencl|cuda|nvidia|cudnn|nccl|tensorrt' | awk '{print $2}'")
        if code == 0 and out:
            info.installed_packages.extend([p for p in out.split('\n') if p.strip()])

        # Also check pip packages for ML frameworks
        code, out, _ = self._run_cmd("pip list 2>/dev/null | grep -iE 'torch|tensorflow|jax|cuda|rocm|nvidia'")
        if code == 0 and out:
            info.installed_packages.extend([f"pip:{line.split()[0]}" for line in out.split('\n') if line.strip()])

        # Check conda packages if conda is available
        code, out, _ = self._run_cmd("conda list 2>/dev/null | grep -iE 'cuda|cudnn|pytorch|tensorflow' | awk '{print $1}'")
        if code == 0 and out:
            info.installed_packages.extend([f"conda:{p}" for p in out.split('\n') if p.strip()])

        # Check for tools that might be installed as binaries (not via dpkg)
        # These are often installed via pip, curl scripts, or standalone binaries
        binary_tools = [
            ("docker-compose", ["docker-compose", "docker compose"]),  # Both old and new style
            ("kubectl", ["kubectl"]),
            ("helm", ["helm"]),
            ("minikube", ["minikube"]),
            ("kind", ["kind"]),
            ("terraform", ["terraform"]),
            ("ansible", ["ansible"]),
            ("node", ["node"]),
            ("npm", ["npm"]),
            ("yarn", ["yarn"]),
            ("go", ["go"]),
            ("rustc", ["rustc"]),
            ("cargo", ["cargo"]),
        ]

        for tool_name, check_commands in binary_tools:
            for check_cmd in check_commands:
                code, _, _ = self._run_cmd(f"command -v {check_cmd.split()[0]} >/dev/null 2>&1")
                if code == 0:
                    info.installed_packages.append(tool_name)
                    break

        # Remove duplicates while preserving order
        seen = set()
        unique_packages = []
        for pkg in info.installed_packages:
            if pkg not in seen:
                seen.add(pkg)
                unique_packages.append(pkg)
        info.installed_packages = unique_packages

    def _detect_resources(self, info: SystemInfo):
        """Detect system resources (disk, memory)"""
        self._log("Checking system resources...")

        # Disk space
        code, out, _ = self._run_cmd("df -h / 2>/dev/null | tail -1")
        if code == 0 and out:
            parts = out.split()
            if len(parts) >= 4:
                info.disk_space = {
                    "total": parts[1],
                    "used": parts[2],
                    "available": parts[3],
                    "use_percent": parts[4] if len(parts) > 4 else "unknown"
                }

        # Memory
        code, out, _ = self._run_cmd("free -h 2>/dev/null | grep Mem")
        if code == 0 and out:
            parts = out.split()
            if len(parts) >= 3:
                info.memory_info = {
                    "total": parts[1],
                    "used": parts[2],
                    "available": parts[6] if len(parts) > 6 else "unknown"
                }


# ============================================================================
# LLM Reasoner - The AI brain of the installer
# ============================================================================

class LLMInterface(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    def query(self, system_prompt: str, user_message: str) -> str:
        pass


class AnthropicLLM(LLMInterface):
    """Anthropic Claude integration"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model

    def query(self, system_prompt: str, user_message: str) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            return response.content[0].text
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")


class OpenAILLM(LLMInterface):
    """OpenAI GPT integration"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

    def query(self, system_prompt: str, user_message: str) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")


class OpenAICompatibleLLM(LLMInterface):
    """
    OpenAI-compatible API integration for local LLM providers.

    Supports:
    - LM Studio (default port 1234)
    - Ollama (default port 11434)
    - LocalAI (default port 8080)
    - text-generation-webui (default port 5000)
    - vLLM (default port 8000)
    - Any other OpenAI-compatible server
    """

    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        """Lazy initialization of the OpenAI client"""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout_seconds
                )
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    def test_connection(self) -> tuple[bool, str]:
        """Test the connection to the local LLM server"""
        try:
            client = self._get_client()
            # Try to list models as a connectivity test
            models = client.models.list()
            model_names = [m.id for m in models.data]
            return True, f"Connected! Available models: {model_names}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def list_models(self) -> list[str]:
        """List available models on the server"""
        try:
            client = self._get_client()
            models = client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            return []

    def query(self, system_prompt: str, user_message: str) -> str:
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)

            # Provide helpful error messages for common issues
            if "Connection refused" in error_msg:
                raise RuntimeError(
                    f"Cannot connect to LLM server at {self.config.base_url}. "
                    f"Make sure the server is running.\n"
                    f"For LM Studio: Start the server in Settings > Local Server\n"
                    f"For Ollama: Run 'ollama serve'\n"
                    f"Original error: {error_msg}"
                )
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                available = self.list_models()
                raise RuntimeError(
                    f"Model '{self.config.model_name}' not found on server.\n"
                    f"Available models: {available}\n"
                    f"Original error: {error_msg}"
                )
            else:
                raise RuntimeError(f"LLM query failed: {error_msg}")


class RawHTTPLLM(LLMInterface):
    """
    Raw HTTP implementation for OpenAI-compatible APIs.
    Use this if you don't want to install the openai package.
    """

    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config

    def query(self, system_prompt: str, user_message: str) -> str:
        import urllib.request
        import urllib.error

        url = f"{self.config.base_url.rstrip('/')}{self.config.chat_endpoint}"

        payload = {
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to LLM server at {url}. "
                f"Make sure the server is running. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"LLM query failed: {e}")


class InstallationReasoner:
    """The AI brain that plans and reasons about installations"""

    SYSTEM_PROMPT = """You are an expert system administrator and software installation specialist.
Your role is to analyze system information, plan software installations, execute commands,
and troubleshoot any issues that arise.

IMPORTANT RULES:
1. Always output valid JSON responses as specified
2. Be extremely careful with system commands - prefer safe, reversible operations
3. CRITICAL: When determining prerequisite status, you MUST check the "installed_packages" list in the system information. If a package appears in that list, mark it as "installed". Do NOT guess or assume - use the actual data provided.
4. Provide clear explanations for each step
5. When troubleshooting, analyze error messages carefully and suggest targeted fixes
6. For group membership changes (like adding user to docker group), note that the user will need to log out/in for changes to take effect - do NOT use 'newgrp' as it spawns a new shell and breaks automation.
7. Use 'groupadd --force' or check if group exists before creating to avoid errors on existing groups.
8. NEVER use placeholder names in commands like IMAGE_NAME, YOUR_IMAGE, EXAMPLE_*, <placeholder>, [placeholder], etc. Always use REAL, SPECIFIC package names, Docker images, URLs, and values. If the request is ambiguous, pick the most popular/common option and explain your choice in the description.
9. For vague requests like "web ui", "docker container", etc., pick a specific well-known solution:
   - "web ui for docker" → use portainer/portainer-ce:latest
   - "web server" → use nginx:latest or apache2
   - "database" → use postgres:latest or mariadb:latest
   - "monitoring" → use grafana/grafana:latest or prometheus:latest
   Always pick REAL software, never templates or placeholders.

You have deep knowledge of:
- Linux system administration (Ubuntu, Debian, RHEL, etc.)
- Windows Subsystem for Linux (WSL1 and WSL2)
- GPU computing stacks (AMD ROCm, NVIDIA CUDA)
- Package management (apt, yum, pip, conda)
- System configuration and environment variables
"""

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def analyze_requirements(self,
                            system_info: SystemInfo,
                            software_request: str) -> dict:
        """Analyze what's needed to install the requested software"""

        # Get installed packages as a clear list for the prompt
        installed_pkgs = system_info.installed_packages if hasattr(system_info, 'installed_packages') else []

        prompt = f"""
Analyze the following system and determine what's needed to install the requested software.

SYSTEM INFORMATION:
{json.dumps(system_info.to_dict(), indent=2)}

ALREADY INSTALLED PACKAGES (use this to determine prerequisite status):
{json.dumps(installed_pkgs, indent=2)}

SOFTWARE REQUEST:
{software_request}

IMPORTANT: When filling out "prerequisites", check the ALREADY INSTALLED PACKAGES list above.
- If a package (like "curl", "gnupg", "gnupg2", "git", etc.) appears in that list, set status to "installed"
- Only set status to "missing" if the package is NOT in the installed packages list

Respond with a JSON object containing:
{{
    "compatible": true/false,
    "compatibility_issues": ["list of blocking issues if not compatible"],
    "prerequisites": [
        {{"name": "...", "status": "installed|missing|outdated", "required_version": "...", "current_version": "..."}}
    ],
    "installation_steps": [
        {{"step": 1, "description": "...", "commands": ["cmd1", "cmd2"], "requires_sudo": true/false, "risk_level": "low|medium|high"}}
    ],
    "estimated_time_minutes": 10,
    "disk_space_required_gb": 5,
    "warnings": ["any important warnings"],
    "post_install_tests": [
        {{"name": "test name", "command": "test command", "expected_output_contains": "success string"}}
    ]
}}

Be thorough and precise. Include all necessary steps for a complete installation.
"""

        response = self.llm.query(self.SYSTEM_PROMPT, prompt)
        return self._parse_json_response(response)

    def troubleshoot(self,
                    system_info: SystemInfo,
                    error_output: str,
                    failed_command: str,
                    context: str) -> dict:
        """Analyze an error and suggest fixes"""

        prompt = f"""
An installation step has failed. Analyze the error and suggest fixes.

SYSTEM INFORMATION:
{json.dumps(system_info.to_dict(), indent=2)}

FAILED COMMAND:
{failed_command}

ERROR OUTPUT:
{error_output}

CONTEXT:
{context}

Respond with a JSON object:
{{
    "error_analysis": "detailed analysis of what went wrong",
    "root_cause": "the fundamental cause of the error",
    "fix_steps": [
        {{"step": 1, "description": "...", "commands": ["cmd1"], "requires_sudo": true/false}}
    ],
    "alternative_approaches": ["other ways to achieve the goal if fix doesn't work"],
    "should_retry_original": true/false
}}
"""

        response = self.llm.query(self.SYSTEM_PROMPT, prompt)
        return self._parse_json_response(response)

    def _parse_json_response(self, response: str) -> dict:
        """
        Extract JSON from LLM response with robust error handling.

        Handles common LLM output issues:
        - JSON wrapped in markdown code blocks
        - Unterminated strings
        - Trailing commas
        - Single quotes instead of double quotes
        - Control characters in strings
        """

        def try_parse(text: str) -> Optional[dict]:
            """Attempt to parse JSON with error recovery"""
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None

        def repair_json(text: str) -> str:
            """Attempt to repair common JSON issues"""
            repaired = text

            # Remove control characters except newlines and tabs
            repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', repaired)

            # Fix unterminated strings at end of JSON
            # Count quotes to detect imbalance
            in_string = False
            escaped = False
            last_quote_pos = -1

            for i, char in enumerate(repaired):
                if escaped:
                    escaped = False
                    continue
                if char == '\\':
                    escaped = True
                    continue
                if char == '"':
                    in_string = not in_string
                    last_quote_pos = i

            # If we ended inside a string, try to close it
            if in_string:
                # Find a reasonable place to terminate
                # Look for the last complete-looking structure
                repaired = repaired.rstrip()
                if not repaired.endswith('"'):
                    repaired += '"'
                # Try to close any open structures
                open_braces = repaired.count('{') - repaired.count('}')
                open_brackets = repaired.count('[') - repaired.count(']')
                repaired += ']' * open_brackets + '}' * open_braces

            # Fix trailing commas before } or ]
            repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

            # Fix single quotes used as string delimiters (be careful not to break apostrophes)
            # Only replace single quotes that look like they're delimiting strings
            # This is a heuristic - may not work in all cases
            # repaired = re.sub(r"(?<=[{,:\[])\s*'([^']*?)'\s*(?=[,}\]:])", r'"\1"', repaired)

            return repaired

        def extract_json_block(text: str) -> Optional[str]:
            """Extract JSON from various formats"""
            # Try markdown code block first
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if json_match:
                return json_match.group(1).strip()

            # Try to find a JSON object
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json_match.group(0)

            return None

        # Strategy 1: Try to find and parse JSON code block
        json_block = extract_json_block(response)
        if json_block:
            result = try_parse(json_block)
            if result:
                return result

            # Try repairing the JSON block
            repaired = repair_json(json_block)
            result = try_parse(repaired)
            if result:
                return result

        # Strategy 2: Try parsing the whole response
        result = try_parse(response)
        if result:
            return result

        # Strategy 3: Try repairing the whole response
        repaired = repair_json(response)
        result = try_parse(repaired)
        if result:
            return result

        # Strategy 4: Try to extract just the JSON object from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            extracted = json_match.group(0)
            result = try_parse(extracted)
            if result:
                return result

            repaired = repair_json(extracted)
            result = try_parse(repaired)
            if result:
                return result

        # All strategies failed - return a safe default with error info
        print("  ⚠️  Warning: Could not parse LLM response as JSON, using fallback")
        print(f"  📝 Raw response (first 300 chars): {response[:300]}...")

        # Return a minimal valid response so the caller can continue
        return {
            "error_analysis": "Failed to parse LLM response - the model returned invalid JSON",
            "root_cause": "LLM output parsing error",
            "fix_steps": [],
            "alternative_approaches": ["Try running the failed command manually"],
            "should_retry_original": False,
            "compatible": True,  # Don't block on parse errors
            "prerequisites": [],
            "installation_steps": [],
            "warnings": ["LLM response could not be parsed"],
            "post_install_tests": [],
            "_parse_error": True,
            "_raw_response_preview": response[:500]
        }


# ============================================================================
# Plan Enhancer - Calculates real metrics for installation plans
# ============================================================================

@dataclass
class PackageMetrics:
    """Metrics for a package from apt-cache"""
    name: str
    size_bytes: int = 0
    installed_size_bytes: int = 0
    download_size_bytes: int = 0
    version: str = ""
    is_installed: bool = False


@dataclass
class DockerImageMetrics:
    """Metrics for a Docker image"""
    image: str  # Full image name (e.g., "nginx:latest")
    compressed_size_bytes: int = 0
    tag: str = "latest"
    is_pulled: bool = False


@dataclass
class PlanMetrics:
    """Real metrics calculated for an installation plan"""
    total_steps: int = 0
    total_commands: int = 0
    packages_to_install: list = field(default_factory=list)
    packages_already_installed: list = field(default_factory=list)
    total_download_size_mb: float = 0.0
    total_disk_space_mb: float = 0.0
    estimated_download_time_seconds: float = 0.0
    estimated_install_time_seconds: float = 0.0
    estimated_total_time_minutes: float = 0.0
    # Docker metrics
    docker_images_to_pull: list = field(default_factory=list)
    docker_images_already_pulled: list = field(default_factory=list)
    docker_download_size_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "total_commands": self.total_commands,
            "packages_to_install": self.packages_to_install,
            "packages_already_installed": self.packages_already_installed,
            "total_download_size_mb": round(self.total_download_size_mb, 2),
            "total_disk_space_mb": round(self.total_disk_space_mb, 2),
            "estimated_download_time_seconds": round(self.estimated_download_time_seconds, 1),
            "estimated_install_time_seconds": round(self.estimated_install_time_seconds, 1),
            "estimated_total_time_minutes": round(self.estimated_total_time_minutes, 1),
            "docker_images_to_pull": self.docker_images_to_pull,
            "docker_images_already_pulled": self.docker_images_already_pulled,
            "docker_download_size_mb": round(self.docker_download_size_mb, 2),
        }


class PlanEnhancer:
    """
    Enhances LLM-generated plans with real metrics calculated from:
    - Actual package sizes from apt-cache
    - Command counts from the plan
    - Realistic time estimates based on download and install speeds
    """

    # Estimated speeds for time calculations
    DOWNLOAD_SPEED_MBPS = 10  # Conservative estimate (10 Mbps)
    INSTALL_SPEED_MB_PER_SEC = 50  # ~50 MB/s for package extraction/config
    BASE_COMMAND_TIME_SEC = 2  # Base time per command (overhead)
    APT_UPDATE_TIME_SEC = 10  # Time for apt update
    APT_INSTALL_OVERHEAD_SEC = 5  # Overhead per apt install command

    # Docker image pull time estimate (seconds per MB at 10 Mbps)
    DOCKER_PULL_TIME_PER_MB = 0.8

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._package_cache: dict[str, PackageMetrics] = {}
        self._docker_cache: dict[str, DockerImageMetrics] = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [PlanEnhancer] {msg}")

    def _run_cmd(self, cmd: str) -> tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)"""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except Exception as e:
            return -1, "", str(e)

    def get_package_info(self, package_name: str) -> Optional[PackageMetrics]:
        """Get package information from apt-cache"""
        # Check cache first
        if package_name in self._package_cache:
            return self._package_cache[package_name]

        # Query apt-cache - get enough lines to capture all metadata fields
        code, out, _ = self._run_cmd(f"apt-cache show {package_name} 2>/dev/null | head -50")
        if code != 0 or not out:
            if self.verbose:
                self._log(f"  Could not get apt-cache info for {package_name}")
            return None

        metrics = PackageMetrics(name=package_name)

        for line in out.split('\n'):
            if line.startswith('Size:'):
                try:
                    metrics.download_size_bytes = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith('Installed-Size:'):
                try:
                    # Installed-Size is in KB
                    metrics.installed_size_bytes = int(line.split(':')[1].strip()) * 1024
                except (ValueError, IndexError):
                    pass
            elif line.startswith('Version:'):
                metrics.version = line.split(':', 1)[1].strip()

        # Log what we found
        if self.verbose and (metrics.download_size_bytes > 0 or metrics.installed_size_bytes > 0):
            dl_mb = metrics.download_size_bytes / (1024 * 1024)
            inst_mb = metrics.installed_size_bytes / (1024 * 1024)
            self._log(f"  {package_name}: {dl_mb:.1f} MB download, {inst_mb:.1f} MB installed")

        # Check if already installed
        code, _, _ = self._run_cmd(f"dpkg -s {package_name} 2>/dev/null | grep -q 'Status: install ok installed'")
        metrics.is_installed = (code == 0)

        self._package_cache[package_name] = metrics
        return metrics

    def extract_packages_from_commands(self, commands: list[str]) -> list[str]:
        """Extract package names from apt/apt-get install commands"""
        packages = []

        for cmd in commands:
            # Debug: log what commands we're analyzing
            if self.verbose:
                self._log(f"Analyzing command: {cmd[:80]}{'...' if len(cmd) > 80 else ''}")

            # Match apt-get install or apt install commands
            # Patterns handled:
            #   apt-get install -y pkg1 pkg2 pkg3
            #   apt install -y pkg1 pkg2 pkg3
            #   sudo apt-get install -y pkg1 pkg2 pkg3
            #   sudo apt install -y pkg1 pkg2 pkg3
            #   DEBIAN_FRONTEND=noninteractive apt-get install -y pkg1 pkg2
            #   sudo DEBIAN_FRONTEND=noninteractive apt-get install -y pkg1 pkg2

            # Remove common prefixes that appear before apt commands
            cleaned_cmd = cmd
            # Remove sudo prefix
            cleaned_cmd = re.sub(r'^sudo\s+', '', cleaned_cmd)
            # Remove environment variable assignments
            cleaned_cmd = re.sub(r'^\s*\w+=[^\s]+\s+', '', cleaned_cmd)
            # Remove sudo again if it was after env vars
            cleaned_cmd = re.sub(r'^sudo\s+', '', cleaned_cmd)

            # Now try to match apt/apt-get install
            match = re.search(r'(?:apt-get|apt)\s+install\s+(.+?)(?:\s*[;&|]|$)', cleaned_cmd)
            if match:
                # Extract package names (filter out options like -y, --no-install-recommends, etc.)
                parts = match.group(1).split()
                for part in parts:
                    # Skip options (start with -)
                    if part.startswith('-'):
                        continue
                    # Skip redirections
                    if part.startswith('>') or part.startswith('<'):
                        continue
                    # Skip if contains = (version pinning handled separately)
                    if '=' in part:
                        # Extract package name before the version specifier
                        pkg = part.split('=')[0]
                    else:
                        pkg = part

                    # Clean and validate package name
                    pkg = pkg.strip()
                    if pkg and pkg not in packages and not pkg.startswith('$'):
                        packages.append(pkg)
                        if self.verbose:
                            self._log(f"  Found package: {pkg}")

        return packages

    def extract_docker_images_from_commands(self, commands: list[str]) -> list[str]:
        """Extract Docker image names from docker pull/run/compose commands"""
        images = []

        for cmd in commands:
            # Match docker pull commands
            # Pattern: docker pull nginx:latest
            # Pattern: sudo docker pull nginx
            pull_match = re.search(r'docker\s+pull\s+([^\s;&|]+)', cmd)
            if pull_match:
                image = pull_match.group(1).strip()
                if image and image not in images:
                    images.append(image)
                    if self.verbose:
                        self._log(f"  Found docker image (pull): {image}")

            # Match docker-compose/docker compose up commands with image references
            # These often pull images like portainer/portainer-ce:latest
            compose_match = re.search(r'docker[\s-]compose\s+', cmd)
            if compose_match and self.verbose:
                self._log(f"  Found docker-compose command (images parsed from YAML at runtime)")

            # Match docker run commands
            # Pattern: docker run -d nginx:latest
            # Pattern: docker run -d --name foo nginx
            # Pattern: docker run -d -p 8080:80 --name webserver nginx:alpine
            run_match = re.search(r'docker\s+run\s+(.+)', cmd)
            if run_match:
                # Parse the run command to find the image (last non-option argument before any command)
                run_args = run_match.group(1)
                # Split and find image - it's typically the first arg that doesn't start with - and isn't a value for an option
                parts = run_args.split()
                skip_next = False
                for i, part in enumerate(parts):
                    if skip_next:
                        skip_next = False
                        continue
                    # Skip options and their values
                    if part.startswith('-'):
                        # Options that take a value
                        if part in ['-p', '--publish', '-v', '--volume', '-e', '--env',
                                   '--name', '-w', '--workdir', '--network', '-u', '--user',
                                   '--entrypoint', '-m', '--memory', '--cpus']:
                            skip_next = True
                        elif '=' not in part and part.startswith('--'):
                            # Long options without = might take next arg
                            pass
                        continue
                    # Skip if looks like a command (after image)
                    if '/' not in part and ':' not in part and not re.match(r'^[a-z0-9_.-]+$', part, re.I):
                        break
                    # This is likely the image
                    if part and part not in images:
                        images.append(part)
                        if self.verbose:
                            self._log(f"  Found docker image (run): {part}")
                    break

        return images

    def get_docker_image_size(self, image: str) -> Optional[DockerImageMetrics]:
        """Get Docker image size from Docker Hub API or local docker"""
        # Check cache first
        if image in self._docker_cache:
            return self._docker_cache[image]

        # Parse image name and tag
        if ':' in image:
            name, tag = image.rsplit(':', 1)
        else:
            name, tag = image, 'latest'

        metrics = DockerImageMetrics(image=image, tag=tag)

        # First try to check if already pulled locally
        code, out, _ = self._run_cmd(f"docker images --format '{{{{.Size}}}}' {image} 2>/dev/null | head -1")
        if code == 0 and out:
            metrics.is_pulled = True
            # Parse size like "540MB" or "1.2GB"
            size_match = re.match(r'([\d.]+)\s*(KB|MB|GB|TB)', out, re.I)
            if size_match:
                size_val = float(size_match.group(1))
                unit = size_match.group(2).upper()
                multipliers = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
                metrics.compressed_size_bytes = int(size_val * multipliers.get(unit, 1))

        # If not pulled locally, try Docker Hub API
        if not metrics.is_pulled or metrics.compressed_size_bytes == 0:
            try:
                import urllib.request
                import urllib.error

                # Handle official images (no namespace)
                if '/' not in name:
                    namespace = 'library'
                    repo = name
                else:
                    namespace, repo = name.split('/', 1)

                url = f"https://hub.docker.com/v2/repositories/{namespace}/{repo}/tags/{tag}"
                req = urllib.request.Request(url, headers={'Accept': 'application/json'})

                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    if 'full_size' in data:
                        metrics.compressed_size_bytes = data['full_size']
                        if self.verbose:
                            size_mb = metrics.compressed_size_bytes / (1024 * 1024)
                            self._log(f"  Docker Hub size for {image}: {size_mb:.1f} MB")
            except Exception as e:
                if self.verbose:
                    self._log(f"  Could not get Docker Hub size for {image}: {e}")

        self._docker_cache[image] = metrics
        return metrics

    def calculate_metrics(self, plan: dict) -> PlanMetrics:
        """Calculate real metrics for an installation plan"""
        self._log("Calculating real installation metrics...")

        metrics = PlanMetrics()

        # Count steps and commands
        steps = plan.get("installation_steps", [])
        metrics.total_steps = len(steps)

        all_commands = []
        for step in steps:
            cmds = step.get("commands", [])
            if isinstance(cmds, str):
                cmds = [cmds]
            all_commands.extend(cmds)

        metrics.total_commands = len(all_commands)

        # Extract packages from commands
        all_packages = self.extract_packages_from_commands(all_commands)
        self._log(f"Found {len(all_packages)} packages in commands")

        # Get info for each package
        total_download = 0
        total_disk = 0

        for pkg_name in all_packages:
            pkg_info = self.get_package_info(pkg_name)
            if pkg_info:
                if pkg_info.is_installed:
                    metrics.packages_already_installed.append(pkg_name)
                else:
                    metrics.packages_to_install.append(pkg_name)
                    total_download += pkg_info.download_size_bytes
                    total_disk += pkg_info.installed_size_bytes

        # Convert to MB
        metrics.total_download_size_mb = total_download / (1024 * 1024)
        metrics.total_disk_space_mb = total_disk / (1024 * 1024)

        # Calculate time estimates
        # Download time: size / speed
        download_time = (metrics.total_download_size_mb * 8) / self.DOWNLOAD_SPEED_MBPS
        metrics.estimated_download_time_seconds = download_time

        # Install time: size / install speed + command overhead
        install_time = metrics.total_disk_space_mb / self.INSTALL_SPEED_MB_PER_SEC
        metrics.estimated_install_time_seconds = install_time

        # Add overhead for each command
        command_overhead = 0
        for cmd in all_commands:
            command_overhead += self.BASE_COMMAND_TIME_SEC
            if 'apt-get update' in cmd or 'apt update' in cmd:
                command_overhead += self.APT_UPDATE_TIME_SEC
            if 'apt-get install' in cmd or 'apt install' in cmd:
                command_overhead += self.APT_INSTALL_OVERHEAD_SEC

        # Extract and analyze Docker images
        docker_images = self.extract_docker_images_from_commands(all_commands)
        if docker_images:
            self._log(f"Found {len(docker_images)} Docker images in commands")

        docker_download_total = 0
        for image in docker_images:
            img_metrics = self.get_docker_image_size(image)
            if img_metrics:
                if img_metrics.is_pulled:
                    metrics.docker_images_already_pulled.append(image)
                else:
                    metrics.docker_images_to_pull.append(image)
                    docker_download_total += img_metrics.compressed_size_bytes

        metrics.docker_download_size_mb = docker_download_total / (1024 * 1024)

        # Calculate Docker pull time and add to download time estimate
        docker_pull_time = metrics.docker_download_size_mb * self.DOCKER_PULL_TIME_PER_MB

        # Update download time to include Docker images
        # Docker download: size in MB * 8 bits / speed in Mbps = seconds
        docker_download_time = (metrics.docker_download_size_mb * 8) / self.DOWNLOAD_SPEED_MBPS
        metrics.estimated_download_time_seconds = download_time + docker_download_time

        # Total time in minutes
        total_seconds = metrics.estimated_download_time_seconds + install_time + command_overhead
        metrics.estimated_total_time_minutes = total_seconds / 60

        # Add minimum time (things always take longer than expected)
        if metrics.estimated_total_time_minutes < 1:
            metrics.estimated_total_time_minutes = 1

        self._log(f"Download: {metrics.total_download_size_mb:.1f} MB (apt) + {metrics.docker_download_size_mb:.1f} MB (docker), "
                  f"Disk: {metrics.total_disk_space_mb:.1f} MB, "
                  f"Time: ~{metrics.estimated_total_time_minutes:.1f} min")

        return metrics

    def enhance_plan(self, plan: dict) -> dict:
        """Enhance a plan with real metrics, replacing LLM estimates"""
        metrics = self.calculate_metrics(plan)

        # Update plan with real values
        enhanced = plan.copy()
        enhanced["_real_metrics"] = metrics.to_dict()

        # Replace LLM estimates with real calculations
        enhanced["estimated_time_minutes"] = max(1, round(metrics.estimated_total_time_minutes))
        enhanced["disk_space_required_gb"] = round(metrics.total_disk_space_mb / 1024, 2)

        # Add detailed breakdown
        enhanced["_metrics_breakdown"] = {
            "download_size_mb": round(metrics.total_download_size_mb, 2),
            "disk_space_mb": round(metrics.total_disk_space_mb, 2),
            "packages_to_install": len(metrics.packages_to_install),
            "packages_already_installed": len(metrics.packages_already_installed),
            "docker_images_to_pull": len(metrics.docker_images_to_pull),
            "docker_images_already_pulled": len(metrics.docker_images_already_pulled),
            "docker_download_size_mb": round(metrics.docker_download_size_mb, 2),
            "total_commands": metrics.total_commands,
        }

        return enhanced, metrics


# ============================================================================
# Command Executor - Safely executes installation commands
# ============================================================================

@dataclass
class CommandResult:
    """Result of executing a command"""
    command: str
    return_code: int
    stdout: str
    stderr: str
    success: bool
    duration_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class DangerousCommandInfo:
    """Information about why a command is dangerous and its alternative"""
    pattern: str           # Regex pattern to match
    reason: str            # Why it's dangerous
    alternative: str       # What to do instead
    action: str            # "skip", "warn", or "transform"
    transform_to: str = "" # Replacement command (if action is "transform")


class CommandSafetyFilter:
    """
    Filters out commands that would break subprocess execution or are dangerous.

    Some commands spawn new shells, require TTY, or have other behaviors
    incompatible with non-interactive subprocess execution.
    """

    # Commands that spawn new shells or require TTY interaction
    DANGEROUS_COMMANDS: list[DangerousCommandInfo] = [
        DangerousCommandInfo(
            pattern=r"(?:sudo\s+)?newgrp\s+(\w+)",
            reason="newgrp spawns a new shell session, which steals stdin and stops the Python process",
            alternative="Group membership will take effect after logout/login or by running: exec su -l $USER",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"(?:sudo\s+)?su\s+-",
            reason="su spawns an interactive login shell",
            alternative="Use 'sudo command' for individual commands instead",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"(?:sudo\s+)?su\s+\w+",
            reason="su spawns an interactive shell as another user",
            alternative="Use 'sudo -u username command' for individual commands instead",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"(?:sudo\s+)?login\b",
            reason="login requires TTY and spawns interactive session",
            alternative="This command cannot be automated",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"(?:sudo\s+)?passwd\b",
            reason="passwd requires interactive TTY input",
            alternative="Use 'echo \"user:password\" | chpasswd' for non-interactive password changes",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"(?:sudo\s+)?visudo\b",
            reason="visudo requires interactive editor",
            alternative="Use 'echo \"content\" | sudo tee -a /etc/sudoers.d/filename' instead",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"(?:sudo\s+)?vipw\b",
            reason="vipw requires interactive editor",
            alternative="Use usermod or direct file editing with proper locking",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"\bssh\s+(?!-o\s+BatchMode)",
            reason="ssh may require interactive authentication",
            alternative="Use 'ssh -o BatchMode=yes' or ensure key-based auth is configured",
            action="warn"
        ),
        DangerousCommandInfo(
            pattern=r"\bsudo\s+-i\b",
            reason="sudo -i spawns an interactive login shell",
            alternative="Use 'sudo command' for individual commands instead",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"\bsudo\s+-s\b",
            reason="sudo -s spawns an interactive shell",
            alternative="Use 'sudo command' for individual commands instead",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"\bexec\s+(?:bash|sh|zsh)\b",
            reason="exec replaces the current shell, breaking the script",
            alternative="Skip this command - it's meant for manual shell sessions",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"\b(?:nano|vim?|emacs|pico|edit)\s+",
            reason="Interactive editors require TTY",
            alternative="Use 'echo content | tee file' or 'cat << EOF > file' for file editing",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"\bmore\b|\bless\b",
            reason="Pagers require interactive TTY",
            alternative="Use 'cat' instead for non-interactive output",
            action="transform",
            transform_to="cat"
        ),
        DangerousCommandInfo(
            pattern=r"\btop\b|\bhtop\b",
            reason="System monitors require interactive TTY",
            alternative="Use 'ps aux' or 'free -h' for non-interactive system info",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"\breboot\b|\bshutdown\b|\bpoweroff\b|\bhalt\b",
            reason="System control commands would terminate the installation",
            alternative="Run these commands manually after installation completes",
            action="skip"
        ),
        DangerousCommandInfo(
            pattern=r"\brm\s+-rf\s+/\s*$|\brm\s+-rf\s+/\*",
            reason="This would delete the entire filesystem!",
            alternative="NEVER run this command",
            action="skip"
        ),
    ]

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        # Compile patterns for efficiency
        self._compiled_patterns = [
            (re.compile(cmd.pattern, re.IGNORECASE), cmd)
            for cmd in self.DANGEROUS_COMMANDS
        ]

    def _log(self, msg: str, level: str = "INFO"):
        if self.verbose:
            prefix = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌", "SKIP": "⏭️", "TRANSFORM": "🔄"}
            print(f"  {prefix.get(level, 'ℹ️')} {msg}")

    def check_command(self, command: str) -> tuple[bool, Optional[DangerousCommandInfo], str]:
        """
        Check if a command is safe to execute.

        Returns:
            Tuple of (is_safe, danger_info, transformed_command)
            - is_safe: True if command can be executed
            - danger_info: DangerousCommandInfo if matched, None otherwise
            - transformed_command: Modified command if transformed, original otherwise
        """
        for pattern, info in self._compiled_patterns:
            if pattern.search(command):
                if info.action == "transform":
                    # Transform the command to a safe alternative
                    transformed = pattern.sub(info.transform_to, command)
                    self._log(f"Transforming '{command}' -> '{transformed}'", "TRANSFORM")
                    return True, info, transformed
                elif info.action == "warn":
                    # Warn but allow execution
                    self._log(f"Warning: {info.reason}", "WARN")
                    return True, info, command
                else:  # skip
                    return False, info, command

        return True, None, command

    def filter_commands(self, commands: list[str]) -> list[tuple[str, bool, str]]:
        """
        Filter a list of commands.

        Returns:
            List of (command, should_execute, reason) tuples
        """
        results = []
        for cmd in commands:
            is_safe, info, transformed = self.check_command(cmd)
            if is_safe:
                results.append((transformed, True, ""))
            else:
                reason = f"{info.reason}. {info.alternative}" if info else "Unknown danger"
                results.append((cmd, False, reason))
        return results


class CommandExecutor:
    """Safely executes commands with logging and rollback support"""

    def __init__(self, dry_run: bool = False, verbose: bool = True):
        self.dry_run = dry_run
        self.verbose = verbose
        self.command_history: list[CommandResult] = []
        self.safety_filter = CommandSafetyFilter(verbose=verbose)
        self.pending_user_actions: list[str] = []  # Actions user needs to do after install

    def _log(self, msg: str, level: str = "INFO"):
        if self.verbose:
            prefix = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌", "SUCCESS": "✅", "CMD": "🔧", "SKIP": "⏭️"}
            print(f"  {prefix.get(level, 'ℹ️')} {msg}")

    def execute(self, command: str, requires_sudo: bool = False,
                timeout: int = 300, env: Optional[dict] = None) -> CommandResult:
        """Execute a command and return the result"""
        import time

        if requires_sudo and not command.startswith("sudo"):
            command = f"sudo {command}"

        # Safety check - filter dangerous commands
        is_safe, danger_info, transformed_command = self.safety_filter.check_command(command)

        if not is_safe:
            self._log(f"SKIPPING dangerous command: {command}", "SKIP")
            self._log(f"Reason: {danger_info.reason}", "WARN")
            self._log(f"Alternative: {danger_info.alternative}", "INFO")

            # Add to pending user actions
            self.pending_user_actions.append(
                f"Manual action required: {danger_info.alternative}"
            )

            result = CommandResult(
                command=command,
                return_code=0,  # Not a failure, just skipped
                stdout="",
                stderr="",
                success=True,  # Consider skipped commands as "success" to continue
                skipped=True,
                skip_reason=f"{danger_info.reason}. {danger_info.alternative}"
            )
            self.command_history.append(result)
            return result

        # Use transformed command if it was modified
        command = transformed_command

        self._log(f"Executing: {command}", "CMD")

        if self.dry_run:
            self._log("(Dry run - command not executed)", "WARN")
            result = CommandResult(
                command=command,
                return_code=0,
                stdout="[DRY RUN]",
                stderr="",
                success=True
            )
            self.command_history.append(result)
            return result

        start_time = time.time()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, **(env or {})}
            )
            duration = time.time() - start_time

            result = CommandResult(
                command=command,
                return_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                success=proc.returncode == 0,
                duration_seconds=duration
            )

            if result.success:
                self._log(f"Command completed in {duration:.1f}s", "SUCCESS")
            else:
                self._log(f"Command failed with code {proc.returncode}", "ERROR")
                if proc.stderr:
                    self._log(f"Error: {proc.stderr[:200]}", "ERROR")

            self.command_history.append(result)
            return result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self._log(f"Command timed out after {timeout}s", "ERROR")
            result = CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                success=False,
                duration_seconds=duration
            )
            self.command_history.append(result)
            return result

        except Exception as e:
            duration = time.time() - start_time
            self._log(f"Command error: {e}", "ERROR")
            result = CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=str(e),
                success=False,
                duration_seconds=duration
            )
            self.command_history.append(result)
            return result


# ============================================================================
# Test Runner - Validates installations
# ============================================================================

class TestRunner:
    """Runs validation tests after installation"""

    def __init__(self, executor: CommandExecutor, verbose: bool = True):
        self.executor = executor
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Test] {msg}")

    def run_tests(self, tests: list[dict]) -> dict:
        """Run a list of tests and return results"""
        results = {
            "total": len(tests),
            "passed": 0,
            "failed": 0,
            "test_results": []
        }

        for test in tests:
            self._log(f"Running: {test['name']}")
            result = self.executor.execute(test["command"], timeout=60)

            # Check if output contains expected string
            expected = test.get("expected_output_contains", "")
            output = result.stdout + result.stderr
            passed = result.success and (not expected or expected in output)

            test_result = {
                "name": test["name"],
                "command": test["command"],
                "passed": passed,
                "output": output[:500],  # Truncate for readability
                "expected": expected
            }
            results["test_results"].append(test_result)

            if passed:
                results["passed"] += 1
                self._log(f"✅ PASSED: {test['name']}")
            else:
                results["failed"] += 1
                self._log(f"❌ FAILED: {test['name']}")

        return results


# ============================================================================
# Main AI Installation Agent
# ============================================================================

class AIInstallationAgent:
    """Main agent that orchestrates the entire installation process"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.analyzer = SystemAnalyzer(verbose=config.verbose)
        self.executor = CommandExecutor(dry_run=config.dry_run, verbose=config.verbose)
        self.test_runner = TestRunner(self.executor, verbose=config.verbose)
        self.plan_enhancer = PlanEnhancer(verbose=config.verbose)

        # Initialize LLM
        if config.llm_provider == LLMProvider.ANTHROPIC:
            llm = AnthropicLLM(config.api_key, config.model_name)
        elif config.llm_provider == LLMProvider.OPENAI:
            llm = OpenAILLM(config.api_key, config.model_name)
        elif config.llm_provider == LLMProvider.OPENAI_COMPATIBLE:
            if config.openai_compatible_config is None:
                # Default to LM Studio config
                config.openai_compatible_config = OpenAICompatibleConfig.lm_studio()
            llm = OpenAICompatibleLLM(config.openai_compatible_config)

            # Test connection on startup
            if config.verbose:
                print(f"🔌 Connecting to local LLM at {config.openai_compatible_config.base_url}")
                success, msg = llm.test_connection()
                if success:
                    print(f"   ✅ {msg}")
                else:
                    print(f"   ⚠️  {msg}")
        else:
            raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")

        self.reasoner = InstallationReasoner(llm)
        self.system_info: Optional[SystemInfo] = None

    def _print_header(self, msg: str):
        print(f"\n{'='*60}")
        print(f"  {msg}")
        print(f"{'='*60}\n")

    def _print_section(self, msg: str):
        print(f"\n--- {msg} ---\n")

    def install(self, software_request: str,
                confirm_steps: bool = True,
                on_step_complete: Optional[Callable] = None) -> dict:
        """
        Main installation method

        Args:
            software_request: Description of what to install (e.g., "AMD ROCm 6.0 for WSL2")
            confirm_steps: If True, asks for confirmation before each step
            on_step_complete: Optional callback after each step

        Returns:
            Installation result dictionary
        """

        self._print_header("🤖 AI Installation Agent Started")
        print(f"Request: {software_request}\n")

        # Phase 1: System Analysis
        self._print_section("Phase 1: Analyzing System")
        self.system_info = self.analyzer.analyze()
        print(f"OS: {self.system_info.distro_name} {self.system_info.distro_version}")
        print(f"Kernel: {self.system_info.kernel_version}")
        print(f"WSL: {'Yes (v' + self.system_info.wsl_version + ')' if self.system_info.is_wsl else 'No'}")

        # Enhanced GPU display
        if self.system_info.gpus:
            print(f"GPU(s) detected: {len(self.system_info.gpus)}")
            for i, gpu in enumerate(self.system_info.gpus):
                gpu_dict = gpu.to_dict() if hasattr(gpu, 'to_dict') else gpu
                vendor = gpu_dict.get('vendor', 'unknown').upper()
                name = gpu_dict.get('name', 'Unknown')
                vram = gpu_dict.get('vram_total', '')
                print(f"  [{i}] {vendor}: {name}" + (f" ({vram})" if vram else ""))
        elif self.system_info.gpu_info:
            print(f"GPU: {', '.join(self.system_info.gpu_info)}")
        else:
            print("GPU: None detected")

        # CUDA/ROCm info
        if self.system_info.nvidia_detected:
            cuda_info = []
            if self.system_info.cuda_version:
                cuda_info.append(f"CUDA {self.system_info.cuda_version}")
            if self.system_info.cudnn_version:
                cuda_info.append(f"cuDNN {self.system_info.cudnn_version}")
            if self.system_info.gpu_driver_version:
                cuda_info.append(f"Driver {self.system_info.gpu_driver_version}")
            if cuda_info:
                print(f"NVIDIA Stack: {', '.join(cuda_info)}")

        if self.system_info.amd_detected:
            rocm_info = []
            if self.system_info.rocm_version:
                rocm_info.append(f"ROCm {self.system_info.rocm_version}")
            if self.system_info.gpu_driver_version and not self.system_info.nvidia_detected:
                rocm_info.append(f"Driver {self.system_info.gpu_driver_version}")
            if rocm_info:
                print(f"AMD Stack: {', '.join(rocm_info)}")

        # Phase 2: AI Analysis & Planning
        self._print_section("Phase 2: AI Planning Installation")
        print("Consulting AI for installation plan...")

        try:
            plan = self.reasoner.analyze_requirements(self.system_info, software_request)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate installation plan: {e}",
                "phase": "planning"
            }

        # Check compatibility
        if not plan.get("compatible", True):
            print("\n❌ System is not compatible:")
            for issue in plan.get("compatibility_issues", []):
                print(f"  • {issue}")
            return {
                "success": False,
                "error": "System not compatible",
                "compatibility_issues": plan.get("compatibility_issues", []),
                "phase": "compatibility_check"
            }

        # Enhance plan with real metrics
        print("Calculating real installation metrics...")
        plan, real_metrics = self.plan_enhancer.enhance_plan(plan)

        # Show plan summary with real metrics
        print(f"\n📋 Installation Plan:")
        print(f"   Steps: {real_metrics.total_steps}")
        print(f"   Total commands: {real_metrics.total_commands}")

        # Show package breakdown
        if real_metrics.packages_to_install:
            print(f"\n📦 Packages to install: {len(real_metrics.packages_to_install)}")
            # Show first 10 packages, truncate if more
            display_pkgs = real_metrics.packages_to_install[:10]
            for pkg in display_pkgs:
                print(f"      • {pkg}")
            if len(real_metrics.packages_to_install) > 10:
                print(f"      ... and {len(real_metrics.packages_to_install) - 10} more")

        if real_metrics.packages_already_installed:
            print(f"   ✓ Already installed: {len(real_metrics.packages_already_installed)} packages")

        # Show Docker images breakdown
        if real_metrics.docker_images_to_pull:
            print(f"\n🐳 Docker images to pull: {len(real_metrics.docker_images_to_pull)}")
            for img in real_metrics.docker_images_to_pull[:10]:
                print(f"      • {img}")
            if len(real_metrics.docker_images_to_pull) > 10:
                print(f"      ... and {len(real_metrics.docker_images_to_pull) - 10} more")

        if real_metrics.docker_images_already_pulled:
            print(f"   ✓ Already pulled: {len(real_metrics.docker_images_already_pulled)} images")

        # Show real size and time estimates
        has_apt = real_metrics.packages_to_install or real_metrics.total_download_size_mb > 0
        has_docker = real_metrics.docker_images_to_pull or real_metrics.docker_download_size_mb > 0

        if has_apt or has_docker:
            print(f"\n📊 Resource Requirements:")
            if has_apt:
                print(f"   APT packages:")
                print(f"      Download size: {real_metrics.total_download_size_mb:.1f} MB")
                print(f"      Disk space needed: {real_metrics.total_disk_space_mb:.1f} MB ({real_metrics.total_disk_space_mb/1024:.2f} GB)")
            if has_docker:
                print(f"   Docker images:")
                print(f"      Download size: {real_metrics.docker_download_size_mb:.1f} MB (compressed)")
            if has_apt and has_docker:
                total_download = real_metrics.total_download_size_mb + real_metrics.docker_download_size_mb
                print(f"   Total download: {total_download:.1f} MB")
        else:
            # No apt packages or Docker images detected
            print(f"\n📊 Resource Requirements:")
            print(f"   ⚠️  No apt packages or Docker images detected in commands")
            print(f"   This installation may use other methods (snap, pip, binary downloads, etc.)")
            print(f"   Size estimates not available")

        print(f"\n⏱️  Time Estimates:")
        print(f"   Download time: ~{real_metrics.estimated_download_time_seconds:.0f} seconds (at 10 Mbps)")
        print(f"   Install time: ~{real_metrics.estimated_install_time_seconds:.0f} seconds")
        print(f"   Total estimated: ~{real_metrics.estimated_total_time_minutes:.1f} minutes")

        if plan.get("warnings"):
            print("\n⚠️  Warnings:")
            for warning in plan["warnings"]:
                print(f"   • {warning}")

        # Show prerequisites
        if plan.get("prerequisites"):
            print("\n📦 Prerequisites:")
            for prereq in plan["prerequisites"]:
                status_icon = "✅" if prereq["status"] == "installed" else "❌"
                print(f"   {status_icon} {prereq['name']}: {prereq['status']}")

        # Confirm to proceed
        if confirm_steps:
            response = input("\n🔹 Proceed with installation? (yes/no): ").lower().strip()
            if response not in ["yes", "y"]:
                return {"success": False, "error": "Installation cancelled by user", "phase": "confirmation"}

        # Phase 3: Execute Installation
        self._print_section("Phase 3: Executing Installation")

        installation_results = []
        for step in plan.get("installation_steps", []):
            step_num = step["step"]
            description = step["description"]

            print(f"\n📌 Step {step_num}: {description}")

            if confirm_steps:
                print(f"   Commands: {step['commands']}")
                response = input("   Execute this step? (yes/no/skip): ").lower().strip()
                if response == "skip":
                    installation_results.append({"step": step_num, "skipped": True})
                    continue
                elif response not in ["yes", "y"]:
                    return {"success": False, "error": "Installation cancelled", "phase": f"step_{step_num}"}

            # Execute commands in this step
            step_success = True
            for cmd in step["commands"]:
                result = self.executor.execute(cmd, requires_sudo=step.get("requires_sudo", False))

                if not result.success:
                    step_success = False
                    print(f"\n❌ Step {step_num} failed!")

                    # Attempt troubleshooting
                    if self.config.max_retries > 0:
                        print("\n🔧 Attempting automated troubleshooting...")
                        fix = self.reasoner.troubleshoot(
                            self.system_info,
                            error_output=result.stderr,
                            failed_command=cmd,
                            context=description
                        )

                        print(f"\n📊 Analysis: {fix.get('error_analysis', 'Unknown error')}")
                        print(f"🔍 Root cause: {fix.get('root_cause', 'Unknown')}")

                        # Try fix steps
                        if fix.get("fix_steps") and confirm_steps:
                            response = input("\n   Attempt automatic fix? (yes/no): ").lower().strip()
                            if response in ["yes", "y"]:
                                for fix_step in fix["fix_steps"]:
                                    # Safely get commands - handle missing or empty commands list
                                    fix_commands = fix_step.get("commands", [])

                                    # Handle case where commands might be a string instead of list
                                    if isinstance(fix_commands, str):
                                        fix_commands = [fix_commands]

                                    # Skip if no commands
                                    if not fix_commands:
                                        print(f"  ⚠️  Skipping fix step (no commands): {fix_step.get('description', 'Unknown')}")
                                        continue

                                    # Execute all commands in this fix step
                                    fix_success = True
                                    for fix_cmd in fix_commands:
                                        if not fix_cmd or not fix_cmd.strip():
                                            continue
                                        fix_result = self.executor.execute(
                                            fix_cmd,
                                            requires_sudo=fix_step.get("requires_sudo", False)
                                        )
                                        if not fix_result.success:
                                            fix_success = False
                                            break

                                    if fix_success and fix.get("should_retry_original"):
                                        # Retry original command
                                        result = self.executor.execute(cmd, requires_sudo=step.get("requires_sudo", False))
                                        if result.success:
                                            step_success = True
                                            print("✅ Fix successful!")
                                            break

                    if not step_success:
                        installation_results.append({
                            "step": step_num,
                            "success": False,
                            "error": result.stderr
                        })

                        response = input("\n   Continue despite failure? (yes/no): ").lower().strip()
                        if response not in ["yes", "y"]:
                            return {
                                "success": False,
                                "error": f"Failed at step {step_num}",
                                "step_results": installation_results,
                                "phase": "execution"
                            }
                        break

            if step_success:
                installation_results.append({"step": step_num, "success": True})
                if on_step_complete:
                    on_step_complete(step_num, description)

        # Phase 4: Validation
        self._print_section("Phase 4: Validating Installation")

        tests = plan.get("post_install_tests", [])
        if tests:
            test_results = self.test_runner.run_tests(tests)
            print(f"\n📊 Test Results: {test_results['passed']}/{test_results['total']} passed")

            if test_results["failed"] > 0:
                print("\n⚠️  Some tests failed. Installation may need attention.")
                for test in test_results["test_results"]:
                    if not test["passed"]:
                        print(f"   ❌ {test['name']}")
        else:
            test_results = {"total": 0, "passed": 0, "failed": 0, "test_results": []}
            print("No validation tests defined.")

        # Phase 5: Show pending user actions (for skipped commands)
        if self.executor.pending_user_actions:
            self._print_section("Phase 5: Required Manual Actions")
            print("⚠️  Some commands were skipped for safety. Please complete these manually:")
            print("")
            for i, action in enumerate(self.executor.pending_user_actions, 1):
                print(f"   {i}. {action}")
            print("")
            print("💡 TIP: For Docker group membership, you typically need to:")
            print("   - Log out and log back in, OR")
            print("   - Run: newgrp docker (in your terminal), OR")
            print("   - Restart your WSL instance: wsl --shutdown")

        # Final summary
        self._print_header("🏁 Installation Complete")
        all_passed = test_results["failed"] == 0

        return {
            "success": all_passed,
            "step_results": installation_results,
            "test_results": test_results,
            "system_info": self.system_info.to_dict(),
            "pending_user_actions": self.executor.pending_user_actions,
            "real_metrics": real_metrics.to_dict(),
            "phase": "complete"
        }


# ============================================================================
# Example Usage & CLI
# ============================================================================

def example_local_llm():
    """Example: Use a local LLM (LM Studio, Ollama, etc.)"""

    print("\n" + "="*60)
    print("  🏠 Local LLM Configuration Examples")
    print("="*60 + "\n")

    # =========================================================================
    # Example 1: LM Studio (most common)
    # =========================================================================
    print("1️⃣  LM Studio Configuration:")
    print("-" * 40)

    # Using preset (simplest)
    lm_studio_config = OpenAICompatibleConfig.lm_studio(
        model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",  # Model loaded in LM Studio
        port=1234  # Default LM Studio port
    )
    print(f"   Base URL: {lm_studio_config.base_url}")
    print(f"   Model: {lm_studio_config.model_name}")

    # =========================================================================
    # Example 2: Ollama
    # =========================================================================
    print("\n2️⃣  Ollama Configuration:")
    print("-" * 40)

    ollama_config = OpenAICompatibleConfig.ollama(
        model="llama3.1",  # or "codellama", "mistral", etc.
        host="localhost",
        port=11434
    )
    print(f"   Base URL: {ollama_config.base_url}")
    print(f"   Model: {ollama_config.model_name}")

    # =========================================================================
    # Example 3: Custom server (any OpenAI-compatible API)
    # =========================================================================
    print("\n3️⃣  Custom Server Configuration:")
    print("-" * 40)

    custom_config = OpenAICompatibleConfig.custom(
        base_url="http://192.168.1.100:8080/v1",  # Your server
        model="my-fine-tuned-model",
        api_key="my-secret-key",  # If required
        max_tokens=8192,
        temperature=0.5,
        timeout_seconds=180
    )
    print(f"   Base URL: {custom_config.base_url}")
    print(f"   Model: {custom_config.model_name}")

    # =========================================================================
    # Example 4: Using the agent with local LLM
    # =========================================================================
    print("\n4️⃣  Full Agent Example (with LM Studio):")
    print("-" * 40)

    config = AgentConfig(
        llm_provider=LLMProvider.OPENAI_COMPATIBLE,
        openai_compatible_config=OpenAICompatibleConfig.lm_studio(
            model="local-model",
            port=1234
        ),
        dry_run=True,
        verbose=True,
        max_retries=2
    )

    print(f"   Provider: {config.llm_provider.value}")
    print(f"   Base URL: {config.openai_compatible_config.base_url}")
    print(f"   Dry Run: {config.dry_run}")

    print("\n✅ Configuration complete! To run an actual installation:")
    print("""
    agent = AIInstallationAgent(config)
    result = agent.install("Install Docker and docker-compose")
    """)

    return {"success": True, "message": "Examples displayed"}


def example_rocm_wsl2():
    """Example: Install AMD ROCm on WSL2"""

    # Configure the agent
    config = AgentConfig(
        llm_provider=LLMProvider.ANTHROPIC,  # or LLMProvider.OPENAI
        model_name="claude-sonnet-4-20250514",  # or "gpt-4o"
        dry_run=True,  # Set to False for actual installation
        verbose=True,
        max_retries=2
    )

    # Create agent
    agent = AIInstallationAgent(config)

    # Define what to install
    software_request = """
    Install AMD ROCm 6.1 for machine learning on WSL2 with the following requirements:
    - Target OS: Ubuntu 22.04 or 24.04 in WSL2
    - Use case: WSL + ROCm for PyTorch
    - Installation flags: --usecase=wsl,rocm --no-dkms
    - Need to verify Windows host has compatible AMD GPU driver
    - Install PyTorch with ROCm support
    - Verify installation can detect AMD GPU
    """

    # Run installation
    result = agent.install(software_request, confirm_steps=True)

    print("\n" + "="*60)
    print("Final Result:", "SUCCESS ✅" if result["success"] else "FAILED ❌")
    print("="*60)

    return result


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI-Powered Software Installation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cloud providers
  %(prog)s "Install AMD ROCm 6.1 for WSL2"
  %(prog)s "Install CUDA toolkit 12.0" --provider openai

  # Local LLM providers (OpenAI-compatible)
  %(prog)s "Install Docker" --provider local --local-preset lm-studio
  %(prog)s "Install Docker" --provider local --local-preset ollama --model llama3.1
  %(prog)s "Install Docker" --provider local --base-url http://192.168.1.100:8080/v1 --model my-model

  # Dry run (test without executing)
  %(prog)s "Install Docker and docker-compose" --dry-run

Supported local providers (presets):
  lm-studio      - LM Studio (default: localhost:1234)
  ollama         - Ollama (default: localhost:11434)
  localai        - LocalAI (default: localhost:8080)
  text-gen-webui - text-generation-webui (default: localhost:5000)
  vllm           - vLLM server (default: localhost:8000)
        """
    )

    parser.add_argument("software", nargs="?", help="Software to install (description)")

    # Provider selection
    parser.add_argument("--provider", "-p",
                       choices=["anthropic", "openai", "local"],
                       default="anthropic",
                       help="LLM provider to use (default: anthropic)")

    # Model configuration
    parser.add_argument("--model", "-m",
                       help="Model name to use")

    # Local LLM configuration
    local_group = parser.add_argument_group("Local LLM Options")
    local_group.add_argument("--local-preset",
                            choices=["lm-studio", "ollama", "localai", "text-gen-webui", "vllm"],
                            default="lm-studio",
                            help="Local LLM provider preset (default: lm-studio)")
    local_group.add_argument("--base-url",
                            help="Custom base URL for local LLM (e.g., http://localhost:1234/v1)")
    local_group.add_argument("--host",
                            default="127.0.0.1",
                            help="Host for local LLM server (default: 127.0.0.1)")
    local_group.add_argument("--port", "-P",
                            type=int,
                            help="Port for local LLM server (overrides preset default)")
    local_group.add_argument("--api-key",
                            help="API key for local server (usually not needed)")
    local_group.add_argument("--temperature",
                            type=float,
                            default=0.7,
                            help="Temperature for generation (default: 0.7)")
    local_group.add_argument("--max-tokens",
                            type=int,
                            default=4096,
                            help="Max tokens for generation (default: 4096)")
    local_group.add_argument("--timeout",
                            type=int,
                            default=120,
                            help="Request timeout in seconds (default: 120)")

    # Execution options
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't execute commands, just show what would happen")
    parser.add_argument("--no-confirm", action="store_true",
                       help="Don't ask for confirmations before each step")
    parser.add_argument("--example", action="store_true",
                       help="Run the ROCm installation example")
    parser.add_argument("--example-local", action="store_true",
                       help="Show local LLM configuration examples")
    parser.add_argument("--test-connection", action="store_true",
                       help="Test connection to LLM server and exit")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models on local server and exit")

    args = parser.parse_args()

    # Handle example modes
    if args.example:
        return example_rocm_wsl2()

    if args.example_local:
        return example_local_llm()

    # Build configuration based on provider
    if args.provider == "local":
        # Build OpenAI-compatible config
        if args.base_url:
            # Custom URL provided
            local_config = OpenAICompatibleConfig.from_url(
                url=args.base_url,
                model=args.model or "local-model",
                api_key=args.api_key or "not-needed"
            )
        else:
            # Use preset with optional overrides
            preset_map = {
                "lm-studio": OpenAICompatibleConfig.lm_studio,
                "ollama": OpenAICompatibleConfig.ollama,
                "localai": OpenAICompatibleConfig.localai,
                "text-gen-webui": OpenAICompatibleConfig.text_gen_webui,
                "vllm": OpenAICompatibleConfig.vllm,
            }

            preset_func = preset_map[args.local_preset]

            # Build kwargs for preset
            preset_kwargs = {}
            if args.model:
                preset_kwargs["model"] = args.model
            if args.port:
                preset_kwargs["port"] = args.port
            if args.host and args.host != "127.0.0.1":
                preset_kwargs["host"] = args.host

            local_config = preset_func(**preset_kwargs)

        # Apply additional settings
        local_config.temperature = args.temperature
        local_config.max_tokens = args.max_tokens
        local_config.timeout_seconds = args.timeout
        if args.api_key:
            local_config.api_key = args.api_key

        config = AgentConfig(
            llm_provider=LLMProvider.OPENAI_COMPATIBLE,
            openai_compatible_config=local_config,
            dry_run=args.dry_run,
            verbose=True
        )

        # Handle test-connection mode
        if args.test_connection:
            print(f"🔌 Testing connection to {local_config.base_url}...")
            llm = OpenAICompatibleLLM(local_config)
            success, msg = llm.test_connection()
            if success:
                print(f"✅ {msg}")
                return {"success": True}
            else:
                print(f"❌ {msg}")
                return {"success": False, "error": msg}

        # Handle list-models mode
        if args.list_models:
            print(f"📋 Fetching models from {local_config.base_url}...")
            llm = OpenAICompatibleLLM(local_config)
            models = llm.list_models()
            if models:
                print("Available models:")
                for model in models:
                    print(f"  • {model}")
                return {"success": True, "models": models}
            else:
                print("❌ Could not fetch models (server may be offline or incompatible)")
                return {"success": False}

    elif args.provider == "openai":
        config = AgentConfig(
            llm_provider=LLMProvider.OPENAI,
            model_name=args.model or "gpt-4o",
            dry_run=args.dry_run,
            verbose=True
        )
    else:  # anthropic
        config = AgentConfig(
            llm_provider=LLMProvider.ANTHROPIC,
            model_name=args.model or "claude-sonnet-4-20250514",
            dry_run=args.dry_run,
            verbose=True
        )

    # Require software argument for actual installation
    if not args.software:
        parser.print_help()
        print("\n❌ Error: Please specify software to install or use --example")
        return

    agent = AIInstallationAgent(config)
    result = agent.install(args.software, confirm_steps=not args.no_confirm)

    print("\n" + json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
