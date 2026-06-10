from __future__ import annotations

import sys
from pathlib import Path


TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}
FILTERED_REQUIREMENTS = Path("/tmp/comfyui-requirements-no-torch.txt")


def package_name(requirement: str) -> str:
    stripped = requirement.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return ""

    candidate = stripped.split(";", 1)[0].strip()
    for separator in ("[", "==", ">=", "<=", "~=", "!=", ">", "<", "="):
        candidate = candidate.split(separator, 1)[0].strip()
    return candidate.lower()


def filter_requirements() -> None:
    lines = []
    for line in Path("requirements.txt").read_text().splitlines():
        if package_name(line) in TORCH_PACKAGES:
            continue
        lines.append(line)

    FILTERED_REQUIREMENTS.write_text("\n".join(lines) + "\n")


def patch_audio_vae() -> None:
    path = Path("comfy/ldm/lightricks/vae/audio_vae.py")
    text = path.read_text()
    original = "import torch\nimport torchaudio\n"
    patched = (
        "import torch\n"
        "try:\n"
        "    import torchaudio\n"
        "except (ImportError, OSError):\n"
        "    torchaudio = None\n"
    )

    if patched in text:
        return
    if original not in text:
        raise SystemExit(f"expected torchaudio import block not found in {path}")

    path.write_text(text.replace(original, patched, 1))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_comfyui.py requirements|source")

    command = sys.argv[1]
    if command == "requirements":
        filter_requirements()
        return
    if command == "source":
        patch_audio_vae()
        return

    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
