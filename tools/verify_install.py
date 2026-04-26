import importlib
import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parents[1]
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "src"))


def require_import(module_name):
    importlib.import_module(module_name)
    print(f"OK import {module_name}")


def require_distribution(distribution_name):
    version = importlib.metadata.version(distribution_name)
    print(f"OK package {distribution_name}=={version}")


def verify_prompt_parser():
    from helpers.prompt import split_weighted_subprompts

    negative, positive = split_weighted_subprompts(
        "lake:0.7 portrait:`0.2+0.1` mountain:-0.2",
        frame=0,
        skip_normalize=True,
    )
    assert positive == [("lake", 0.7), ("portrait", 0.30000000000000004)]
    assert negative == [("mountain", 0.2)]
    print("OK prompt parser")


def verify_png_writer():
    from helpers.render import save_8_16_or_32bpc_image

    with tempfile.TemporaryDirectory() as tmp:
        image = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 1024)
        save_8_16_or_32bpc_image(image, tmp, "depth.png", 16)
        output_file = Path(tmp) / "depth.png"
        assert output_file.is_file()
        assert output_file.stat().st_size > 0
    print("OK 16-bit PNG writer")


def verify_model_paths():
    from helpers.model_load import get_model_output_paths

    with tempfile.TemporaryDirectory() as tmp:
        root = SimpleNamespace(
            models_path=str(Path(tmp) / "models"),
            output_path=str(Path(tmp) / "outputs"),
            mount_google_drive=False,
            models_path_gdrive="",
            output_path_gdrive="",
        )
        models_path, output_path = get_model_output_paths(root)
        assert Path(models_path).is_dir()
        assert Path(output_path).is_dir()
    print("OK model/output path setup")


def main():
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    for distribution_name in ("ipython", "numpngw", "numexpr", "pydantic"):
        require_distribution(distribution_name)

    for module_name in (
        "clip",
        "helpers.save_images",
        "helpers.settings",
        "helpers.render",
        "helpers.model_load",
        "helpers.aesthetics",
        "helpers.prompts",
        "helpers.animation",
        "helpers.depth",
        "helpers.video",
        "py3d_tools",
    ):
        require_import(module_name)

    verify_prompt_parser()
    verify_png_writer()
    verify_model_paths()
    print("Deforum install verification passed")


if __name__ == "__main__":
    main()
