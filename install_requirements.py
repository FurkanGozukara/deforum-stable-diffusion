import argparse
import importlib.metadata
import platform
import subprocess
import sys


CORE_PACKAGES = [
    ("clean-fid", "clean-fid==0.1.35"),
    ("einops", "einops"),
    ("ftfy", "ftfy"),
    ("ipython", "ipython"),
    ("jsonmerge", "jsonmerge"),
    ("kornia", "kornia"),
    ("matplotlib", "matplotlib"),
    ("numexpr", "numexpr"),
    ("omegaconf", "omegaconf"),
    ("opencv-python", "opencv-python"),
    ("pandas", "pandas"),
    ("pytorch-lightning", "pytorch-lightning"),
    ("resize-right", "resize-right"),
    ("scikit-image", "scikit-image"),
    ("scikit-learn", "scikit-learn"),
    ("timm", "timm"),
    ("torchdiffeq", "torchdiffeq"),
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("albumentations", "albumentations"),
    ("more-itertools", "more-itertools"),
    ("devtools", "devtools"),
    ("validators", "validators"),
    ("numpngw", "numpngw"),
    ("open-clip-torch", "open-clip-torch"),
    ("torchsde", "torchsde"),
    ("pydantic", "pydantic<2"),
]

JUPYTER_PACKAGES = [
    ("colab-convert", "colab-convert"),
    ("ipywidgets", "ipywidgets"),
    ("jupyterlab", "jupyterlab"),
    ("jupyter_http_over_ws", "jupyter_http_over_ws"),
    ("notebook", "notebook"),
]


def is_installed(distribution_name):
    try:
        importlib.metadata.version(distribution_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def pip_install_packages(packages, verbose=False, pre=False, extra_index_url=None):
    for distribution_name, package_spec in packages:
        if is_installed(distribution_name):
            print(f"..already installed {distribution_name}")
            continue

        print(f"..installing {package_spec}")
        cmd = [sys.executable, "-m", "pip", "install"]
        if pre:
            cmd.append("--pre")
        if not verbose:
            cmd.append("-q")
        cmd.append(package_spec)
        if extra_index_url:
            cmd.extend(["--extra-index-url", extra_index_url])

        if verbose:
            print(" ".join(cmd))

        subprocess.check_call(cmd)


def install_requirements(
    verbose=False,
    include_torch=False,
    include_xformers=False,
    with_jupyter=False,
):
    os_system = platform.system()
    print(f"system detected: {os_system}")
    print(f"python executable: {sys.executable}")

    if include_torch:
        torch_packages = [
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("torchaudio", "torchaudio"),
        ]
        pip_install_packages(torch_packages, verbose=verbose)
    else:
        print("..skipping torch/torchvision/torchaudio; using the active environment")

    pip_install_packages(CORE_PACKAGES, verbose=verbose)

    if with_jupyter:
        pip_install_packages(JUPYTER_PACKAGES, verbose=verbose)
    else:
        print("..skipping Jupyter packages; pass --with-jupyter to install notebook support")

    if include_xformers:
        xformers_packages = [("xformers", "xformers")]
        pip_install_packages(xformers_packages, verbose=verbose)
    else:
        print("..skipping xformers; using the active environment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="print pip output")
    parser.add_argument(
        "--include-torch",
        action="store_true",
        help="install torch packages instead of preserving the active environment",
    )
    parser.add_argument(
        "--include-xformers",
        action="store_true",
        help="install xformers instead of preserving the active environment",
    )
    parser.add_argument(
        "--with-jupyter",
        action="store_true",
        help="install Jupyter notebook/lab packages",
    )
    args = parser.parse_args()
    install_requirements(
        verbose=args.verbose,
        include_torch=args.include_torch,
        include_xformers=args.include_xformers,
        with_jupyter=args.with_jupyter,
    )
