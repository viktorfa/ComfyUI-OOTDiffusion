from setuptools import setup, find_packages

setup(
    name="ComfyUI-OOTDiffusion",
    version="0.0.1",
    description="A description of your project",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/viktorfa/ComfyUI-OOTDiffusion",
    packages=find_packages(include=["ootd", "ootd.*"]),
    package_data={
        "ootd.humanparsing.modules": ["src/**/*"],
    },
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "scipy",
        "scikit-image",
        "opencv-python",
        "pillow",
        "diffusers==0.24.0",
        "transformers",
        "accelerate",
        "matplotlib",
        "tqdm",
        "gradio",
        "config",
        "einops",
        "ninja==1.10.2",
        "basicsr",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
