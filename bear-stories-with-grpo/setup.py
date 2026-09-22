from setuptools import setup


setup(
    name="rl_pa2",
    version="0.1.0",
    description="GRPO post-training of TinyStories-33M with sparse and dense rewards",
    python_requires=">=3.10",
    py_modules=[
        "eval",
        "grpo",
        "rewards",
        "tests",
        "train",
        "utils",
    ],
    install_requires=[
        "torch>=2.1",
        "transformers>=4.40",
        "accelerate>=0.27",
        "huggingface_hub>=0.23",
        "safetensors>=0.4",
        "numpy>=1.24",
        "matplotlib>=3.7",
    ],
    extras_require={
        "semantic": [
            "sentence-transformers>=2.2",
        ],
        "notebook": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
    },
)
