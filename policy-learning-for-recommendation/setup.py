from setuptools import setup

setup(
    name="rl_pa1",
    version="0.1.0",
    description="Policy gradient & Plackett-Luce ranking policies for recommendation (PG / PL)",
    python_requires=">=3.9",
    py_modules=[
        "base",
        "env",
        "models",
        "pg",
        "pl",
        "visualize",
        "test",
    ],
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23",
        "matplotlib>=3.6",
        "seaborn>=0.12",
    ],
    extras_require={
        "notebook": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rl-pa1-visualize=visualize:main",
        ],
    },
)
