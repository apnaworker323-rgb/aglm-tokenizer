from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aglm-tokenizer",
    version="1.0.0",
    author="AGLM Team",
    description="Universal Multilingual & Romanized Indic Tokenizer with 1.55M+ Full-Capacity Vocabulary",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apnaworker323-rgb/aglm-tokenizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=[
        "regex==2026.5.9",
        "tiktoken>=0.5.0",
        "transformers>=4.38.0",
        "openpyxl>=3.1.0",
        "pandas>=2.0.0",
        "flask>=2.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ]
    }
)
