from setuptools import setup, find_packages

# Read the content of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="FineTune_Information_Extractor_for_NLPTasks_based_T5_Small",  # Library name
    version="1.0.3",  # Initial release version
    author="Mehrdad ALMASI, Demival VASQUES FILHO",
    author_email="mehrdad.al.2023@gmail.com, demival.vasques@uni.lu",
    description="A library for fine-tuning T5-small models to perform information extraction for various NLP tasks.",  
    long_description=long_description,  # Load detailed description from README
    long_description_content_type="text/markdown",  # README format
    url="https://github.com/mehrdadalmasi2020/FineTune_Information_Extractor_for_NLPTasks_based_T5_Small",  # GitHub repository
    packages=find_packages(),  # Automatically find and include all packages
    include_package_data=True,  # Include additional non-Python files specified in MANIFEST.in
    install_requires=[  # Required dependencies based on your imports
        "transformers>=4.20.0,<5.0.0",  # T5 model and Tokenizer from Hugging Face
        "torch>=1.7.0,<2.0.0",  # PyTorch for model training and GPU usage
        "pandas>=1.1.0",  # Data handling with Pandas
        "scikit-learn>=1.0",  # For dataset splitting and machine learning utilities
        "numpy>=1.19.0,<1.24.0",  # Numpy for array handling
        "openpyxl>=3.0.0",  # Required for reading and writing Excel files with pandas
        "datasets>=1.15.0,<2.0.0",  # For loading datasets from Hugging Face
    ],
    classifiers=[  # Classifiers help users find your project by defining its audience, environment, and license
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",  # Indicating the library is in the early stages of development
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",  # Ensuring compatibility with Python 3.6 and above
    keywords="NLP, T5, information extraction, transformers, text generation, fine-tuning",  # Keywords for better searchability
    project_urls={  # Additional links that are useful for the users of your library
        "Documentation": "https://github.com/mehrdadalmasi2020/FineTune_Information_Extractor_for_NLPTasks_based_T5_Small",
        "Source": "https://github.com/mehrdadalmasi2020/FineTune_Information_Extractor_for_NLPTasks_based_T5_Small",
        "Tracker": "https://github.com/mehrdadalmasi2020/FineTune_Information_Extractor_for_NLPTasks_based_T5_Small/issues",
    },
)
