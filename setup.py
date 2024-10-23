from setuptools import setup, find_packages

# Read the content of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="FineTune_Information_Extractor_for_NLPTasks_based_mBART",  # library name
    version="1.0.5",  # Initial release version for the mBART-based version
    author="Mehrdad ALMASI, Demival VASQUES FILHO",
    author_email="mehrdad.al.2023@gmail.com, demival.vasques@uni.lu",
    description="A library for fine-tuning mBART models to perform information extraction for various NLP tasks.",  
    long_description=long_description,  # Load detailed description from README
    long_description_content_type="text/markdown",  # README format
    url="https://github.com/mehrdadalmasi2020/FineTune-Information-Extractor-mBART",  # GitHub repository URL
    packages=find_packages(),  # Automatically find and include all packages
    include_package_data=True,  # Include additional non-Python files specified in MANIFEST.in
    install_requires=[  # Required dependencies based on your imports
        "transformers>=4.20.0,<5.0.0",  # Hugging Face Transformers, for mBART model and Tokenizer
        "torch>=1.7.0,<2.0.0",  # PyTorch for model training and GPU usage
        "pandas>=1.1.0",  # Data handling with Pandas
        "scikit-learn>=1.0",  # For dataset splitting and machine learning utilities
        "numpy>=1.19.0,<1.24.0",  # Numpy for array handling
        "openpyxl>=3.0.0",  # Required for reading and writing Excel files with pandas
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
    keywords="NLP, mBART, information extraction, transformers, multilingual, fine-tuning",  # Keywords for better searchability
    project_urls={  # Additional links that are useful for the users of your library
        "Documentation": "https://github.com/mehrdadalmasi2020/FineTune-Information-Extractor-mBART",
        "Source": "https://github.com/mehrdadalmasi2020/FineTune-Information-Extractor-mBART",
        "Tracker": "https://github.com/mehrdadalmasi2020/FineTune-Information-Extractor-mBART/issues",
    },
)
