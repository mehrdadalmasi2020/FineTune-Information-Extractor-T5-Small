# Fine-Tune Information Extractor for NLP Tasks based on mBART


Fine-Tune Information Extractor for NLP Tasks based on **mBART** is a powerful library designed for fine-tuning the pre-trained `mBART` model on custom information extraction tasks. The library provides an intuitive interface for loading datasets, fine-tuning the mBART model, and exporting results efficiently in multilingual contexts.

To fine-tune the **mBART model** included in this library, please ensure that your system has a **GPU with at least 20GB of memory** (depending on the input training text, memory usage can grow up to 40GB).
This requirement is necessary for training large models like `facebook/mbart-large-cc25` on moderate to large datasets.


## Table of Contents
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Fine-tuning the Model](#fine-tuning-the-model)
- [Model Loading and Inference](#model-loading-and-inference)
- [Results](#results)
- [Authors](#authors)
- [License](#license)
- [Example Usage](#example-usage)


## Key Features
- **mBART Fine-tuning**: Fine-tune the `mBART` model for custom multilingual information extraction tasks.
- **Customizable Task Instructions**: Supports flexible task instructions (e.g., "Extract Authors", "Extract Keywords") across multiple languages.
- **Text Preprocessing**: Combines and tokenizes multiple text columns for input into the model, optimized for multilingual data.
- **GPU Support**: Utilizes GPU acceleration for faster training, with performance gains for large multilingual datasets.


## Quick Start
The primary interface for interacting with this library is the `InfoExtractionModel` class, which allows you to load data, fine-tune the mBART model, and generate output for a given input text. The model supports multilingual information extraction tasks and can be fine-tuned with minimal configuration.

## Fine-tuning the Model
To fine-tune the `mBART` model, you need to prepare three datasets: one for training, one for evaluation, and optionally one for testing. These files should be provided in Excel or CSV format. 
The process involves selecting the correct text column and target information column for information extraction. Since mBART is a multilingual model, you can specify the language of your dataset (e.g., French, German) and ensure that the corresponding language codes are used during the fine-tuning process.

### 1. Prepare Train and Evaluation Files
You must provide three separate files for training, and evaluation. Each file should include columns containing the text from which you want to extract information (e.g., abstracts, articles) and the specific target information (e.g., authors, dates, keywords) to extract.

- **Train File**: Contains the data to train the model.
- **Evaluation File**: Contains the data for validating the model during training.

Since mBART is a multilingual model, ensure that the text in each file corresponds to the appropriate language and that you specify the language codes (e.g., `fr_XX` for French, `de_DE` for German) during fine-tuning. This will allow mBART to effectively handle and extract information in the target language.


### 2. Load and Show Available Columns
The first step is to load the dataset and display the available columns, allowing the user to select which columns contain the text and target information for extraction. You can specify the path to each file (training, and evaluation), and the code will display the available columns so you can choose the one that contains the text (e.g., combined_text) and the one that contains the target information (e.g., authors, keywords).

You should ensure that the text column corresponds to the appropriate language of the dataset and adjust the language codes accordingly when fine-tuning the model.


```python
import pandas as pd

# Load the dataset (Excel or CSV)
train_file_path = input("Please provide the path for the training dataset file: ")
eval_file_path = input("Please provide the path for the evaluation dataset file: ")

# Load the training data to display columns
train_data = pd.read_excel(train_file_path) if train_file_path.endswith('.xlsx') else pd.read_csv(train_file_path)

# Show available columns
print("\n--- Available Columns in Training Dataset ---")
print(train_data.columns)

# Ask the user to specify the text and target columns
text_column = input("\nEnter the name of the column containing the text data (e.g., 'combined_text'): ").strip()
target_information_column = input("Enter the name of the column containing the target information (e.g., 'Authors'): ").strip()

# Extract text and target information from training and evaluation datasets
texts_train = train_data[text_column].tolist()
targets_train = train_data[target_information_column].tolist()

# Load the evaluation data
eval_data = pd.read_excel(eval_file_path) if eval_file_path.endswith('.xlsx') else pd.read_csv(eval_file_path)

# Show available columns in the evaluation dataset
print("\n--- Available Columns in Evaluation Dataset ---")
print(eval_data.columns)

texts_eval = eval_data[text_column].tolist()
targets_eval = eval_data[target_information_column].tolist()


```
### 3. Define the Task Instruction:
Next, the user must define the task instruction. This defines the type of information that needs to be extracted from the text. Below are some example task scenarios:

#### Example Task Scenarios:
- Extract Authors
- Extract Publication Dates
- Extract Keywords
- Extract Abstract

The task instruction should be chosen based on the user's specific needs. Here's how the user can input their task:

```python
# Define the task instruction
print("\n--- Example Task Scenarios ---")
print("1. Extract Authors")
print("2. Extract Publication Dates")
print("3. Extract Keywords")
print("4. Extract Abstract")
print("Choose the task you want to perform.")

task_instruction = input("Enter your task (e.g., 'Extract authors', 'Extract keywords'): ").strip()
print(f"You chose: {task_instruction}")

```
### 4. Train and Fine-Tune the Model:
Once the dataset is ready, and the task instruction is set, you can train the model by specifying the task and number of epochs.

```python
# Fine-tune the model
num_epochs = int(input("\nHow many epochs would you like to train for? (e.g., 3, 5, 10): "))
print(f"Training for {num_epochs} epoch(s).")

# Initialize the extraction model
from FineTune_Information_Extractor_for_NLPTasks_based_mBART import InfoExtractionModel
extractor = InfoExtractionModel()

 Ask the user for the model save path
save_model_path = input("\nPlease enter the path where you want to save the trained model (default: './info_extraction_model'): ").strip() or './info_extraction_model'
print(f"Model will be saved at: {save_model_path}")

# Train the model
extractor.train(texts_train, targets_train, texts_eval, targets_eval, task_instruction, num_epochs, save_model_path)
extractor.save_fine_tuned_model(save_model_path)

```
## Load the Trained Model:
After training the model, you can load the trained model for inference or evaluation using the following code:


### 1. Load a Trained Model:
```python
# Load the trained model
extractor.load_model('./info_extraction_model')
print("Model loaded from './info_extraction_model'")

```

### 2. Extract Information from New Text:
You can now use the loaded model to extract information from new text inputs. 
The `extract` method accepts the text and task instruction (which was defined earlier, such as "Extract Authors").

```python
# Extract information from new text
new_text='Complete resolution of cutaneous larva migrans with topical ivermectin: A case report Francesca  Magri, Camilla  Chello, Giulia  Pranteda, Guglielmo  Pranteda Cutaneous larva migrans (CLM; also called creeping eruption) is a cutaneous ectoparasitosis commonly observed in tropical countries. It is characterized by an erythematous, pruritic, and raised lesion with linear or serpiginous distribution, typically localized at the lower extremities. Oral ivermectin represents the most recommended current treatment, with important adverse effects associated. We report the clinical case of a 52‐year old with CLM, successfully treated with topical ivermectin.'
a=extractor.extract(new_text,task_instruction)

```
In this example, `new_text` contains the input text you want the model to process based on the task instruction (e.g., "Extract Authors").


## Clearing Memory:
After completing the evaluation, it's important to clear up memory (especially if you're working with large datasets and models on GPUs). The script uses garbage collection (`gc.collect()`) and CUDA memory clearing (`torch.cuda.empty_cache()`) to free up any allocated memory:
```python
# Clear memory after the process
del texts_train, targets_train, texts_eval, targets_eval, train_data, eval_data
torch.cuda.empty_cache()
gc.collect()
```
### Example output :

```plaintext
Extracting information based on task: Complete resolution of cutaneous larva migrans with topical ivermectin: A case report Francesca  Magri, Camilla  Chello, Giulia  Pranteda, Guglielmo  Pranteda Cutaneous larva migrans (CLM; also called creeping eruption) is a cutaneous ectoparasitosis commonly observed in tropical countries. It is characterized by an erythematous, pruritic, and raised lesion with linear or serpiginous distribution, typically localized at the lower extremities. Oral ivermectin represents the most recommended current treatment, with important adverse effects associated. We report the clinical case of a 52‐year old with CLM, successfully treated with topical ivermectin.
Extracted information: Francesca Magri, Camilla Chello, Giulia Pranteda, Guglielmo Pranteda
```

### Authors

- Mehrdad ALMASI (email: mehrdad.al.2023@gmail.com)
- Demival VASQUES FILHO (email: demival.vasques@uni.lu)

### License

This project is licensed under the MIT License - see the LICENSE file for details.


## Example Usage

This section provides a complete example of how to load a dataset, split it into training and validation sets, fine-tune the `mBART` model, and evaluate it for information extraction.

### 1. Build and Split the Dataset

If you do not have a dataset, we will build one together using the `leminda-ai/s2orc_small` dataset from Hugging Face. If you already have a dataset, you can skip to the section where you input the training dataset path.

The example below demonstrates how to load the dataset, process it, split it into training and validation sets, and convert it into a Pandas DataFrame.

Make sure the selected columns for training and validation do not contain null values.

!pip install --upgrade datasets

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset


# Ask the user if they have a dataset or want to build one
build_dataset = input("Do you want to build a new dataset? (yes/no): ").strip().lower()

if build_dataset == 'yes':
    # Step 1: Load the dataset
    dataset = load_dataset("leminda-ai/s2orc_small",split="train")

    # Convert to a pandas dataframe
    # Reduce the size for demonstration purposes

    df = pd.DataFrame(dataset[:2000])

    # Step 2: Extract author names from the 'authors' column
    def extract_author_names(authors_list):
      if authors_list and isinstance(authors_list, list):
        return ', '.join([author.get("name", "") for author in authors_list])
      return "Unknown"

    df['author_names'] = df['authors'].apply(extract_author_names)

    # Step 3: Create a new column combining 'title', 'author_names', and 'abstract' (paperAbstract)
    df['combined_text'] = df['title'] + " " + df['author_names'] + " " + df['paperAbstract']
    df['Authors'] = df['author_names']

    # Step 4: Save the updated dataset to CSV
    df.to_excel('updated_s2orc_small_with_authors.xlsx', index=False)
    print(df.head)

    print("Dataset updated and saved as 'updated_s2orc_small_with_authors.xlsx'.")


      # Step 6: Split the dataset into training and validation sets
    data=df
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Step 7: Save the datasets
    train_file_path = input("Please enter the path where the training dataset should be saved (default: ./train_data.xlsx): ").strip() or './train_data.xlsx'
    val_file_path = input("Please enter the path where the validation dataset should be saved (default: ./validation_data.xlsx): ").strip() or './validation_data.xlsx'

    train_data.to_excel(train_file_path, index=False)
    val_data.to_excel(val_file_path, index=False)

    print(f"Training dataset saved to: {train_file_path}")
    print(f"Validation dataset saved to: {val_file_path}")

else:
    # Step 1: Proceed with user inputs for dataset paths if they already have the datasets
    train_file_path = input("Please enter the path to the training file (CSV or Excel): ").strip()
    val_file_path = input("Please enter the path to the validation file (CSV or Excel): ").strip()

# Step 2: Load the datasets based on their file types
if train_file_path.endswith('.csv'):
    train_data = pd.read_csv(train_file_path)
elif train_file_path.endswith('.xlsx'):
    train_data = pd.read_excel(train_file_path)
else:
    raise ValueError("Unsupported file format. Please provide a CSV or Excel file for the training data.")

if val_file_path.endswith('.csv'):
    val_data = pd.read_csv(val_file_path)
elif val_file_path.endswith('.xlsx'):
    val_data = pd.read_excel(val_file_path)
else:
    raise ValueError("Unsupported file format for validation dataset. Please provide a CSV or Excel file.")
```
### 2. Fine-tune and Evaluate the Model

Once you have your dataset loaded and split, you can fine-tune the `mBART` model using the following script.

```python
from FineTune_Information_Extractor_for_NLPTasks_based_mBART import InfoExtractionModel


# Step 3: Create the model instance
model = InfoExtractionModel()

# Step 4: Load the dataset
train_columns = train_data.columns
val_columns = val_data.columns

# Step 5: Ask the user to choose the columns for training
print(f"Available columns in training dataset: {train_columns}")
text_column = input(f"Please choose the text column from the training dataset (e.g. combined_text): ").strip()
target_column = input(f"Please choose the target column (e.g., 'Authors') from the training dataset: ").strip()

# Step 6: Prepare the data for training
texts_train = train_data[text_column].tolist()
labels_train = train_data[target_column].tolist()

texts_eval = val_data[text_column].tolist()
labels_eval = val_data[target_column].tolist()

# Step 7: User provides task instruction and number of epochs
task_instruction = input("Enter the task instruction (e.g., 'Extract Authors'): ").strip()
num_epochs = int(input("Enter the number of epochs (e.g., 3): "))

# Step 8: Train the model
model.train(texts_train, labels_train, texts_eval, labels_eval, task_instruction, num_epochs)

# Save the fine-tuned model
save_model_path = './info_extraction_model'
model.save_fine_tuned_model(save_model_path)
print(f"Model saved at {save_model_path}")
```
### 3. Extract Information and Evaluate Results
Now that the model is trained, you can extract information from new data and evaluate the results.

```python
# Step 9: Load the trained model
model.load_model('./info_extraction_model')
print("Model loaded from './info_extraction_model'")

# Step 10: Extract information from new text
new_text='Complete resolution of cutaneous larva migrans with topical ivermectin: A case report Francesca  Magri, Camilla  Chello, Giulia  Pranteda, Guglielmo  Pranteda Cutaneous larva migrans (CLM; also called creeping eruption) is a cutaneous ectoparasitosis commonly observed in tropical countries. It is characterized by an erythematous, pruritic, and raised lesion with linear or serpiginous distribution, typically localized at the lower extremities. Oral ivermectin represents the most recommended current treatment, with important adverse effects associated. We report the clinical case of a 52‐year old with CLM, successfully treated with topical ivermectin.'
a=model.extract(new_text,task_instruction)

```