# 🚀 Rubick: Automating Loss Function Generation Using LLMs

## 📘 Project Overview

This project automates a critical aspect of the neural network training pipeline: **loss function generation**. By leveraging large language models (LLMs) like **CodeLlama**, we dynamically generate PyTorch-compatible loss functions based on natural language task descriptions.

The system builds a full pipeline that covers:

* Prompt engineering and construction
* Code generation based on task description
* Unit test generation
* Dynamic code validation and correction in case of errors

This work lays the foundation for automating the entire ML model development cycle — enabling **adaptive, task-aware AI systems** that minimize manual intervention.

---

## 🎯 Key Features

* ⚙️ **Task-Specific Loss Functions**: Automatically generates loss functions tailored to the given task.
* 🧪 **Automated Testing**: Dynamically generates and runs unit tests to validate correctness.
* 🔁 **Self-Correcting Loop**: Identifies and fixes errors in generated code using model-powered feedback.
* 🧠 **Model-Agnostic Design**: Easily switchable between different LLM backends (e.g., CodeLlama, DeepSeek, StarCoder).

---

## ⚠️ Challenges Addressed

| Issue              | Description                                                                     |
| ------------------ | ------------------------------------------------------------------------------- |
| **Echoing**        | LLMs repeating prompt content                                                   |
| **Code Drift**     | Long sessions can cause format inconsistency                                    |
| **Malformed Code** | Syntax errors or incorrect imports occasionally occur                           |
| **Test Failures**  | Some test cases are invalid or misaligned with the task semantics               |

---

## ✅ Best Practices

* Use **instruction-tuned LLMs** (e.g., `codellama/CodeLlama-7b-Instruct-hf`, `DeepSeek-Coder-6.7B-Instruct`) for higher reliability.
* Clear cache or session state regularly to avoid code drift.
* Maintain strict prompt templates for both **loss generation** and **unit test generation**.

---

## 🧪 Output Samples

Curated examples are available in the `notebooks/` folder to highlight the model’s behavior in both successful and failure scenarios.

| Sample   | Description                                                                     | File                                          |
| -------- | ------------------------------------------------------------------------------- | --------------------------------------------- |
| Sample 1 | Loss generation for MNIST classification and end-to-end implementation          | `notebooks/rubick_mnist_e2e.ipynb`            |
| Sample 2 | Loss generation for CIFAR-10 classification and end-to-end implementation       | `notebooks/rubick_cifar10_e2e.ipynb`          |
| Sample 3 | Example showcasing errornous loss generation and auto-fixing the generated code | `notebooks/rubick_catsdog_autofix_demo.ipynb` |
| Sample 4 | Example showcasing errornous loss generation and auto-fixing the generated code | `notebooks/rubick_mnist_autofix_demo.ipynb`   |

These samples showcase:

* The impact of prompt phrasing
* Adherence to PyTorch standards
* Effectiveness of the feedback loop

> 🔧 Additional samples and benchmark tasks are being added to improve generalization and robustness testing.

---

## 🔬 Current Research Focus

We're analyzing how variations in task descriptions influence the generated loss functions:

* Task phrasing vs. function semantics
* Loop stability and correction efficiency
* Generalizability across datasets

---

## 🧹 System Architecture

### `Rubick` Class (Core Engine)

* **Loss Function Generator**: Generates PyTorch loss functions from prompts.
* **Unit Test Generator**: Produces validation tests based on the generated code.
* **Error Classifier**: Diagnoses test failures and identifies the correction target.
* **Executor**: Dynamically loads, executes, and evaluates generated Python modules.

### 🗺️ Internal Component Flow

![UML Diagram](docs/uml_diagram.png)


---

## 🧰 Technologies & Frameworks

* Python
* PyTorch
* Hugging Face Transformers
* CodeLlama / DeepSeek
* `unittest`, `importlib`, `io`, `re`
* **Rutgers Amarel HPC Cluster**

---

## 🗂️ Project Structure

```bash
Rubick: Automating-Loss-Function-Generation-Using-LLMs/
│
├── rubick/                        # Core pipeline logic
│   ├── __init__.py
│   └── rubick.py
│
├── notebooks/                     # LLM experiment notebooks
│   ├── rubick_castdog_autofix_demo.ipynb
│   ├── rubick_cifar10_e2e.ipynb
│   ├── rubick_mnist_autofix_demo.ipynb
│   └── rubick_mnist_e2e.ipynb
│
├── tests/                         # Unit test scripts
│   ├── __init__.py
│   └── test_rubick.py
│
├── docs/                          # HPC cluster connection and gpu access documentation
│   └── clusterConnectionSteps.md
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files to ignore
└── README.md                     # Project overview (this file)
```

---

## 👥 Team

* Gaurav Shetty
* Naimish Sharma
* Under the guidance of Prof. Mauro Sanchirico

---

## 📨 Contributing

We welcome contributions! Feel free to:

* Open an issue
* Add new sample outputs
* Suggest improvements to the generation prompts or pipeline
