# Automating Loss Function Generation Using LLMs

## Project Description
This project focuses on automating a critical component of the neural network training pipeline: **loss function generation**. By leveraging large language models (LLMs) such as CodeQwen-1.5-7B, we dynamically generate PyTorch-compatible loss functions from natural language task descriptions. 

The system builds a robust pipeline that handles prompt construction, code generation, validation through unit tests, and iterative correction in case of errors. Through this process, we investigate how LLMs can replace manual engineering efforts in model development.

This work serves as a foundational step toward automating the entire machine learning development and training pipeline — enabling adaptive, task-aware systems that reduce the need for human intervention in model design.

The system uses a multi-step process: prompt-based generation of loss function code, extraction and saving of the function, automatic test case generation, and validation via dynamic code execution. Failures trigger error classification and correction steps, all coordinated within a self-improving loop.

## Highlights
- Generates task-specific loss functions dynamically using code-generation LLMs.
- Automates unit testing and validation of the generated loss functions.
- Employs a feedback loop to iteratively fix errors or refine code using model outputs.
- Model independent framework and compatible with other LLM models.

## Key Challenges
- **Echoing**: LLMs often repeat prompt content, requiring string filtering and structured extraction.
- **Code Drifting**: In longer sessions or repeated calls, the LLM sometimes begins to deviate from previously followed instructions eg: ignoring format constraints.
- **Model Limitations**: Models occasionally fail to understand edge-case task semantics or generate malformed Python code.
- **Test Inaccuracy**: In some cases, unit tests are incorrectly generated, which can mislead the correction loop.

## Best Practices & Tips
- Use **instruction-tuned LLMs** with strong code generation capabilities (e.g., `codellama/CodeLlama-7b-Instruct-hf`, `DeepSeek-Coder-6.7B-Instruct`) for reliable results.
- Ensure to clean cache after long sessions of usage to avoid drifting
- Maintain structured prompts for both **loss function generation** and **unit test creation**.

## Output Samples

To demonstrate how the model performs across different scenarios, several curated samples have been added to the `output_samples/` directory. These include both successful and failed generations, showcasing the strengths and limitations of the system.

### 🔍 Included Samples:
- **Sample 1**: Loss function generation for image classification using the MNIST dataset  
  `output_samples/CIFAR-10_SAMPLE.ipynb`

- **Sample 2**: Loss function generation tailored for the CIFAR-10 dataset  
  `output_samples/MNIST_SAMPLE.ipynb`

- **Sample 3**: Example where the model classifies the error in the generated code and fixes it  
  `output_samples/code_testing_example.ipynb`

These examples help illustrate:
- How task phrasing influences the generated loss logic
- How well the LLM adheres to PyTorch standards
- The effectiveness of the automated correction and testing pipeline

> ⚠️ More samples and benchmark cases are being added to broaden coverage across tasks and error types.

## Current Focus
We are currently studying how changes in the **natural language task description** influence the structure and semantics of the generated loss function. This includes analyzing:
- Differences in function logic based on task phrasing.
- Robustness of the generation loop.
- Generalization of the learned loss functions across datasets.

## System Components
- **Rubick Class**: Core engine that controls model inference, prompt formatting, loss generation, testing, and retries.
  - **Loss Function Generator**: Dynamically creates loss function based on the given task description.
  - **Unit Test Generator**: Dynamically creates test cases based on the generated loss function.
  - **Error Classifier**: Categorizes and routes errors to the appropriate correction mechanism.
  - **Executor**: Imports and runs generated code in isolated runtime environments with output capture.

## Technologies Used
- PyTorch  
- Hugging Face Transformers  
- codellama/CodeLlama-7b-Instruct-hf  
- Python `unittest`, `importlib`, `re`, `io`  
- Deployed on Rutgers Amarel HPC cluster  

## Team Members
- Gaurav Shetty  
- Naimish Sharma

---

Feel free to open an issue or contribute examples if you're exploring LLM-based automation of model training workflows.
