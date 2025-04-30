import torch
import pandas
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import sys
import unittest
import importlib.util
import io
import contextlib
import zipfile

class Rubick:
    def __init__(self, model_id, token, prompt):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map={"": self.device}
        )
        self.prompt = prompt
        self.loss_generation_message = (
            "You are an AI agent tasked with writing a **PyTorch-compatible loss function** based on a task description.\n"
            "### Rules you MUST follow:\n"
            "- Name the class `AutoLoss`.\n"
            "- Inherit from `torch.nn.Module`.\n"
            "- Import necessary PyTorch libraries - torch, torch.nn, torch.nn.functional.\n"
            "- DO NOT import any non-PyTorch or external libraries.\n"
            "- Output only valid Python code.\n"
            "- Wrap your code inside proper ```python``` code fences.\n"
            "- DO NOT add any explanations, comments, or extra text. Output ONLY the code.\n"
            "- If unsure, output a minimal valid PyTorch loss class template.\n\n"
            "### Task Description:\n"
            f"{self.prompt}\n\n"
            "### Response - Write the Loss Function Code:\n"
            "```python\n"
        )


        # look up survey | add extra info
        self.test_generation_message = (
            "You are an AI agent tasked with writing a unit test in Python to validate a PyTorch-compatible loss function class.\n\n"
            "You will be given the code of the loss function, write a python unit test code to check if the function is running without any error.\n"
            "Instructions:\n"
            "- Write **only** valid Python code. No explanations, comments, or extra text.\n"
            "- Wrap the code **inside** a ```python``` code block.\n"
            "- Import **only** necessary PyTorch libraries.\n"
            "- Do **not** import unused packages or modules.\n"
            "- Ensure the test runs independently.\n"
            "- If unsure, leave the code minimal but runnable.\n\n"
        )


        #test classification
        self.error_classification_message = (
            "You are an expert AI agent tasked with classifying where the error is occurring in a PyTorch loss function testing task.\n\n"
            "You will be given:\n"
            "- The Loss Function Code (PyTorch)\n"
            "- The Unit Test Code (PyTorch unittest)\n"
            "- The Error Output (from running the unit test)\n\n"
            "Your task:\n"
            "- Carefully read the Loss Function Code, Unit Test Code, and Error Output.\n"
            "- Decide which code is causing the problem: the Loss Function, the Test Code, or if it cannot be determined, output Unknown.\n"
            "- Only respond with one of these three words: 'loss' for error in loss function code, 'test' for error in test function code, or 'unknown'. No explanation. No extra text.\n\n"
            "### Loss Function Code - stored in temp_code.py:\n"
            "{loss_code}\n\n"
            "### Unit Test Code - stored in temp_test.py:\n"
            "{test_code}\n\n"
            "### Error Output - occured when testing code present in temp_code.py using unittest present in temp_test.py:\n"
            "{error_output}\n\n"
            "### Classification:\n"
        )


    def _write_file(self, path, content):
        with open(path, "w") as f:
            f.write(content)
    
    def _classify_error(self, loss_code, test_code, error_output):
        classification_prompt = self.error_classification_message.format(
            loss_code=loss_code,
            test_code=test_code,
            error_output=error_output
        )
        
        classification = self.generate_response(classification_prompt, max_length=20, temperature=0.1)
        classification = classification.split("### Classification:\n")[-1].strip().lower()
        print("\n Rubick found an error and says the error is in:", classification,"-", type(classification))
        if classification in ["loss", "test", "unknown"]:
            return classification
        else:
            # Fallback to manual rules if model output is invalid
            if "AutoLoss" in error_output or "TypeError" in error_output or "AttributeError" in error_output:
                return "loss"
            if "AssertionError" in error_output or "FAILED" in error_output:
                return "test"
            return "unknown"

    def check_python_tag(self, response):
        code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL) # check response if contains no code
        if code_blocks:
            status = True
        else:
            print("Code blocks not properly formatted or not found in LLM response\n")
            status = False
        return status    
            

    def _build_test_generation_prompt(self, loss_code):
        return (
            "You are an AI agent tasked with writing a unit test in Python to validate a PyTorch-compatible loss function class.\n\n"
            "You will be given the code of the loss function, write a python unit test code to check if the function is running without any error.\n"
            "Instructions:\n"
            "- Write **only** valid Python code. No explanations, comments, or extra text.\n"
            "- Wrap the code **inside** a ```python``` code block.\n"
            "- Import **only** necessary PyTorch libraries.\n"
            "- Do **not** import unused packages or modules.\n"
            "- Ensure the test runs independently.\n"
            "- If unsure, leave the code minimal but runnable.\n\n"
            "Loss function which you need to test:\n\n"
            f"{loss_code}\n\n"
            "### Response - Write the Unit Test Code:\n"
            "```python\n"
        )
    
    def _build_loss_fix_prompt(self, loss_code, error_output):
        return (
            "You are an AI tasked with fixing a PyTorch loss function class.\n\n"
            "Instructions:\n"
            "- Only fix errors related to PyTorch code correctness.\n"
            "- Only import PyTorch libraries (torch, torch.nn, torch.nn.functional).\n"
            "- Output ONLY valid Python code wrapped inside ```python``` and ```.\n"
            "- Do NOT add any explanations, comments, markdown headings, or extra text.\n"
            "- If you cannot fix it, still output a valid PyTorch loss class.\n\n"
            "Previous Loss Function Code:\n"
            f"{loss_code}\n\n"
            "Error Log:\n"
            f"{error_output}\n\n"
            "### Write ONLY the corrected Loss Function Code below:\n"
            "```python\n"
        )
    
    
    def _build_test_fix_prompt(self, loss_code, old_test_code, error_output):
        return (
            "You are an AI tasked with fixing or generating a UnitTest for a PyTorch-compatible loss function class named `AutoLoss`.\n\n"
            "### Strict Rules:\n"
            "- Always wrap the complete response inside a single ```python``` block.\n"
            "- Only generate valid Python code. No explanation, no comments, no notes, no sources.\n"
            "- Assume the `AutoLoss` class is saved in `temp_code.py`.\n"
            "- Import it using: `from temp_code import AutoLoss`\n"
            "- Only use PyTorch-related modules and standard Python libraries needed for testing.\n"
            "- If unsure, still output the best possible valid UnitTest code inside the ```python``` block.\n\n"
            "### Given Information:\n"
            "**Loss Function Code:**\n"
            f"{loss_code}\n\n"
            "**Previous Unit Test Code:**\n"
            f"{old_test_code}\n\n"
            "**Error Message:**\n"
            f"{error_output}\n\n"
            "### Response - Write the corrected Unit Test Code inside ```python``` block:\n"
        )


    
    # Gives prompt and generates LLM response
    def generate_response(self, messages, max_length=512, temperature=0.5):
        if isinstance(messages, list):
            chat_formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            chat_formatted_prompt = messages
    
        inputs = self.tokenizer(chat_formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95
            )
    
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    # Extract code from the response
    def extract_code(self, model_output):
        code_blocks = re.findall(r"```python(.*?)```", model_output, re.DOTALL)
    
        # Filter out empty or whitespace-only blocks
        code_blocks = [block.strip() for block in code_blocks if block.strip()]
    
        if code_blocks:
            extracted_code = code_blocks[0]  # Safely take the first non-empty block
        else:
            extracted_code = "Could not be extracted"  # No valid code found
    
        return extracted_code

    
    # Extract imports from the code
    def extract_imports(self, code_str):
        import_lines = re.findall(r'^\s*(?:import|from)\s+[^\n]+', code_str, re.MULTILINE)
        return "\n".join(import_lines)


    # Import required libraries
    def execute_imports(self, import_block):
        for line in import_block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                exec(stripped, globals())
            except ModuleNotFoundError as e:
                print(f"❌ Missing library: '{e.name}' — please install it.")
            except Exception as e:
                print(f"⚠️ Failed to import line: '{stripped}'\n   Reason: {e}")
                
    
    # Execute the code and store the variables in a local environment
    def execute_code(self, extracted_code):
        try:
            print("\n=== Extracted Code Being Executed ===")
            print(extracted_code)
            print("=====================================\n")
            local_vars = {}
            exec(extracted_code, globals(), local_vars)
            if "AutoLoss" not in local_vars:
                raise ValueError("AutoLoss class could not be found in the executed code.")
            self.generated_loss_class = local_vars["AutoLoss"]
            setattr(self, "AutoLoss", self.generated_loss_class)
        except Exception as E:
            print("❌ Error during loss class exec:", E)


    def load_module_from_file(self, file_path, module_name="temp_code"):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    
    # To validate the generated loss function using the unit test
    def run_unit_tests(self, test_code: str, loss_code: str, temp_dir="temp_gen") -> (bool, str):
        os.makedirs(temp_dir, exist_ok=True)
        code_file = os.path.join(temp_dir, "temp_code.py")
        test_file = os.path.join(temp_dir, "temp_test.py")

        if "temp_code" in sys.modules:
            del sys.modules["temp_code"]
        if "temp_test" in sys.modules:
            del sys.modules["temp_test"]
        
        sys.path.insert(0, temp_dir)
        module_name = "temp_test"
        spec = importlib.util.spec_from_file_location(module_name, test_file)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)

        suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)

        output_stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=output_stream, verbosity=2)

        with contextlib.redirect_stdout(output_stream), contextlib.redirect_stderr(output_stream):
            result = runner.run(suite)

        sys.path.pop(0)
        test_output = output_stream.getvalue()
        return result.wasSuccessful(), test_output

                        
            
    # generate and extract code based on the message            
    def generate_code(self, message):
        status = False
        limit = 0
        while not status and limit<3:
            response = self.generate_response(message)
            status = self.check_python_tag(response)
            limit +=1
        if status:
            exec_code = self.extract_code(response)
            exec_imp = self.extract_imports(exec_code)
        else:
            print("code extraction failed")
            exec_code,exec_imp = "", ""
        return exec_code,exec_imp
    
    
    def smart_loop_testing(self, full_loop_attempts=3, half_loop_attempts=3):
        full_loop_counter = 0
        code_validity = False
        while full_loop_counter < full_loop_attempts and code_validity == False:
            inner_loop_counter = 0
            status = False
            loss_code, loss_imp = self.generate_code(self.loss_generation_message)
        
            self._write_file("temp_gen/temp_code.py", loss_code)
        
            # Prepare initial unit test
            test_generation_message = self._build_test_generation_prompt(loss_code)
            test_code, test_imp = self.generate_code(test_generation_message)
            test_code = f"from temp_code import AutoLoss\n\n{test_code}"
            self._write_file("temp_gen/temp_test.py", test_code)

            print("Here is initial code generated for loop: ", full_loop_counter)
            print("\nLoss function code:\n ", loss_code)
            print("\nTest function code:\n", test_code)
        
            while inner_loop_counter < half_loop_attempts and status==False:
                try:
                    status, error_output = self.run_unit_tests(test_code, loss_code)
                    code_validity = status
                except Exception as e:
                    status = False
                    error_output = f"Exception during unit tests: {str(e)}"
        
                print(f"\n[Attempt {inner_loop_counter+1}/{half_loop_attempts}] Status: {status}")
                print(f"Error Output:\n{error_output}\n")
        
                if status:
                    self.extracted_code = loss_code
                    print("Tests passed successfully!\n")
                    if inner_loop_counter > 0:
                        print("Rubick was able to solve the error successfully.\n")
                else:
                    # Classify the error to decide what to fix
                    fix_target = self._classify_error(loss_code, test_code, error_output)
            
                    if fix_target == "loss":
                        print("[Fixing Loss Function]")
                        self.loss_generation_message = self._build_loss_fix_prompt(loss_code, error_output)
                        loss_code, loss_imp = self.generate_code(self.loss_generation_message)
                        self._write_file("temp_gen/temp_code.py", loss_code)
            
                    elif fix_target == "test":
                        print("[Fixing Unit Test]")
                        self.test_generation_message = self._build_test_fix_prompt(loss_code, test_code, error_output)
                        test_code, test_imp = self.generate_code(self.test_generation_message)
                        test_code = f"from temp_code import AutoLoss\n\n{test_code}"
                        self._write_file("temp_gen/temp_test.py", test_code)
            
                    else:
                        print("[Unknown error. Retrying loss function fix by default]")
                        self.loss_generation_message = self._build_loss_fix_prompt(loss_code, error_output)
                        loss_code, loss_imp = self.generate_code(self.loss_generation_message)
                        self._write_file("temp_gen/temp_code.py", loss_code)
                inner_loop_counter += 1
            full_loop_counter += 1
            code_validity = status
        if not code_validity:
            print("Couldn't generate loss function. Please reformat the prompt for task description\n")
            return "","",code_validity
        else:
            return loss_code, loss_imp, code_validity

    
    # Returns the latest generated Loss Function
    def cast_loss(self):
        return self.extracted_code

    # Main function to initiate the process
    def process_start(self):
        print("Starting loss function generation process")
        loss_code, loss_imp, status = self.smart_loop_testing()
        if status:
            print("Loss function generated successfully. Executing generated function.")
            # self.execute_imports(loss_imp)
            # self.execute_code(loss_code)
            module = self.load_module_from_file("temp_gen/temp_code.py")  # Or wherever the loss is saved
            self.AutoLoss = getattr(module, "AutoLoss")
            print("Loss function executed and is now ready to use!")
        else:
            print("❌ Could not generate a valid loss function after 3 attempts.")
