import unittest
from unittest.mock import patch, MagicMock
import torch
from rubick import Rubick

class TestRubick(unittest.TestCase):

    # Testing the initialization function
    @patch('rubick.AutoTokenizer.from_pretrained')
    @patch('rubick.AutoModelForCausalLM.from_pretrained')
    def test_initialization(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        rubick = Rubick("fake-model-id", "fake-token", "fake-prompt")
        self.assertIsNotNone(rubick.tokenizer)
        self.assertIsNotNone(rubick.model)
        self.assertTrue(isinstance(rubick.device, torch.device))

    # Testing the code extraction function
    @patch('rubick.AutoTokenizer.from_pretrained')
    @patch('rubick.AutoModelForCausalLM.from_pretrained')
    def test_extract_code(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        rubick = Rubick("fake-model-id", "fake-token", "fake-prompt")
        code_block = "```python\nclass AutoLoss:\n    pass\n```"
        empty_block = ""
        multi_code_block = "```python\nclass First:\n    pass\n```   ```python\nclass Second:\n    pass\n```"
        extracted_code_block = rubick.extract_code(code_block)
        extracted_empty_block = rubick.extract_code(empty_block)
        extracted_multi_block = rubick.extract_code(multi_code_block)
        self.assertIn("class AutoLoss", extracted_code_block)
        self.assertEqual("Could not be extracted", extracted_empty_block)
        self.assertIn("class First", extracted_multi_block)
        self.assertNotIn("class Second", extracted_multi_block)

    # Testing the code generation function
    @patch.object(Rubick, 'generate_response')
    @patch('rubick.AutoModelForCausalLM.from_pretrained')
    @patch('rubick.AutoTokenizer.from_pretrained')
    def test_generate_code(self, mock_tokenizer, mock_model, mock_generate_response):
        mock_generate_response.side_effect = [
            "No code here",
            "Still broken",
            "```python\nimport torch\nclass AutoLoss(nn.Module): pass\n```"
        ]
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        rubick = Rubick("fake-model-id", "fake-token", "fake-prompt")
        code, imports = rubick.generate_code("mocked prompt")
        self.assertIn("AutoLoss", code)
        self.assertIn("import torch", imports)
        
    # Testing the code generation and error correction loop
    @patch.object(Rubick, 'run_unit_tests')
    @patch.object(Rubick, 'generate_code')
    @patch('rubick.AutoModelForCausalLM.from_pretrained')
    @patch('rubick.AutoTokenizer.from_pretrained')
    def test_smart_loop_success_first_try(self, mock_tokenizer, mock_model, mock_generate_code, mock_run_tests):
        # Mocking the loss code and test code generation
        mock_generate_code.side_effect = [
            ("loss_code", "import torch"),  # generate loss
            ("def test_loss(): pass", "import unittest")  # generate test
        ]
        mock_run_tests.return_value = (True, "")  # test passes on first try
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        rubick = Rubick("model-id", "token", "fake-prompt")
        loss, imp, status = rubick.smart_loop_testing()
        self.assertEqual(loss, "loss_code")
        self.assertTrue(status)



if __name__ == "__main__":
    unittest.main()