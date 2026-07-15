import unittest
from data.tokenizer import CaptionTokenizer

class TestCaptionTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CaptionTokenizer()

    def test_basic_tokenization(self):
        caption = "A girl sits in a pool ."
        expected = ["a", "girl", "sits", "in", "a", "pool"]
        self.assertEqual(self.tokenizer.tokenize(caption), expected)

    def test_contraction_tokenization(self):
        caption = "It's a beautiful day, isn't it?"
        expected = ["it's", "a", "beautiful", "day", "isn't", "it"]
        self.assertEqual(self.tokenizer.tokenize(caption), expected)

    def test_lowercase_and_strip(self):
        caption = "  Hello WORLD!  "
        expected = ["hello", "world"]
        self.assertEqual(self.tokenizer.tokenize(caption), expected)

    def test_vietnamese_lowercase_and_strip(self):
            caption = " Xin chào Việt NAM  "
            expected = ["xin", "chào", "việt", "nam"]
            self.assertEqual(self.tokenizer.tokenize(caption), expected)

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.tokenizer.tokenize(123)  # type: ignore

if __name__ == "__main__":
    unittest.main() 
