import unittest
import torch
from data.tokenizer import CaptionTokenizer
from data.vocabulary import Vocabulary
from data.collate import ImageCaptionCollator
from models.text_embedding import TextEmbedding

class TestRoleCComponents(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CaptionTokenizer()
        self.vocab = Vocabulary(self.tokenizer)
        
        # Build a small dummy vocabulary manually for testing
        self.vocab.token_to_id = {
            self.vocab.PAD_TOKEN: self.vocab.PAD_TOKEN_ID,
            self.vocab.BOS_TOKEN: self.vocab.BOS_TOKEN_ID,
            self.vocab.EOS_TOKEN: self.vocab.EOS_TOKEN_ID,
            self.vocab.UNK_TOKEN: self.vocab.UNK_TOKEN_ID,
            "a": 4,
            "girl": 5,
            "sits": 6,
            "in": 7,
            "pool": 8
        }
        self.vocab.id_to_token = {v: k for k, v in self.vocab.token_to_id.items()}

    def test_tokenizer(self):
        caption = "A girl sits in a pool ."
        tokens = self.tokenizer.tokenize(caption)
        expected_tokens = ["a", "girl", "sits", "in", "a", "pool"]
        self.assertEqual(tokens, expected_tokens)

    def test_vocabulary_encode_decode(self):
        caption = "A girl sits in a pool"
        
        # Test encode
        encoded = self.vocab.encode(caption, max_length=10)
        expected_encoded = [
            self.vocab.BOS_TOKEN_ID,
            4, 5, 6, 7, 4, 8, # a, girl, sits, in, a, pool
            self.vocab.EOS_TOKEN_ID
        ]
        self.assertEqual(encoded, expected_encoded)

        # Test decode (skip special tokens)
        decoded = self.vocab.decode(encoded, skip_special_tokens=True)
        self.assertEqual(decoded, "a girl sits in a pool")

        # Test decode (keep special tokens)
        decoded_with_special = self.vocab.decode(encoded, skip_special_tokens=False)
        self.assertEqual(decoded_with_special, "<BOS> a girl sits in a pool")

    def test_collator(self):
        collator = ImageCaptionCollator(pad_token_id=self.vocab.pad_token_id)
        
        # Mock batch of data
        dummy_batch = [
            {
                "filename": "img1.jpg",
                "image": torch.randn(3, 224, 224),
                "caption": "a girl sits",
                "caption_ids": torch.tensor([self.vocab.BOS_TOKEN_ID, 4, 5, 6, self.vocab.EOS_TOKEN_ID]) # BOS, a, girl, sits, EOS
            },
            {
                "filename": "img2.jpg",
                "image": torch.randn(3, 224, 224),
                "caption": "a pool",
                "caption_ids": torch.tensor([self.vocab.BOS_TOKEN_ID, 4, 8, self.vocab.EOS_TOKEN_ID]) # BOS, a, pool, EOS
            }
        ]
        
        collated = collator(dummy_batch)
        
        # Check keys
        self.assertIn("images", collated)
        self.assertIn("caption_ids", collated)
        self.assertIn("caption_padding_mask", collated)
        
        # Check shapes
        self.assertEqual(collated["images"].shape, (2, 3, 224, 224))
        # Longest caption is length 5
        self.assertEqual(collated["caption_ids"].shape, (2, 5))
        self.assertEqual(collated["caption_padding_mask"].shape, (2, 5))
        
        # Check padding of second sentence (length 4 padded to 5)
        self.assertEqual(collated["caption_ids"][1, -1].item(), self.vocab.pad_token_id)
        self.assertTrue(collated["caption_padding_mask"][1, -1].item())
        self.assertFalse(collated["caption_padding_mask"][0, -1].item())

    def test_text_embedding(self):
        d_model = 64
        max_sequence_length = 20
        embedding_layer = TextEmbedding(
            vocabulary_size=len(self.vocab),
            d_model=d_model,
            max_sequence_length=max_sequence_length,
            pad_token_id=self.vocab.pad_token_id,
            dropout=0.0
        )
        
        # Batch of token ids (batch_size=2, sequence_length=5)
        token_ids = torch.tensor([
            [1, 4, 5, 6, 2],
            [1, 4, 8, 2, 0]
        ])
        
        embeddings = embedding_layer(token_ids)
        self.assertEqual(embeddings.shape, (2, 5, d_model))
        
        # Ensure padding token embedding is scaled correctly or remains deterministic
        # (Though dropout is 0.0, we want to check that it runs without errors)
        self.assertFalse(torch.isnan(embeddings).any())

    def test_image_captioning_transformer(self):
        from models.image_captioning_transformer import ImageCaptioningTransformer
        model = ImageCaptioningTransformer(
            image_size=224,
            patch_size=16,
            vocabulary_size=len(self.vocab),
            max_caption_length=10,
            pad_token_id=self.vocab.pad_token_id,
            d_model=64,
            num_heads=2,
            d_ff=128,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.0
        )
        
        # Batch of images: [B, 3, 224, 224]
        images = torch.randn(2, 3, 224, 224)
        
        # Batch of caption ids: [B, seq_len]
        caption_ids = torch.tensor([
            [1, 4, 5, 2],
            [1, 4, 8, 2]
        ])
        
        logits = model(images, caption_ids)
        
        # Kích thước của logits đầu ra phải là [Batch_Size, seq_len, vocabulary_size] -> [2, 4, len(vocab)]
        self.assertEqual(logits.shape, (2, 4, len(self.vocab)))

if __name__ == "__main__":
    unittest.main()
