import os
import unittest

import numpy as np
import torch

from train import get_batch


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create temporary data for testing
        self.data_dir = "temp_data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.train_data = np.arange(1000, dtype=np.uint16)
        self.val_data = np.arange(1000, 2000, dtype=np.uint16)
        self.block_size = 32
        self.batch_size = 8

        np.save(os.path.join(self.data_dir, "train.bin"), self.train_data)
        np.save(os.path.join(self.data_dir, "val.bin"), self.val_data)

    def tearDown(self):
        # Clean up temporary data directory
        os.remove(os.path.join(self.data_dir, "train.bin.npy"))
        os.remove(os.path.join(self.data_dir, "val.bin.npy"))
        os.rmdir(self.data_dir)

    def test_get_batch_train(self):
        split = "train"
        config = {
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        x, y = get_batch(split, config=config, train_data_in=self.train_data)

        # Check if x and y have the correct shapes
        self.assertEqual(x.shape, (self.batch_size, self.block_size))
        self.assertEqual(y.shape, (self.batch_size, self.block_size))

        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_get_batch_val(self):
        split = "val"
        config = {
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        x, y = get_batch(split, config=config, val_data_in=self.val_data)
        # Check if x and y have the correct shapes
        self.assertEqual(x.shape, (self.batch_size, self.block_size))
        self.assertEqual(y.shape, (self.batch_size, self.block_size))

        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_data_location_commavq(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data/commavq")
        files = os.listdir(data_dir)
        self.assertIn("meta.pkl", files)
        self.assertIn("train.bin", files)
        self.assertIn("val.bin", files)

    def test_data_length_commavq(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data/commavq")
        train_data = np.memmap(
            os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
        )
        val_data = np.memmap(
            os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
        )
        self.assertEqual(len(train_data), 15480100000)
        self.assertEqual(len(val_data), 15480100)


if __name__ == "__main__":
    unittest.main()
