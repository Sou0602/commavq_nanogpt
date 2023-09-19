import argparse
import os

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm as tqdm

from config.eval_commavq import DatasetConstants, ModelConfig, PathsConfig


def main():
    """
    Main function for evaluation.

    Args:
        ort_path (str, optional): Path to the ONNX model. Defaults to None.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate a pre-trained GPT model on the commavq dataset."
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        help="Path to the Onnx file. If not provided, the default checkpoint path from configuration is used.",
    )
    args = parser.parse_args()

    system_config = ModelConfig()
    paths_config = PathsConfig()
    data_consts = DatasetConstants()

    torch.manual_seed(system_config.SEED + system_config.seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    try:
        files, indices = get_eval_files_indices(consts=data_consts)
        if args.onnx_path is None:
            ort_path = os.path.join(
                os.path.dirname(__file__), paths_config.out_dir, "ckpt.onnx"
            )
        elif not args.ckpt_path.endswith(".onnx"):
            print("Error: Invalid Onnx file. The file must have a .onnx extension.")
            return
        else:
            ort_path = args.onnx_path
            print("Onnx File Path updated to the Input Path")

        ## For inference session stability and speed-ups
        ## https://onnxruntime.ai/docs/performance/tune-performance/
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        sess_options.add_session_config_entry("session.dynamic_block_base", "4")

        assert system_config.device == "cuda"

        provider = "CUDAExecutionProvider"
        ort_session = onnxruntime.InferenceSession(
            ort_path, sess_options, providers=[provider]
        )
        binding = ort_session.io_binding()

        total_losses = []
        losses, sizes = [], []
        for f in tqdm(files):
            tokens = np.load(f)
            tokens = preprocess_tokens(tokens, consts=data_consts)
            for ii in indices:
                with torch.no_grad():
                    loss, size = compute_loss(ort_session, binding, tokens, ii)
                    losses.append(loss)
                    sizes.append(size)

        total_loss = np.sum(losses) / np.sum(sizes)
        total_losses.append(total_loss)
        print(f"total loss {np.mean(total_losses)}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_eval_files_indices(consts, data_path="commaai/commavq"):
    """
    Load evaluation dataset and generate indices for slicing data.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        list: List of file paths.
        list: List of data indices for slicing.
    """
    ds = load_dataset(data_path, num_proc=40)
    files = ds["40"]["path"]  # 40 is the validation set

    # Create data slicing
    indices = np.arange(0, consts.N * consts.TOKENS_PER_FRAME)
    indices = np.array(np.split(indices, consts.N // consts.CONTEXT_SIZE_FRAMES))
    # Batch them
    indices = [indices[i : i + consts.BS] for i in range(0, len(indices), consts.BS)]

    return files, indices


def preprocess_tokens(tokens, consts):
    """
    Preprocess tokens for model input.

    Args:
        tokens (numpy.ndarray): Input tokens.

    Returns:
        torch.Tensor: Processed tokens as a PyTorch tensor.
    """
    tokens = tokens.reshape(
        consts.N_FRAMES, consts.TOKENS_PER_FRAME - 1
    )  # TOKENS_PER_FRAME includes the BOS token
    tokens = np.c_[np.ones(len(tokens), dtype=np.int64) * consts.BOS_TOKEN, tokens]
    tokens = tokens.reshape(-1)
    tokens = torch.from_numpy(tokens).long().cuda()
    return tokens


def compute_loss(ortsession, binding, tokens, indices):
    """
     Compute the loss for a batch of tokens.

    Args:
         ort_session (onnxruntime.InferenceSession): Inference Session.
         binding (onnxruntime.io_binding): ONNX Runtime IO Binding.
         tokens (torch.Tensor): Input tokens.
         indices (numpy.ndarray): Data indices for slicing.

     Returns:
         float: Loss for the batch.
         int: Batch size.
    """
    with torch.no_grad():
        x = tokens[indices.ravel()]
        x = x.reshape(indices.shape[0], indices.shape[1])

        X_tensor = x.contiguous()
        binding.bind_input(
            name="x",
            device_type="cuda",
            device_id=0,
            element_type=np.int64,
            shape=tuple(X_tensor.shape),
            buffer_ptr=X_tensor.data_ptr(),
        )

        binding.bind_output("y", "cuda")
        ortsession.run_with_iobinding(binding)
        ort_output = binding.copy_outputs_to_cpu()[0]
        pred = torch.from_numpy(ort_output).to("cuda")
        y = tokens[indices.ravel() + 1]
        y = y.reshape(indices.shape[0], indices.shape[1])
        loss = (
            F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
            .detach()
            .cpu()
            .numpy()
            * indices.shape[0]
        )
        return loss, indices.shape[0]


if __name__ == "__main__":
    main()
