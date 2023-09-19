"""
Reads a torch model from checkpoint path in config and then converts it into an onnx model to onnx path in config/input
"""
import argparse
import os

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn.functional as F

from config.eval_commavq import DatasetConstants, ModelConfig, PathsConfig
from model import GPT, GPTConfig


def main(ckpt_path=None, onnx_path=None):
    # System configuration

    parser = argparse.ArgumentParser(
        description="Evaluate a pre-trained GPT model on the commavq dataset."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the checkpoint file. If not provided, the default checkpoint path from configuration is used.",
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
    data_config = DatasetConstants()

    torch.manual_seed(system_config.SEED + system_config.seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn

    # Load pre-trained GPT model
    if args.ckpt_path is None:
        ckpt_path = os.path.join(
            os.path.dirname(__file__), paths_config.out_dir, "ckpt.pt"
        )
    elif not args.ckpt_path.endswith(".pt"):
        print("Error: Invalid checkpoint file. The file must have a .pt extension.")
        return
    else:
        ckpt_path = args.ckpt_path
        print("Checkpoint Path updated to the Input Checkpoint")

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

    model = get_eval_model(ckpt_path=ckpt_path, device=system_config.device)

    print(f"Loading model {ckpt_path}")
    print(f"Using device {system_config.device}")

    if onnx_path is None:
        onnx_path = os.path.join(
            os.path.dirname(__file__), paths_config.out_dir, "ckpt.onnx"
        )

    ## Dummy Input in the Eval.py script.
    dummy_input = torch.ones(
        (data_config.BS, GPTConfig.block_size), dtype=torch.int64, device="cuda"
    )
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={
            # dict value: manually named axes
            "x": {0: "my_custom_axis_name"},
            # list value: automatic names
            "y": [0],
        },
    )
    # check if onnx output close to torch
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    sess_options = onnxruntime.SessionOptions()
    provider_cuda = "CUDAExecutionProvider"
    ort_session = onnxruntime.InferenceSession(
        onnx_path, sess_options, providers=[provider_cuda]
    )

    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().detach().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    torch_out, _ = model(dummy_input)
    np.testing.assert_allclose(
        torch_out.cpu().detach().numpy(), ort_outs[0], rtol=1e-03, atol=1e-03
    )
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def get_eval_model(ckpt_path, device):
    """
    Initialize and load a pre-trained GPT model.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        device (str): Device to use for the model ('cpu', 'cuda', 'cuda:0', etc.).

    Returns:
        GPT: Loaded GPT model.
    """
    # Model configuration and loading
    init_config = GPTConfig()
    model_args = initialize_model_args(init_config)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    model_args = update_model_args(model_args, checkpoint_model_args)
    # Create and load the model
    gptconfig = GPTConfig(**model_args)
    model = GPT(gptconfig)
    state_dict = checkpoint["model"]
    state_dict = fix_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def initialize_model_args(init_config):
    """
    Initialize the model configuration arguments by updating them from a checkpoint.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        init_config (GPTConfig): Initial GPT configuration.

    Returns:
        dict: Updated model configuration arguments.
    """

    model_args = dict(
        n_layer=init_config.n_layer,
        n_head=init_config.n_head,
        n_embd=init_config.n_embd,
        block_size=init_config.block_size,
        bias=init_config.bias,
        vocab_size=None,
        dropout=init_config.dropout,
    )
    return model_args


def update_model_args(model_args, checkpoint_model_args):
    """
    Update the model configuration arguments by updating them from a checkpoint model arguments

    Args:

        init_config (GPTConfig): Initial GPT configuration.
        checkpoint_config (GPTConfig): Checkpoint GPT configuration

    Returns:
        dict: Updated model configuration arguments.
    """
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    return model_args


def fix_state_dict_keys(state_dict):
    """
    Fix the keys of the state dictionary.

    Args:
        state_dict (dict): State dictionary of the model.

    Returns:
        dict: State dictionary with fixed keys.
    """

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    return state_dict


if __name__ == "__main__":
    main()
