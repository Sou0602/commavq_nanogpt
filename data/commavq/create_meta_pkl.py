import argparse
import os
import pickle


def save_meta_data(n_heads):
    """
    Serialize and save metadata to a pickle file based on the number of heads.

    Args:
        n_heads (int): Number of heads (6 or 8).

    Raises:
        AssertionError: If n_heads is not 6 or 8.

    Notes:
        The 'vocab_size' in the metadata is determined based on the number of heads.

    """
    assert n_heads == 8 or n_heads == 6, "Number of heads must be 6 or 8."

    meta = {"vocab_size": 1048 if n_heads == 8 else 1026}
    # Rounding 1026 to 1048 to work with the Triton compiler for the number of heads

    pickle_file_path = os.path.join(os.path.dirname(__file__), "meta.pkl")

    # Check if the file already exists and delete it
    if os.path.exists(pickle_file_path):
        os.remove(pickle_file_path)

    # Serialize and save the dictionary to a pickle file
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(meta, pickle_file)


def main():
    """
    Main function to handle command-line arguments and save metadata.

    Command Line Arguments:
        --n_heads (int): Number of heads (6 or 8).

    """
    parser = argparse.ArgumentParser(
        description="Save meta data with a specified number of heads."
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        required=True,
        choices=[6, 8],
        help="Number of heads (6 or 8)",
    )
    args = parser.parse_args()

    save_meta_data(args.n_heads)


if __name__ == "__main__":
    main()
