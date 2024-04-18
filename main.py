import argparse
import logging

from simpletransformers.language_modeling import LanguageModelingModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "num_train_epochs": 10,
    "save_eval_checkpoints": True,
    "block_size": 256,
    "max_seq_length": 256,
    # "save_model_every_epoch": False,
    "learning_rate": 1e-4,
    "train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "mlm": False,
    "dataset_type": "simple",
    "logging_steps": 100,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 3000,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": True,
    "sliding_window": True,
    #"use_multiprocessing": False,
    "vocab_size": 10000,
    "output_dir": f"outputs/from_scratch_",
    "best_model_dir": f"outputs/from_scratch/best_model",
    "fp16": False,
    "local_rank": -1,
    "handle_chinese_chars": True,
#   "tokenizer_name": "bert-base-chinese",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="Local rank. Necessary for using the torch.distributed.launch utility.",
)
args = parser.parse_args()

train_args["local_rank"] = args.local_rank

train_file = f"train.txt"
test_file = f"test.txt"

model = LanguageModelingModel(
    "openai-gpt",
    None,
    args=train_args,
    train_files=train_file,
    use_cuda=False
)

model.train_model(
    train_file,
    eval_file=test_file,
)

model.eval_model(test_file)