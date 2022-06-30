import sys

import logging
import numpy as np
import os
import random
import sys
import time
import torch

from argparse import Namespace
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)

import tvm
from tvm import relay

# Setup warnings
import warnings

warnings.filterwarnings(action="ignore", category=DeprecationWarning, module=r".*")
warnings.filterwarnings(action="default", module=r"torch.quantization")

# Setup logging level to WARN. Change it accordingly
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.WARN,
)

# logging.getLogger("transformers.modeling_utils").setLevel(
#    logging.WARN)  # Reduce logging

print(torch.__version__)


configs = Namespace()

# The output directory for the fine-tuned model, $OUT_DIR.
configs.output_dir = "./MRPC/"

# The data directory for the MRPC task in the GLUE benchmark, $GLUE_DIR/$TASK_NAME.
configs.data_dir = "./glue_data/MRPC"

# The model name or path for the pre-trained model.
configs.model_name_or_path = "bert-base-uncased"
# The maximum length of an input sequence
configs.max_seq_length = 128

# Prepare GLUE task.
configs.task_name = "MRPC".lower()
configs.processor = processors[configs.task_name]()
configs.output_mode = output_modes[configs.task_name]
configs.label_list = configs.processor.get_labels()
configs.model_type = "bert".lower()
configs.do_lower_case = True

# Set the device, batch size, topology, and caching flags.
configs.device = "cpu"
configs.eval_batch_size = 1
configs.n_gpu = 0
configs.local_rank = -1
configs.overwrite_cache = False


# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)


# load model
model = BertForSequenceClassification.from_pretrained(configs.output_dir, return_dict=False)
model.to(configs.device)

# quantize model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# print(quantized_model)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / (1024 * 1024))
    os.remove("temp.p")


print_size_of_model(model)
print_size_of_model(quantized_model)


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (
        ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    )
    eval_outputs_dirs = (
        (args.output_dir, f"{args.output_dir}-MM")
        if args.task_name == "mnli"
        else (args.output_dir,)
    )


    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=True
        )

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)

        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info(f"***** Running evaluation {prefix} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results |= result

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info(f"***** Eval results {prefix} *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the othersv will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        f'cached_{"dev" if evaluate else "train"}_{list(filter(None, args.model_name_or_path.split("/"))).pop()}_{str(args.max_seq_length)}_{str(task)}',
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir)
            if evaluate
            else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    return TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )


def time_model_evaluation(model, configs, tokenizer):
    eval_start_time = time.time()
    result = evaluate(configs, model, tokenizer, prefix="")
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print(result)
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))


def perf_bench_torch(pt_model, inp, n_repeat):
    inputs = {
        "input_ids": inp[0],
        "attention_mask": inp[1],
        "labels": inp[2],
        "token_type_ids": inp[3]
    }

    with torch.no_grad():
        pt_model.eval()

        for _ in range(3):
            pt_model(**inputs)

        t1 = time.time()
        for _ in range(n_repeat):
            pt_model(**inputs)
        t2 = time.time()

        elapsed = (t2 - t1) * 1e3 / n_repeat
        print("Torch elapsed ms:", elapsed)

        return elapsed


# define the tokenizer
tokenizer = BertTokenizer.from_pretrained(
    configs.output_dir, do_lower_case=configs.do_lower_case
)

quantized_output_dir = f"{configs.output_dir}quantized/"
if not os.path.exists(quantized_output_dir):
    os.makedirs(quantized_output_dir)
    quantized_model.save_pretrained(quantized_output_dir)


batch_size = configs.eval_batch_size
inputs = (torch.ones(batch_size, 128, dtype=torch.int64),
          torch.ones(batch_size, 128, dtype=torch.int64),
          torch.ones(batch_size, 128, dtype=torch.int64))

input_shapes = [("input_ids", (inputs[0].shape, "int64")),
                ("attention_mask", (inputs[1].shape, "int64")),
                ("token_type_ids", (inputs[2].shape, "int64"))]

script_module = torch.jit.trace(quantized_model, inputs).eval()
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

# modify below for older cpus
target = "llvm -mcpu=cascadelake"
# target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    # opt_mod, opt_params = relay.optimize(mod, target="llvm -mcpu=cascadelake -libs=mkl", params=params)
    # print(opt_mod["main"])
    lib = relay.build(mod, target=target, params=params)
    # graph, libs, params = relay.build(mod, target=target, params=params)

# # from tvm.contrib.debugger import debug_runtime

# # runtime = debug_runtime.create(graph, libs, tvm.cpu(0), "dump")
runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cpu(0)))

runtime.set_input("input_ids", inputs[0].numpy())
runtime.set_input("attention_mask", inputs[1].numpy())
runtime.set_input("token_type_ids", inputs[2].numpy())

runtime.run()

n_repeat = 100

print("Running TVM time evaluator")
ftimer = runtime.module.time_evaluator("run", tvm.cpu(0), number=1, repeat=n_repeat)
prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
print(prof_res)
print("TVM elapsed ms mean, median and std:", np.mean(prof_res), np.median(prof_res), np.std(prof_res))

print("Running PyTorch benchmark")
inputs = (torch.ones(batch_size, 128, dtype=torch.int64),
          torch.ones(batch_size, 128, dtype=torch.int64),
          torch.ones(1, dtype=torch.int64),
          torch.ones(batch_size, 128, dtype=torch.int64))
perf_bench_torch(quantized_model, inputs, n_repeat)


def evaluate_tvm(args, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (
        ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    )
    eval_outputs_dirs = (
        (args.output_dir, f"{args.output_dir}-MM")
        if args.task_name == "mnli"
        else (args.output_dir,)
    )


    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=True
        )

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        # Eval!
        logger.info(f"***** Running evaluation {prefix} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        # eval_loss = 0.0
        # nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.detach().cpu().numpy() for t in batch)

            runtime.set_input("input_ids", batch[0])
            runtime.set_input("attention_mask", batch[1])
            runtime.set_input("token_type_ids", batch[2])

            runtime.run()

            logits = np.reshape(runtime.get_output(0).asnumpy(), (-1, 2))
            if preds is None:
                preds = logits
                # print(preds.shape)
                out_label_ids = batch[3]
            else:
                preds = np.append(preds, logits, axis=0)
                out_label_ids = np.append(out_label_ids, batch[3], axis=0)

        # print(preds.shap)
        # eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        # print(preds)
        # print(out_label_ids)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results |= result

        output_eval_file = os.path.join(eval_output_dir, f"{prefix}_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info(f"***** Eval results {prefix} *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def time_tvm_model_evaluation():
    eval_start_time = time.time()
    result = evaluate_tvm(configs, prefix="tvm")
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print(result)
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))


time_tvm_model_evaluation()

# # Evaluate the original FP32 BERT model
# print("Evaluating PyTorch full precision accuracy and performance:")
# time_model_evaluation(model, configs, tokenizer)

# Evaluate the INT8 BERT model after the dynamic quantization
print("Evaluating PyTorch quantization accuracy and performance:")
time_model_evaluation(quantized_model, configs, tokenizer)
