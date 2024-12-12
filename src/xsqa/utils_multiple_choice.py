# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """


import csv
import glob
import json
import random
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from tqdm.auto import tqdm

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
from transformers.tokenization_utils_base import TruncationStrategy


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    val = "val"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MultipleChoiceDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            task: str,
            overwrite_cache=False,
            mode: Split = Split.train,
            num_choices=None,
            train_file=None,
            val_file=None,
            test_file=None,
            percentage=None,
        ):
            processor = processors[task]()
            if mode.value == "train":
                PREFIX = train_file.split("/")[-1].replace(".jsonl", "")
            elif mode.value == "val":
                PREFIX = val_file.split("/")[-1].replace(".jsonl", "")
            elif mode.value == "test":
                PREFIX = test_file.split("/")[-1].replace(".jsonl", "")
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}".format(
                    PREFIX
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    logger.info(f"Num Choices {num_choices}")
                    label_list = processor.get_labels(num_choices=num_choices)
                    if mode == Split.val:
                        examples = processor.get_val_examples(val_file, num_choices=num_choices)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(test_file, num_choices=num_choices)
                    else:
                        examples = processor.get_train_examples(train_file, num_choices=num_choices, percentage=percentage)
                    logger.info("Num examples: %s", len(examples))
                    self.features = convert_examples_to_features(
                        examples,
                        label_list
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

 
class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, file_path, num_choices):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_val_examples(self, file_path, num_choices):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, file_path, num_choices):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self, num_choices):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

def convert_ABC_to_123(s):
    assert len(s) == 1 and s in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return ord(s)-65
    

class XCSRProcessor(DataProcessor):
    """Processor for the XCSQA data set."""

    def get_train_examples(self, file_path, num_choices=5, percentage=100):
        """See base class."""
        logger.info("LOOKING train AT {}".format(file_path))
        return self._create_examples(self._read_json(file_path), "train", num_choices=num_choices, percentage=percentage)

    def get_val_examples(self, file_path, num_choices=5):
        """See base class."""
        logger.info("LOOKING val AT {} ".format(file_path))
        return self._create_examples(self._read_json(file_path), "val", num_choices=num_choices)

    def get_test_examples(self, file_path, num_choices):
        """See base class."""
        logger.info("LOOKING test AT {} ".format(file_path))
        return self._create_examples(self._read_json(file_path), "test", num_choices=num_choices)

    def get_labels(self, num_choices):
        """See base class."""
        return [str(i) for i in range(1,num_choices+1)]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type, num_choices, percentage=None):
        """Creates examples for the training and validation sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in [str(i) for i in range(1,num_choices+1)]:
                return int(truth)
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []  
        # we deleted example which has more than or less than four choices
        if percentage is not None:
            random.seed(42)
            random.shuffle(lines)
            split_point = int(len(lines) * percentage / 100)
            lines = lines[:split_point]
        print(len(lines))
        for line in tqdm(lines, desc="read data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) != num_choices:
                print(num_choices)
                print(len(data_raw["question"]["choices"]))
            assert len(data_raw["question"]["choices"]) == num_choices
            truth = convert_ABC_to_123(data_raw["answerKey"])
            assert truth != "None" or type=="test"
            question_choices = data_raw["question"]
            question = question_choices["stem"] # TODO: debug for only choices.
            # question = "This is a question."    # debug for only choices.
            id = data_raw["id"]
            options = question_choices["choices"]
            #if len(options) == num_choices:
            examples.append(
                InputExample(
                    example_id=id,
                    question=question,
                    contexts=[" " for i in range(num_choices) ],
                    endings=[options[i]["text"] for i in range(num_choices)],
                    label=truth,
                )
            ) 
        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))  

        return examples
"""
def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
) -> List[InputFeatures]:
    
    Loads a data file into a list of `InputFeatures`
    

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), total=len(examples), desc="convert examples to features"):
        # if ex_index % 100 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = ""
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            feature += f"{example.question}{ending}##"
        features.append((feature, example.label))
    return features
"""
def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """


    features = []
    for (ex_index, example) in tqdm(enumerate(examples), total=len(examples), desc="convert examples to features"):
        # if ex_index % 100 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = {'prompts': [], 'choices': []}
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            feature['prompts'].append(example.question)
            feature['choices'].append(ending)
        features.append((feature, example.label))
    return features

class CollatorXSQA:
    def __init__(
        self,
        tokenizer,
        num_labels,
        max_length=512,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        **tokenizer_kwargs
    ):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.num_labels = num_labels
        if self.tokenizer_kwargs.get("max_length", None) is None:
            self.tokenizer_kwargs["max_length"] = max_length
        if self.tokenizer_kwargs.get("padding", None) is None:
            self.tokenizer_kwargs["padding"] = padding
        if self.tokenizer_kwargs.get("truncation", None) is None:
            self.tokenizer_kwargs["truncation"] = truncation
        if self.tokenizer_kwargs.get("add_special_tokens", None) is None:
            self.tokenizer_kwargs["add_special_tokens"] = add_special_tokens
    
    def __call__(self, batch):
        prompts_flattened =[]
        for elem in batch:
            prompts_flattened += elem[0]['prompts'] 
        choices_flattened = []
        for elem in batch:
            choices_flattened += elem[0]['choices'] 
        # labels = [elem.get(["label"], None) for elem in batch if elem.get(["label"], None) is not None]

        tokenized = self.tokenizer(prompts_flattened,choices_flattened, return_tensors="pt", **self.tokenizer_kwargs)
        batch_size = len(prompts_flattened) // self.num_labels
        # labels = torch.tensor(labels, dtype=torch.long)
        input = {k: v.view(batch_size, self.num_labels, -1) for k, v in tokenized.items()}
        labels = [elem[1] for elem in batch]
        return input, labels
processors = {"xcsr": XCSRProcessor}
