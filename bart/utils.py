import os

from torch.utils.data import Dataset

import csv

class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, data_dir="./cnn_dm/", type_path="train", block_size=1024):
        super(SummarizationDataset,).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []

        print("loading " + type_path + " source.")

        with open(os.path.join(data_dir, type_path + ".source"), "r") as f:
            for text in f.readlines():  # each text is a line and a full story
                tokenized = tokenizer.batch_encode_plus(
                    [text], max_length=block_size, pad_to_max_length=True, return_tensors="pt"
                )
                self.source.append(tokenized)
            f.close()

        print("loading " + type_path + " target.")

        with open(os.path.join(data_dir, type_path + ".target"), "r") as f:
            for text in f.readlines():  # each text is a line and a summary
                tokenized = tokenizer.batch_encode_plus(
                    [text], max_length=56, pad_to_max_length=True, return_tensors="pt"
                )
                self.target.append(tokenized)
            f.close()

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()

        src_mask = self.source[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}


class SpaceDataset(Dataset):
    def __init__(self, tokenizer, data_dir="./space/", type_path="train", block_size=1024):
        super(SummarizationDataset,).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []

        print("loading " + type_path +".csv" )

        with open(os.path.join(data_dir, type_path + ".csv") ,encoding='utf-8') as f:
            reader = csv.DictReader(f)
            title = reader.fieldnames
            for index, row in enumerate(reader):
                if row["abstract"]=='' or row["title"]=='':
                    # print(row["abstract"], row["title"])
                    continue
                abstract_tokenized = tokenizer.batch_encode_plus(
                    [row["abstract"]], max_length=block_size, pad_to_max_length=True, return_tensors="pt"
                )
                self.source.append(abstract_tokenized)

                title_tokenized = tokenizer.batch_encode_plus(
                    [row["title"]], max_length=block_size, pad_to_max_length=True, return_tensors="pt"
                )
                self.target.append(title_tokenized)
            f.close()

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()

        src_mask = self.source[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}


def format_summary(translation):
    """ Transforms the output of the `from_batch` function
    into nicely formatted summaries.
    """
    raw_summary, _, _ = translation
    summary = (
        raw_summary.replace("[unused0]", "")
        .replace("[unused3]", "")
        .replace("[PAD]", "")
        .replace("[unused1]", "")
        .replace(r" +", " ")
        .replace(" [unused2] ", ". ")
        .replace("[unused2]", "")
        .strip()
    )

    return summary


def format_rouge_scores(scores):
    return """\n
****** ROUGE SCORES ******

** ROUGE 1
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}

** ROUGE 2
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}

** ROUGE L
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}""".format(
        scores["rouge-1"]["f"],
        scores["rouge-1"]["p"],
        scores["rouge-1"]["r"],
        scores["rouge-2"]["f"],
        scores["rouge-2"]["p"],
        scores["rouge-2"]["r"],
        scores["rouge-l"]["f"],
        scores["rouge-l"]["p"],
        scores["rouge-l"]["r"],
    )


def save_rouge_scores(str_scores):
    with open("../output/space/bart/rouge_scores.txt", "w") as output:
        output.write(str_scores)

def compute_rouge():
    import rouge
    rouge_evaluator = rouge.Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=4,
            length_limit_type="words",
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True,
        )
    generated_summaries = open("../output/space/bart/test_predictions.txt").readlines()
    reference_summaries = open("../output/space/bart/test_targets.txt").readlines()
    scores = rouge_evaluator.get_scores(generated_summaries, reference_summaries)
    str_scores = format_rouge_scores(scores)
    save_rouge_scores(str_scores)
    print(str_scores)


if __name__ == "__main__":
    compute_rouge()
