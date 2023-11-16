import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from glossary import normalize_word
from random import choice
from random import choices
from random import sample
import numpy as np


def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0


def path2rest(path, split, annotations, label2ans):
    iid = int(path.split("/")[-1].split("_")[-1][:-4])

    _annot = annotations[split][iid]
    _annot = list(_annot.items())
    qids, qas = [a[0] for a in _annot], [a[1] for a in _annot]

    for qid, qa in zip(qids, qas):
        question = qa[0]
        answer = qa[1] if "test" not in split else None
        
        if answer is not None:
            answer_labels = answer["labels"]
            answer_scores = answer["scores"]
            answers = [label2ans[l] for l in answer_labels]
        else:
            answer_labels = []
            answer_scores = []
            answers = []

        yield [[question], [answers], [answer_labels], [answer_scores], iid, [qid], split]


def make_arrow(root, dataset_root):
    with open(f"{root}/v2_OpenEnded_mscoco_train2014_questions.json", "r") as fp:
        questions_train2014 = json.load(fp)["questions"]
    with open(f"{root}/v2_OpenEnded_mscoco_val2014_questions.json", "r") as fp:
        questions_val2014 = json.load(fp)["questions"]
    # with open(f"{root}/v2_OpenEnded_mscoco_test2015_questions.json", "r") as fp:
    #     questions_test2015 = json.load(fp)["questions"]
    # with open(f"{root}/v2_OpenEnded_mscoco_test-dev2015_questions.json", "r") as fp:
    #     questions_test_dev2015 = json.load(fp)["questions"]

    with open(f"{root}/v2_mscoco_train2014_annotations.json", "r") as fp:
        annotations_train2014 = json.load(fp)["annotations"]
    with open(f"{root}/v2_mscoco_val2014_annotations.json", "r") as fp:
        annotations_val2014 = json.load(fp)["annotations"]

    annotations = dict()

    for split, questions in zip(
        ["train", "val"],
        # ["train", "val", "test", "test-dev"],
        [
            questions_train2014,
            questions_val2014,
            # questions_test2015,
            # questions_test_dev2015,
        ],
    ):
        _annot = defaultdict(dict)
        for q in tqdm(questions):
            _annot[q["image_id"]][q["question_id"]] = [q["question"]]

        annotations[split] = _annot

    all_major_answers = list()

    for split, annots in zip(
        ["train", "val"], [annotations_train2014, annotations_val2014],
    ):
        _annot = annotations[split]
        for q in tqdm(annots):
            all_major_answers.append(q["multiple_choice_answer"])
    
    all_major_answers = [normalize_word(word) for word in tqdm(all_major_answers)]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
    sorted_answers = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}

    with open('major_answers.json', 'w') as fp:
        json.dump(sorted_answers, fp)

    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())

    for split, annots in zip(
        ["train", "val"], [annotations_train2014, annotations_val2014],
    ):
        _annot = annotations[split]
        print(f'Get answer_count in split {split}\n')
        for q in tqdm(annots):
            answers = q["answers"]
            answer_count = {}
            for answer in answers:
                answer_ = answer["answer"]
                answer_count[answer_] = answer_count.get(answer_, 0) + 1

            labels = []
            scores = []
            for answer in answer_count:
                if answer not in ans2label:
                    continue
                labels.append(ans2label[answer])
                score = get_score(answer_count[answer])
                scores.append(score)

            _annot[q["image_id"]][q["question_id"]].append(
                {"labels": labels, "scores": scores,}
            )

    for split in ["train", "val"]:
        filtered_annot = dict()
        for ik, iv in annotations[split].items():
            new_q = dict()
            for qk, qv in iv.items():
                if len(qv[1]["labels"]) != 0:
                    new_q[qk] = qv
            if len(new_q) != 0:
                filtered_annot[ik] = new_q
        annotations[split] = filtered_annot

    for split in [
        "train",
        "val",
        # "test",
        # "test-dev",
    ]:
        annot = annotations[split]
        split_name = {
            "train": "train2014",
            "val": "val2014",
            # "test": "test2015",
            # "test-dev": "test2015",
        }[split]
        paths = list(glob(f"{root}/{split_name}/*.jpg"))
        random.shuffle(paths)
        annot_paths = [
            path
            for path in paths
            if int(path.split("/")[-1].split("_")[-1][:-4]) in annot
        ]

        if len(paths) == len(annot_paths):
            print("all images have caption annotations")
        else:
            print("not all images have caption annotations")
        print(
            len(paths), len(annot_paths), len(annot),
        )

      
        bs = [entry for path in tqdm(annot_paths) for entry in path2rest(path, split, annotations, label2ans)]

        df_initial = pd.DataFrame(
            bs,
            columns=[
                "questions",
                "answers",
                "answer_labels",
                "answer_scores",
                "image_id",
                "question_id",
                "split",
            ],
        )
        df_initial.to_csv(f'../{split}_initial_dataframe.csv', index=False)

        if split == 'val':
            df_test_initial = df_initial[-1000:]
            df_test_initial.to_csv(f'../test_initial_dataframe.csv', index=False)

        # table = pa.Table.from_pandas(df_initial)

        # os.makedirs(dataset_root, exist_ok=True)
        # with pa.OSFile(f"{dataset_root}/vqav2_{split}.arrow", "wb") as sink:
        #     with pa.RecordBatchFileWriter(sink, table.schema) as writer:
        #         writer.write_table(table)


    # table = pa.ipc.RecordBatchFileReader(
    #     pa.memory_map(f"{dataset_root}/vqav2_val.arrow", "r")
    # ).read_all()

    # pdtable = table.to_pandas()


    # df1 = pdtable[:-1000]
    # df2 = pdtable[-1000:]

    # df1 = pa.Table.from_pandas(df1)
    # df2 = pa.Table.from_pandas(df2)

    # with pa.OSFile(f"{dataset_root}/vqav2_trainable_val.arrow", "wb") as sink:
    #     with pa.RecordBatchFileWriter(sink, df1.schema) as writer:
    #         writer.write_table(df1)

    # with pa.OSFile(f"{dataset_root}/vqav2_rest_val.arrow", "wb") as sink:
    #     with pa.RecordBatchFileWriter(sink, df2.schema) as writer:
    #         writer.write_table(df2)



def make_vqa_image(data_path):
    df_initial = pd.read_csv(data_path)
    unique_image_ids = df_initial['image_id'].unique()

    def get_neg_image_ids(image_ids, k=9):
        return np.array([sample(list(set(unique_image_ids) - {x}), k) for x in image_ids])

    df_initial['best_answer_idx'] = df_initial['answer_scores'].apply(lambda x: np.argmax(x[0]))
    df_initial['best_answer'] = df_initial.apply(lambda row: row['answers'][0][row['best_answer_idx']], axis=1)

    df_initial['sentence'] = df_initial['questions'].str[0] + ' ' + df_initial['best_answer']

    # Adding multiple neg_image_id columns
    neg_image_ids_matrix = get_neg_image_ids(df_initial['image_id'].values)
    for i in range(9):
        df_initial[f'neg_image_id_{i+1}'] = neg_image_ids_matrix[:, i]

    df_final = df_initial[['sentence', 'image_id'] + [f'neg_image_id_{i+1}' for i in range(9)]]
    df_final.columns = ['sentence', 'pos_image_id'] + [f'neg_image_id_{i+1}' for i in range(9)]

    df_final.to_csv(f'.../data/vqa_image.csv', index=False)




def make_vqa_text(data_path, all_major_answers, output):
    df_initial = pd.read_csv(data_path)

    def get_neg_answers(all_major_answers, k=9):
        filtered_answers = [ans for ans in all_major_answers if ans.lower() not in ["yes", "no"]]
        return [sample(filtered_answers, k) for _ in range(len(df_initial))]

    df_initial['best_answer_idx'] = df_initial['answer_scores'].apply(lambda x: np.argmax(eval(x)[0]))
    df_initial['best_answer'] = df_initial.apply(lambda row: eval(row['answers'])[0][row['best_answer_idx']], axis=1)

    df_initial['sentence'] = df_initial['questions'].apply(lambda x: eval(x)[0]) + ' ' + df_initial['best_answer']

    neg_answers_matrix = get_neg_answers(all_major_answers)
    for i in range(9):
        df_initial[f'neg_answer_{i+1}'] = [neg_answers[i] for neg_answers in neg_answers_matrix]

    for i in range(9):
        df_initial[f'neg_sentence_{i+1}'] = df_initial['questions'].apply(lambda x: eval(x)[0]) + ' ' + df_initial[f'neg_answer_{i+1}']

    df_final = df_initial[['sentence', 'image_id'] + [f'neg_sentence_{i+1}' for i in range(9)]]
    df_final.columns = ['sentence', 'pos_image_id'] + [f'neg_sentence_{i+1}' for i in range(9)]

    df_final.to_csv(output, index=False)



def main():
    # root = '../vqav2'
    # arrows_root = '../vqav2'
    # make_arrow(root, arrows_root)

    with open('major_answers.json', 'r') as fp:
        sorted_filtered_answers = json.load(fp)
    all_major_answers = list(sorted_filtered_answers.keys())

    # make_vqa_text('../data/train_initial_dataframe.csv', all_major_answers, output = '../data/vqa_text_train.csv')
    # make_vqa_text('../data/val_initial_dataframe.csv', all_major_answers, output = '../data/vqa_text_val.csv')
    make_vqa_text('../data/test_initial_dataframe.csv', all_major_answers, output = '../data/vqa_text_test.csv')

if __name__=="__main__":
    main()