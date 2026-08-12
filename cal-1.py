import json
import numpy as np

from scipy.special import softmax

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    roc_curve
)

from collections import Counter


score_file = (
    "./infer_eer/"
    "temp_checkpoint-15865.json"
)


label_file = (
    "./data/"
    "asv_test_fake_audio.jsonl"
)



# ============================
# Load model logits
# ============================

with open(score_file, "r") as f:
    score_data = json.load(f)



scores = {}

for wav, value in score_data.items():

    # According to infer_eer_8.py:
    #
    # logits_target =
    # word_probs[:, [52317,12768]]
    #
    # 52317 -> Fake
    # 12768 -> Real
    #
    # Therefore:
    #
    # value[0] = Fake logit
    # value[1] = Real logit


    fake_logit = value[0]
    real_logit = value[1]


    # P(fake)
    prob_fake = softmax(
        [
            fake_logit,
            real_logit
        ]
    )[0]


    scores[wav] = prob_fake



print(
    "Loaded scores:",
    len(scores)
)



# ============================
# Load labels
# ============================


labels = []
pred_scores = []


with open(label_file, "r") as f:

    for line in f:

        item = json.loads(line)


        audio_path = (
            item["messages"][0]["audio"]
        )


        answer = (
            item["messages"][1]["content"]
        )


        if answer.strip() == "Fake.":

            label = 1

        elif answer.strip() == "Real.":

            label = 0

        else:

            continue



        if audio_path in scores:

            labels.append(label)

            pred_scores.append(
                scores[audio_path]
            )



labels = np.array(labels)

pred_scores = np.array(pred_scores)



print(
    "Matched samples:",
    len(labels)
)



print(
    "Label distribution:",
    Counter(labels)
)



# ============================
# Sanity check
# ============================


fake_scores = []

real_scores = []


for i in range(len(labels)):

    if labels[i] == 1:

        fake_scores.append(
            pred_scores[i]
        )

    else:

        real_scores.append(
            pred_scores[i]
        )


print(
    "Fake mean score:",
    np.mean(fake_scores)
)


print(
    "Real mean score:",
    np.mean(real_scores)
)



# ============================
# AUC
# ============================


auc = roc_auc_score(
    labels,
    pred_scores
)


fpr, tpr, thresholds = roc_curve(
    labels,
    pred_scores
)


fnr = 1 - tpr


eer_index = np.nanargmin(
    np.abs(fnr-fpr)
)


eer = (
    fpr[eer_index]
    +
    fnr[eer_index]
) / 2


eer_threshold = thresholds[eer_index]




pred_labels = (
    pred_scores >= eer_threshold
).astype(int)


acc = accuracy_score(
    labels,
    pred_labels
)


print("\n============================")

print(
    f"EER : {eer*100:.3f}%"
)

print(
    f"AUC : {auc*100:.3f}%"
)

print(
    f"ACC : {acc*100:.3f}%"
)

print(
    f"Threshold : {eer_threshold:.6f}"
)

print("============================")