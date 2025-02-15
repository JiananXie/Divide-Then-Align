import os
import json
import random

def possible_positive_ctxs(source,item,data_type):
    if source == "nq": 
        if data_type == "train":
            ctx_pool = [ctx for ctx in item["positive_ctxs"] if "has_answer" in ctx and ctx["has_answer"]]
            seen_ids = set()
            unique_ctxs = []
            for ctx in ctx_pool:
                if ctx["id"] not in seen_ids:
                    seen_ids.add(ctx["id"])
                    unique_ctxs.append(ctx)
            return unique_ctxs
        elif data_type == "test":
            seen_ids = set()
            unique_ctxs = []
            for ctx in item["ctxs"]:
                if ctx["id"] not in seen_ids:
                    seen_ids.add(ctx["id"])
                    unique_ctxs.append(ctx)
            if len(unique_ctxs) < 5:
                return []
            top_3 = unique_ctxs[:3]
            remaining = random.sample(unique_ctxs[3:], 2)
            return top_3 + remaining
    elif source == "triviaqa":
        seen_ids = set()
        unique_ctxs = []
        for ctx in item["positive_ctxs"]:
            if ctx["psg_id"] not in seen_ids:
                seen_ids.add(ctx["psg_id"])
                unique_ctxs.append(ctx)
        if len(unique_ctxs) < 5:
            return []
        top_3 = unique_ctxs[:3]
        remaining = random.sample(unique_ctxs[3:], 2)
        return top_3 + remaining
    elif source == "webq":
        seen_ids = set()
        unique_ctxs = []
        for ctx in item["ctxs"]:
            if ctx["id"] not in seen_ids:
                seen_ids.add(ctx["id"])
                unique_ctxs.append(ctx)
        if len(unique_ctxs) < 5:
            return []
        top_3 = unique_ctxs[:3]
        remaining = random.sample(unique_ctxs[3:], 2)
        return top_3 + remaining

def possible_negative_ctxs(source,item,data_type):
    if source == "nq":
        if data_type == "train":
            seen_ids = set()
            unique_ctxs = []
            for ctx in item["hard_negative_ctxs"]:
                if ctx["id"] not in seen_ids:
                    seen_ids.add(ctx["id"])
                    unique_ctxs.append(ctx)
            if len(unique_ctxs) < 5:
                return []
            top_3 = unique_ctxs[:3]
            remaining = random.sample(unique_ctxs[3:], 2)
            return top_3 + remaining
        elif data_type == "test":
             seen_ids = set()
             unique_ctxs = []
             for ctx in item["ctxs"]:
                 if ctx["id"] not in seen_ids:
                     seen_ids.add(ctx["id"])
                     unique_ctxs.append(ctx)
             if len(unique_ctxs) < 5:
                 return []
             bottom_3 = unique_ctxs[-3:]
             remaining = random.sample(unique_ctxs[:-3], 2)
             return bottom_3 + remaining
    elif source == "triviaqa":
        seen_ids = set()
        unique_ctxs = []
        for ctx in item["positive_ctxs"]:
            if ctx["psg_id"] not in seen_ids:
                seen_ids.add(ctx["psg_id"])
                unique_ctxs.append(ctx)
        if len(unique_ctxs) < 5:
            return []
        bottom_3 = unique_ctxs[-3:]
        remaining = random.sample(unique_ctxs[:-3], 2)
        return bottom_3 + remaining
    elif source == "webq":
        seen_ids = set()
        unique_ctxs = []
        for ctx in item["ctxs"]:
            if ctx["id"] not in seen_ids:
                seen_ids.add(ctx["id"])
                unique_ctxs.append(ctx)
        if len(unique_ctxs) < 5:
            return []
        bottom_3 = unique_ctxs[-3:]
        remaining = random.sample(unique_ctxs[:-3], 2)
        return bottom_3 + remaining


def filter_possible_ctxs(data_path,source,data_type):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered_data = []
    for item in data:
        positive_ctxs = possible_positive_ctxs(source, item, data_type)
        negative_ctxs = possible_negative_ctxs(source, item, data_type)
        if len(positive_ctxs) == 0 or len(negative_ctxs) == 0:
            continue
        temp = {
            "question": item["question"],
            "answers": item["answers"],
            "possible_golden_ctxs": positive_ctxs,
            "possible_noisy_ctxs": negative_ctxs,
            "answer": item["answer"]
        }
        filtered_data.append(temp)
    print("filtered data with positive and negative ctxs:",len(filtered_data))
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)