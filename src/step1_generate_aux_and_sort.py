import argparse
import os
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util


def clean_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\-\*\d\.\)\s]+", "", text)
    text = text.strip().strip('"').strip("'")
    return text


def parse_sentences(response: str, max_chars: int = 77):
    sentences = []

    for line in response.split("\n"):
        line = clean_sentence(line)

        if not line:
            continue
        if len(line.split()) < 3:
            continue
        if len(line) > max_chars:
            continue
        if line.lower().startswith(("sure", "here", "below", "okay", "these are")):
            continue

        sentences.append(line)

    return sentences


def compute_similarity(anchor, candidates, sim_model):
    if len(candidates) == 0:
        return []

    anchor_emb = sim_model.encode(anchor, normalize_embeddings=True)
    cand_emb = sim_model.encode(candidates, normalize_embeddings=True)
    scores = util.cos_sim(anchor_emb, cand_emb)[0]

    return [
        {
            "sentence": sent,
            "similarity": float(score),
            "char_len": len(sent),
        }
        for sent, score in zip(candidates, scores)
    ]


def sort_prompts_by_similarity(prompts, semantic_concept, sim_model):
    scored = compute_similarity(
        anchor=semantic_concept,
        candidates=prompts,
        sim_model=sim_model,
    )
    scored = sorted(scored, key=lambda x: x["similarity"], reverse=True)
    return [item["sentence"] for item in scored], scored


def generate_filtered_prompts(
    model,
    tokenizer,
    sim_model,
    semantic_concept,
    mode,
    batch_generate,
    max_new_tokens,
    final_num,
    max_chars,
):
    def generate_sentences(
        model,
        tokenizer,
        semantic_concept,
        mode,
        num_generate=5,
        max_new_tokens=512,
        max_chars=77,
    ):
        def build_instruction(semantic_concept: str, num_sentences: int, mode: str):
            focus = f"""
- The generated prompts should strongly contain or imply {semantic_concept}.
- Use concrete visual descriptions.
"""
            return f"""
Generate {num_sentences} image-generation prompts based on the following semantic concept.

Semantic concept:
{semantic_concept}

Requirements:
{focus}
- Each prompt must be shorter than {max_chars} characters.
- Use short, simple, visually descriptive sentences.
- Do not exceed {max_chars} characters per line.
- Avoid explanations, reasoning, factual statements, or conversational text.
- Output only the generated prompts.
- Put each prompt on a separate line.
"""

        instruction = build_instruction(
            semantic_concept=semantic_concept,
            num_sentences=num_generate,
            mode=mode,
        )

        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": instruction,
            },
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.05,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"\n[Generated Response]\n{response}\n")

        return parse_sentences(response, max_chars=max_chars)

    selected_prompts = []
    debug_items = []
    seen = set()

    round_idx = 0
    max_trials = final_num * 20

    while len(selected_prompts) < final_num and round_idx < max_trials:
        round_idx += 1

        candidates = generate_sentences(
            model=model,
            tokenizer=tokenizer,
            semantic_concept=semantic_concept,
            mode=mode,
            num_generate=batch_generate,
            max_new_tokens=max_new_tokens,
            max_chars=max_chars,
        )

        unique_candidates = []
        for sent in candidates:
            key = sent.lower().strip()
            if key not in seen and len(sent) <= max_chars:
                unique_candidates.append(sent)

        if len(unique_candidates) == 0:
            print(f"[WARN] Round {round_idx}: no valid candidates under {max_chars} chars.")
            continue

        scored = compute_similarity(
            anchor=semantic_concept,
            candidates=unique_candidates,
            sim_model=sim_model,
        )

        scored = sorted(scored, key=lambda x: x["similarity"], reverse=True)
        debug_items.extend(scored)

        best_item = scored[0]
        best_sentence = best_item["sentence"]

        selected_prompts.append(best_sentence)
        seen.add(best_sentence.lower().strip())

        print(
            f"[SELECT] {len(selected_prompts)}/{final_num} | "
            f"round={round_idx} | "
            f"sim={best_item['similarity']:.4f} | "
            f"len={len(best_sentence)} | "
            f"{best_sentence}"
        )

    selected_prompts, selected_sorted_debug = sort_prompts_by_similarity(
        prompts=selected_prompts,
        semantic_concept=semantic_concept,
        sim_model=sim_model,
    )

    debug_items = sorted(debug_items, key=lambda x: x["similarity"], reverse=True)

    return selected_prompts, {
        "all_candidates_sorted": debug_items,
        "selected_prompts_sorted": selected_sorted_debug,
    }


def safe_makedirs_for_file(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def pad_prompts_to_same_length(short_prompts, long_prompts, semantic_concept, sim_model):
    if len(short_prompts) == 0:
        return short_prompts

    diff = len(long_prompts) - len(short_prompts)

    if diff <= 0:
        return short_prompts

    scored = compute_similarity(
        anchor=semantic_concept,
        candidates=short_prompts,
        sim_model=sim_model,
    )
    scored = sorted(scored, key=lambda x: x["similarity"], reverse=True)

    top_prompts = [item["sentence"] for item in scored]
    padded = list(short_prompts)

    for i in range(diff):
        padded.append(top_prompts[i % len(top_prompts)])

    return padded


def main(args):
    print(f"STEP 1. Loading generation model: {args.gen_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.gen_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.gen_model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    print(f"(1.2) Similarity model: {args.sim_model_name}")
    sim_model = SentenceTransformer(args.sim_model_name)

    print(f"Prompt               : {args.prompt}")
    print(f"Biased semantic      : {args.biased_semantic}")
    print(f"Bias-absent semantic : {args.bias_absent_semantic}")
    print(f"Aux num              : {args.aux_num}")
    print(f"Batch generate       : {args.batch_generate}")
    print(f"Task                 : {args.task}")
    print(f"Max chars            : {args.max_chars}")

    biased_debug = {}

    if args.task != "concept":
        biased_prompts, biased_debug = generate_filtered_prompts(
            model=model,
            tokenizer=tokenizer,
            sim_model=sim_model,
            semantic_concept=args.biased_semantic,
            mode="biased",
            batch_generate=args.batch_generate,
            max_new_tokens=args.max_new_tokens,
            final_num=args.aux_num,
            max_chars=args.max_chars,
        )
    else:
        biased_prompts = []

    bias_absent_prompts, bias_absent_debug = generate_filtered_prompts(
        model=model,
        tokenizer=tokenizer,
        sim_model=sim_model,
        semantic_concept=args.bias_absent_semantic,
        mode="bias_absent",
        batch_generate=args.batch_generate,
        max_new_tokens=args.max_new_tokens,
        final_num=args.aux_num,
        max_chars=args.max_chars,
    )

    if args.task == "concept":
        biased_prompts = [
            f"{e} in {args.biased_semantic} style"
            for e in bias_absent_prompts
            if len(f"{e} in {args.biased_semantic} style") <= args.max_chars
        ]

        biased_prompts, biased_sorted_debug = sort_prompts_by_similarity(
            prompts=biased_prompts,
            semantic_concept=args.biased_semantic,
            sim_model=sim_model,
        )

    else:
        if len(biased_prompts) > len(bias_absent_prompts):
            bias_absent_prompts = pad_prompts_to_same_length(
                short_prompts=bias_absent_prompts,
                long_prompts=biased_prompts,
                semantic_concept=args.bias_absent_semantic,
                sim_model=sim_model,
            )

        elif len(bias_absent_prompts) > len(biased_prompts):
            biased_prompts = pad_prompts_to_same_length(
                short_prompts=biased_prompts,
                long_prompts=bias_absent_prompts,
                semantic_concept=args.biased_semantic,
                sim_model=sim_model,
            )

        biased_prompts = [p for p in biased_prompts if len(p) <= args.max_chars]
        bias_absent_prompts = [p for p in bias_absent_prompts if len(p) <= args.max_chars]

        biased_prompts, biased_sorted_debug = sort_prompts_by_similarity(
            prompts=biased_prompts,
            semantic_concept=args.biased_semantic,
            sim_model=sim_model,
        )

        bias_absent_prompts, bias_absent_sorted_debug = sort_prompts_by_similarity(
            prompts=bias_absent_prompts,
            semantic_concept=args.bias_absent_semantic,
            sim_model=sim_model,
        )

    if args.task == "concept":
        bias_absent_prompts, bias_absent_sorted_debug = sort_prompts_by_similarity(
            prompts=bias_absent_prompts,
            semantic_concept=args.bias_absent_semantic,
            sim_model=sim_model,
        )

    final_json = [
        {
            "id": args.id,
            "prompt": args.prompt,
            "base_entity_biased_prompts": biased_prompts,
            "base_entity_bias_absent_prompts": bias_absent_prompts,
            "num_biased_prompts": len(biased_prompts),
            "num_bias_absent_prompts": len(bias_absent_prompts),
            "max_chars": args.max_chars,
        }
    ]

    save_path = args.output_file
    safe_makedirs_for_file(save_path)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    print(json.dumps(final_json, indent=2, ensure_ascii=False))
    print(f"Saved final json to: {save_path}")

    if args.debug_output_path is not None:
        os.makedirs(args.debug_output_path, exist_ok=True)

        save_prompt = args.prompt.replace(" ", "_")
        debug_path = os.path.join(
            args.debug_output_path,
            f"{save_prompt}_debug.json",
        )

        debug_json = {
            "biased_debug": biased_debug,
            "bias_absent_debug": bias_absent_debug,
            "biased_sorted_debug": biased_sorted_debug,
            "bias_absent_sorted_debug": bias_absent_sorted_debug,
        }

        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_json, f, indent=2, ensure_ascii=False)

        print(f"Saved debug json to: {debug_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--biased_semantic", type=str, required=True)
    parser.add_argument("--bias_absent_semantic", type=str, required=True)

    parser.add_argument("--aux_num", type=int, default=5)
    parser.add_argument("--batch_generate", type=int, default=5)

    parser.add_argument(
        "--gen_model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--sim_model_name",
        type=str,
        default="BAAI/bge-large-en-v1.5",
    )

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_chars", type=int, default=77)

    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--debug_output_path", type=str, default=None)
    parser.add_argument("--task", type=str, default="object")

    args = parser.parse_args()
    main(args)