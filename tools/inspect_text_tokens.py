#!/usr/bin/env python3
"""[debug-only] Inspect how HunyuanVideo tokenizes a prompt.

For each text path used during generation (the MLLM / LLaMA encoder and the CLIP
encoder), this prints/saves, per sequence position:

    index | token_id | decoded_token | attention_mask

so you can see which positions are real tokens and which are padding before the
text features enter the diffusion transformer.

It uses the *exact* tokenizers and prompt templates the model uses at inference
(via hyvideo.text_encoder.load_tokenizer + hyvideo.constants.PROMPT_TEMPLATE) and
loads only the tokenizers — not the multi-GB encoder weights.

The MLLM path mirrors the pipeline: the user prompt is wrapped in the video
prompt template and tokenized to (text_len + crop_start). The first `crop_start`
positions are template/system tokens that are cropped off before the DiT; the
remaining `text_len` positions (mask=1 then padding) are what the transformer
sees. `crop_start` is reported in the output so you can locate that boundary.

Usage
-----
    python tools/inspect_text_tokens.py \
        --prompt "A red car drives past a white fence" \
        --outdir debug_outputs

    # if the checkpoints are in ../ckpts (so tokenizers resolve):
    python tools/inspect_text_tokens.py --model-base ../ckpts \
        --prompt "A red car drives past a white fence" --outdir debug_outputs

Outputs debug_outputs/token_inspection_<slug>.json and .txt. Easy to remove:
delete this file.
"""

import argparse
import json
import os
import re
import sys


def slugify(text, max_words=3):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return "_".join(words[:max_words]) if words else "prompt"


def build_token_rows(tokenizer, input_ids, attention_mask):
    """One dict per position: index, token_id, decoded_token, attention_mask."""
    ids = input_ids.tolist()
    mask = attention_mask.tolist()
    # convert_ids_to_tokens keeps subword boundaries (preferred over decode()).
    pieces = tokenizer.convert_ids_to_tokens(ids)
    rows = []
    for i, (tid, piece, m) in enumerate(zip(ids, pieces, mask)):
        rows.append(
            {
                "index": i,
                "token_id": int(tid),
                "decoded_token": piece,
                # decode() of the single id, in case the merged form is clearer.
                "decoded_token_text": tokenizer.decode([tid]),
                "attention_mask": int(m),
            }
        )
    return rows


def inspect_encoder(name, tokenizer_type, text, template, max_length, crop_start):
    """Tokenize `text` exactly as the pipeline does and return an encoder record."""
    from hyvideo.text_encoder import load_tokenizer

    tokenizer, tokenizer_path = load_tokenizer(tokenizer_type=tokenizer_type)

    # Apply the prompt template for the MLLM path (CLIP has no template).
    tokenized_text = template.format(text) if template else text

    batch = tokenizer(
        tokenized_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0]

    record = {
        "name": name,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_path": str(tokenizer_path),
        "uses_prompt_template": template is not None,
        "crop_start": int(crop_start),  # positions removed before the DiT (MLLM)
        "max_length": int(max_length),
        "input_ids_shape": list(batch["input_ids"].shape),
        "attention_mask_shape": list(batch["attention_mask"].shape),
        "num_real_tokens": int(attention_mask.sum().item()),
        "tokens": build_token_rows(tokenizer, input_ids, attention_mask),
    }
    return record


def write_txt(path, prompt, encoders):
    lines = []
    lines.append(f"prompt: {prompt}")
    lines.append("")
    for enc in encoders:
        lines.append("=" * 72)
        lines.append(f"encoder: {enc['name']}")
        lines.append(f"tokenizer_class: {enc['tokenizer_class']}")
        lines.append(f"tokenizer_path: {enc['tokenizer_path']}")
        lines.append(f"uses_prompt_template: {enc['uses_prompt_template']}")
        lines.append(f"crop_start (template tokens removed before DiT): {enc['crop_start']}")
        lines.append(f"max_length: {enc['max_length']}")
        lines.append(f"input_ids_shape: {enc['input_ids_shape']}")
        lines.append(f"attention_mask_shape: {enc['attention_mask_shape']}")
        lines.append(f"num_real_tokens (mask==1): {enc['num_real_tokens']}")
        lines.append("-" * 72)
        lines.append(f"{'index':>5} | {'token_id':>8} | {'attn':>4} | decoded_token")
        lines.append("-" * 72)
        for t in enc["tokens"]:
            lines.append(
                f"{t['index']:>5} | {t['token_id']:>8} | "
                f"{t['attention_mask']:>4} | {t['decoded_token']}"
            )
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A red car drives past a white fence",
        help="Prompt to tokenize.",
    )
    parser.add_argument(
        "--outdir", type=str, default="debug_outputs", help="Output directory."
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="Checkpoint root holding text_encoder/ and text_encoder_2/ "
        "(sets MODEL_BASE so tokenizers resolve). Defaults to $MODEL_BASE or ./ckpts.",
    )
    parser.add_argument("--text-len", type=int, default=256, help="MLLM text length.")
    parser.add_argument("--text-len-2", type=int, default=77, help="CLIP text length.")
    parser.add_argument(
        "--prompt-template-video",
        type=str,
        default="dit-llm-encode-video",
        help="Prompt template key used for video generation (data_type=video).",
    )
    args = parser.parse_args()

    # MODEL_BASE must be set before importing hyvideo.constants (read at import).
    if args.model_base is not None:
        os.environ["MODEL_BASE"] = args.model_base

    # Make 'hyvideo' importable when run from anywhere.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from hyvideo.constants import PROMPT_TEMPLATE

    video_tpl = PROMPT_TEMPLATE[args.prompt_template_video]
    crop_start = video_tpl.get("crop_start", 0)

    encoders = []
    # MLLM / LLaMA path: video prompt template, length = text_len + crop_start.
    encoders.append(
        inspect_encoder(
            name="mllm_llm_text_encoder",
            tokenizer_type="llm",
            text=args.prompt,
            template=video_tpl["template"],
            max_length=args.text_len + crop_start,
            crop_start=crop_start,
        )
    )
    # CLIP path: no template, max_length = text_len_2.
    encoders.append(
        inspect_encoder(
            name="clip_text_encoder_2",
            tokenizer_type="clipL",
            text=args.prompt,
            template=None,
            max_length=args.text_len_2,
            crop_start=0,
        )
    )

    result = {"prompt": args.prompt, "encoders": encoders}

    os.makedirs(args.outdir, exist_ok=True)
    slug = slugify(args.prompt)
    json_path = os.path.join(args.outdir, f"token_inspection_{slug}.json")
    txt_path = os.path.join(args.outdir, f"token_inspection_{slug}.txt")

    with open(json_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_txt(txt_path, args.prompt, encoders)

    print(f"prompt: {args.prompt!r}")
    for enc in encoders:
        print(
            f"  [{enc['name']}] {enc['tokenizer_class']} "
            f"max_length={enc['max_length']} real_tokens={enc['num_real_tokens']} "
            f"crop_start={enc['crop_start']}"
        )
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
