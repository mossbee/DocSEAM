"""Answer a question about an ordered list of page images."""

import argparse
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(model_path, device):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": device}
    )
    processor.tokenizer.padding_side = "left"
    return model.eval(), processor


def prepare_inputs(processor, image_paths, question):
    pages = [
        {"type": "image", "image": f"file://{Path(path).resolve()}",
         "min_pixels": 32768, "max_pixels": 524288}
        for path in image_paths
    ]
    parts = question.split("<image>")
    if len(parts) == 1:
        content = [*pages, {"type": "text", "text": question}]
    else:
        if len(parts) - 1 != len(pages):
            raise ValueError("Each <image> placeholder requires one page image.")
        content = []
        for index, part in enumerate(parts):
            if part:
                content.append({"type": "text", "text": part})
            if index < len(pages):
                content.append(pages[index])

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    images, _ = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size)
    return processor(text=[prompt], images=images, padding=True, return_tensors="pt")


def decode_answer(processor, tokens):
    text = processor.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for marker in ("<|im_end|>", "<|im_start|>", "\nuser\n", "\nassistant\n", "\nUser\n", "\nAssistant\n"):
        if marker in text:
            return text.split(marker, 1)[0].rstrip("\n")
    return text.rstrip("\n")


@torch.inference_mode()
def answer(model, processor, image_paths, question, max_new_tokens=2048):
    inputs = prepare_inputs(processor, image_paths, question).to(model.device)
    tokenizer = processor.tokenizer
    tokens = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )
    return decode_answer(processor, tokens[0, inputs.input_ids.shape[1]:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local DocSEAM folder or Hugging Face model ID")
    parser.add_argument("--images", nargs="+", required=True, help="Page images in document order")
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", default=2048, type=int)
    args = parser.parse_args()
    model, processor = load_model(args.model, args.device)
    print(answer(model, processor, args.images, args.question, args.max_new_tokens))
