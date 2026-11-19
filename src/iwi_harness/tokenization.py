"""Public token lookup using the selected target's own tokenizer."""


def tokenize_target(target, prompt=None, texts=None, offset=0, limit=32):
    """Page through a rendered prompt and look up short independent answer spellings."""
    if prompt is None and texts is None:
        raise ValueError("supply a prompt or short literal texts")
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError("offset must be nonnegative and limit must be between 1 and 64")
    result = {}
    tokenizer = getattr(target, "tokenizer", None)
    if prompt is not None:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 65536:
            raise ValueError("prompt must be nonblank and at most 65536 characters")
        ids = target.encode(prompt)
        if not 1 <= len(ids) <= 2048:
            raise ValueError("expected 1 to 2048 encoded prompt tokens")
        page = ids[offset:offset + limit]
        to_pieces = getattr(tokenizer, "convert_ids_to_tokens", None)
        pieces = to_pieces(page) if callable(to_pieces) else [target.decode([i]) for i in page]
        result["rendered_prompt"] = {
            "length": len(ids), "next_offset": offset + limit if offset + limit < len(ids) else None,
            "tokens": [{"position": index, "input_token_id": token, "piece": piece}
                       for index, (token, piece) in enumerate(zip(page, pieces, strict=True), start=offset)],
        }
    if texts is not None:
        if not isinstance(texts, list) or not 1 <= len(texts) <= 8:
            raise ValueError("supply 1 to 8 literal texts")
        encode = getattr(target, "encode_text", None)
        if not callable(encode):
            if tokenizer is None:
                raise ValueError("this target adapter must provide encode_text for literal text lookup")
            def encode(text):
                return tokenizer.encode(text, add_special_tokens=False)
        spellings = []
        for text in texts:
            if not isinstance(text, str) or not 1 <= len(text) <= 512:
                raise ValueError("literal text must have 1 to 512 characters")
            ids = encode(text)
            if len(ids) > 64:
                raise ValueError("each literal text may encode to at most 64 tokens")
            spellings.append({"text": text, "token_ids": ids})
        result["literal_texts"] = spellings
    return result
