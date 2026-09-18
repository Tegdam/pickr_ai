import hashlib
import os

import pytest

from bench.capture.tokens import QwenTokenizer


class StubHF:
    chat_template = "{% for m in messages %}<|{{m.role}}|>{{m.content}}{% endfor %}<|assistant|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return "".join(f"<|{m['role']}|>{m['content']}" for m in messages) + "<|assistant|>"

    def __call__(self, text):
        class R:
            input_ids = text.split()
        return R()


def test_render_and_count_with_stub():
    tok = QwenTokenizer(StubHF(), model_id="stub", revision="r1")
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello there"}]
    assert tok.render(msgs) == "<|system|>sys<|user|>hello there<|assistant|>"
    assert tok.count("a b c") == 3
    assert tok.model_id == "stub" and tok.revision == "r1"
    assert tok.template_sha256 == hashlib.sha256(StubHF.chat_template.encode()).hexdigest()


@pytest.mark.skipif(not os.environ.get("BENCH_HF_TESTS"), reason="needs the Qwen tokenizer in the HF cache; set BENCH_HF_TESTS=1")
def test_real_qwen_template_has_chatml_markers():
    tok = QwenTokenizer.load()
    text = tok.render([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert text.startswith("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n")
    assert tok.count(text) > 0 and tok.revision
