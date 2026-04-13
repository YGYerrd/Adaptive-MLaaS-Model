import contextlib
import numpy as np

from mlaas_data_generator.models.adapters.hf_core import HFCore
from mlaas_data_generator.models.adapters.hf_task import CausalLMGenerationSpec


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.ndim = self.value.ndim
        self.shape = self.value.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value)

    def item(self):
        return self.value.item()

    def sum(self):
        return FakeTensor(np.asarray(self.value.sum()))

    def __ne__(self, other):
        return FakeTensor(self.value != other)


class FakeTorch:
    long = "long"

    @staticmethod
    def tensor(value, dtype=None, device=None):
        return FakeTensor(value)

    @staticmethod
    @contextlib.contextmanager
    def no_grad():
        yield


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    padding_side = "right"


class DummyGenerationModel:
    def __init__(self):
        self.forward_calls = 0
        self.config = type("Cfg", (), {"is_encoder_decoder": False})()

    def eval(self):
        return self

    def generate(self, **kwargs):
        batch = kwargs["input_ids"].numpy()
        generated = []
        for row in batch:
            prompt_tokens = [tok for tok in row.tolist() if tok != 0]
            generated.append(prompt_tokens + [7, 8])
        max_len = max(len(row) for row in generated)
        padded = [row + [0] * (max_len - len(row)) for row in generated]
        return FakeTensor(np.asarray(padded, dtype=np.int64))

    def __call__(self, **kwargs):
        self.forward_calls += 1
        labels = kwargs["labels"].numpy()
        logits = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.float32)
        return type("Out", (), {"logits": FakeTensor(logits), "loss": FakeTensor(np.asarray(0.5, dtype=np.float32))})


class DummyGenerationSpec:
    name = "seq2seq_generation"
    supports_generation = True

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        enc = {
            "input_ids": torch.tensor(xb["input_ids"], dtype=torch.long, device=device),
            "attention_mask": torch.tensor(xb["attention_mask"], dtype=torch.long, device=device),
        }
        labels = None
        if yb is not None and not inference_only:
            labels = torch.tensor(yb, dtype=torch.long, device=device)
        return enc, labels, {"ignore_index": int(ignore_index)}

    def build_forward_inputs(self, enc, labels_t=None, inference_only=False):
        out = dict(enc)
        if labels_t is not None and not inference_only:
            out["labels"] = labels_t
        return out

    def generate_predictions(self, model, enc, tokenizer, torch, generation_config):
        generated = model.generate(**enc, **generation_config)
        in_len = enc["input_ids"].shape[1]
        return FakeTensor(generated.numpy()[:, in_len:])

    def extract_loss(self, torch, outputs, logits, labels_t, extra):
        return outputs.loss

    def batch_metric_statistics(self, torch, logits, labels_t, extra):
        return None

    def batch_metric_statistics_from_outputs(self, torch, outputs, labels_t, extra):
        return None

    def metrics_from_statistics(self, stats):
        return None

    def metrics(self, y_true, y_pred, y_extra=None):
        common = min(y_true.shape[-1], y_pred.shape[-1])
        score = float((y_true[:, :common] == y_pred[:, :common]).mean())
        return {"primary": score, "secondary": 0.25, "named_metrics": {"token_accuracy": score}}


def test_causal_lm_inference_only_strips_supervised_suffix_from_prompt_tokens():
    spec = CausalLMGenerationSpec()
    fake_torch = FakeTorch()
    xb = {
        "input_ids": np.asarray([[11, 12, 21, 22, 99]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1, 1, 1, 1]], dtype=np.int64),
    }
    yb = np.asarray([[-100, -100, 21, 22, 99]], dtype=np.int64)

    enc, labels_t, extra = spec.encode_batch(
        DummyTokenizer(),
        xb,
        yb,
        max_length=5,
        torch=fake_torch,
        device="cpu",
        inference_only=True,
    )

    assert enc["input_ids"].numpy().tolist() == [[11, 12]]
    assert enc["attention_mask"].numpy().tolist() == [[1, 1]]
    assert labels_t.numpy().tolist() == yb.tolist()
    assert extra["ignore_index"] == -100


def test_hfcore_eval_inference_only_generation_uses_teacher_forced_labels_for_metrics_and_loss():
    core = HFCore.__new__(HFCore)
    core.torch = FakeTorch()
    core.task_spec = DummyGenerationSpec()
    core.tokenizer = DummyTokenizer()
    core.model = DummyGenerationModel()
    core.generation_config = {}
    core.batch_size = 2
    core.device = "cpu"
    core.label_pad_value = -100
    core.max_length = 4
    core.model_id = "dummy"
    core.weight_format = None
    core.task_tag = None
    core.tokenizer_load_s = 0.0
    core.model_load_s = 0.0
    core.tokenizer_cache_hit = True
    core.model_cache_hit = True

    xs = {
        "input_ids": np.asarray([[5, 6], [7, 8]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1], [1, 1]], dtype=np.int64),
    }
    ys = np.asarray([[7, 8], [7, 8]], dtype=np.int64)

    loss, primary, secondary, qos = core.eval(xs, ys, inference_only=True)

    assert np.isclose(loss, 0.5)
    assert np.isclose(primary, 1.0)
    assert np.isclose(secondary, 0.25)
    assert qos["eval_supervised_token_count"] == 4
    assert qos["tokens_total"] == 4
    assert core.model.forward_calls == 1
    assert core.tokenizer.padding_side == "left"


def test_causal_lm_encode_batch_left_pads_dict_inputs_even_without_labels():
    spec = CausalLMGenerationSpec()
    fake_torch = FakeTorch()
    tok = DummyTokenizer()
    xb = {
        "input_ids": np.asarray([[10, 11, 0, 0], [20, 21, 22, 0]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.int64),
    }

    enc, labels_t, _ = spec.encode_batch(
        tok,
        xb,
        None,
        max_length=4,
        torch=fake_torch,
        device="cpu",
        inference_only=True,
    )

    assert tok.padding_side == "left"
    assert enc["input_ids"].numpy().tolist() == [[0, 0, 10, 11], [0, 20, 21, 22]]
    assert enc["attention_mask"].numpy().tolist() == [[0, 0, 1, 1], [0, 1, 1, 1]]
    assert labels_t is None
