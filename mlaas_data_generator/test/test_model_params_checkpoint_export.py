import json

import numpy as np

from mlaas_data_generator.federated.model_params import write_final_model_parameters


class FakePretrainedModel:
    def __init__(self):
        self.calls = []

    def state_dict(self):
        return {"dense.weight": np.asarray([[1.0, 2.0]], dtype=np.float32)}

    def save_pretrained(self, path, safe_serialization=True):
        self.calls.append((path, safe_serialization))
        with open(f"{path}/model.safetensors", "w", encoding="utf-8") as handle:
            handle.write("fake")


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def save_pretrained(self, path):
        self.calls.append(path)
        with open(f"{path}/tokenizer.json", "w", encoding="utf-8") as handle:
            handle.write("{}")


class FakeCore:
    def __init__(self):
        self.model = FakePretrainedModel()
        self.tokenizer = FakeTokenizer()


class FakeAdapter:
    def __init__(self):
        self.core = FakeCore()


def test_hf_like_model_exports_save_pretrained_checkpoint(tmp_path):
    adapter = FakeAdapter()

    path = write_final_model_parameters(
        output_dir=tmp_path,
        run_id="run_1",
        model_role="global",
        model_id="global",
        round_idx=2,
        model_type="hf_finetune",
        task_type="classification",
        model=adapter,
        config={"learning_rate": 5e-5, "optimizer": "adamw"},
        metadata={"case_name": "case_a"},
    )

    assert path is not None
    checkpoint_dir = tmp_path / "run_1" / "global"
    assert path == str(checkpoint_dir)
    assert (checkpoint_dir / "model.safetensors").exists()
    assert (checkpoint_dir / "tokenizer.json").exists()
    assert adapter.core.model.calls == [(str(checkpoint_dir), True)]

    metadata = json.loads((checkpoint_dir / "_mlaas_metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifact_type"] == "huggingface_pretrained"
    assert metadata["saved_components"] == ["model", "tokenizer"]
    assert metadata["training_parameters"]["learning_rate"] == 5e-5


def test_non_hf_payload_still_exports_json(tmp_path):
    path = write_final_model_parameters(
        output_dir=tmp_path,
        run_id="run_2",
        model_role="global",
        model_id="global",
        round_idx=1,
        model_type="mlp",
        task_type="classification",
        payload={"dense.weight": np.asarray([[1.0, 2.0]], dtype=np.float32)},
        config={"learning_rate": 0.01},
    )

    assert path == str(tmp_path / "run_2" / "global.json")
    document = json.loads((tmp_path / "run_2" / "global.json").read_text(encoding="utf-8"))
    assert document["parameter_source"] == "payload"
    assert document["summary"]["total_tensors"] == 1
