# MLaaS Dataset Generator

Framework for generating MLaaS service datasets for selection, composition, reliability, and performance experiments.

The project simulates Machine Learning as a Service (MLaaS) providers by running configurable local, federated, Hugging Face, and generic model workloads. Runs are written to a SQLite database with normalized run, round, client, and measurement tables. CSV and XLSX files are used as command inputs, generated manifests, and analysis exports.

## Current Capabilities

- Federated simulation with client-level participation, dropout, aggregation, and per-round measurements.
- Local datasets: `mnist`, `fashion_mnist`, `cifar10`, `digits`, `iris`, `wine`, `california_housing`, and `diabetes`.
- Hugging Face text, image, and multimodal task support through manifest rows.
- Registry-driven manifest builder for compatible model/dataset/task combinations.
- Generic manifest cases for Keras image classification, sklearn image classification, tabular regression, and clustering.
- IID, quantity skew, Dirichlet label skew, shard, label-per-client, and custom split strategies.
- SQLite output in `outputs/federated.db`, plus manifest result logs and optional CSV exports.
- Runtime metrics for quality, resource use, latency, federated dynamics, perturbation, trust, and explainability.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

Run commands from the repository root.

## CLI Entry Point

Use the package module directly:

```bash
python -m mlaas_data_generator.cli.main <command> [options]
```

Available commands:

| Command | Purpose |
| --- | --- |
| `generate` | Run one configured dataset/model experiment. |
| `wizard` | Interactive helper for building a JSON config. |
| `merge` | Merge CSV outputs into one CSV. |
| `autogen` | Generate many local/generic study runs. |
| `hf-manifest` | Build a registry-driven HF/generic run manifest. |
| `run-manifest` | Execute manifest rows from CSV or XLSX. |
| `evaluate-dynamics` | Inspect federated learning dynamics in a SQLite DB. |

## Recommended Workflow: Manifest Builder

The manifest builder is the main workflow for the current project state. It reads the curated registries in:

- `mlaas_data_generator/registry/models.py`
- `mlaas_data_generator/registry/datasets.py`

It scores compatible model/dataset pairs, samples workload settings, and writes a manifest that can be reviewed, edited, dry-run, and executed.

### Build a Small Test Manifest

```bash
python -m mlaas_data_generator.cli.main hf-manifest ^
  --manifest-profile test ^
  --task-keys text_classification,image_classification,tabular_regression,clustering ^
  --total-runs 8 ^
  --avg-sample-size 128 ^
  --output outputs/run_manifest.xlsx
```

PowerShell also accepts backticks for line continuation. On macOS/Linux, replace `^` with `\`.

For broader MLaaS catalogs, prefer explicit coverage over `--total-runs` balancing:

```bash
python -m mlaas_data_generator.cli.main hf-manifest ^
  --task-keys text_classification,token_classification,image_classification ^
  --models-per-task 20 ^
  --datasets-per-model 1 ^
  --service-variants-per-pair 3 ^
  --dataset-split-variants-per-pair 1 ^
  --max-models-per-family 2 ^
  --manifest-profile balanced ^
  --output outputs/run_manifest.xlsx
```

### Validate a Manifest Without Running Models

```bash
python -m mlaas_data_generator.cli.main run-manifest ^
  --file outputs/run_manifest.xlsx ^
  --sheet runs ^
  --dry_run
```

Dry runs resolve defaults, normalize column names, validate required fields, and write a status file to `outputs/run_manifest_results.csv`.

### Execute a Manifest

```bash
python -m mlaas_data_generator.cli.main run-manifest ^
  --file outputs/run_manifest.xlsx ^
  --sheet runs
```

Successful runs are written to `outputs/federated.db`. Row-level success, skipped, or failed statuses are written to `outputs/run_manifest_results.csv`. Runtime failures are appended to `outputs/run_failures.log`.

## Manifest Builder Options

| Option | Meaning |
| --- | --- |
| `--task-keys` | Comma-separated registry task keys to include. If omitted, all supported HF and generic tasks are considered. |
| `--models-per-task` | Maximum models selected per requested HF task. |
| `--max-models-per-family` | Optional cap on how many selected models may come from the same base family within a task. |
| `--datasets-per-model` | Maximum compatible datasets selected per model. |
| `--run-regimes` | Comma-separated run regimes, usually `finetune_transfer`, `inference_only`, or both. |
| `--variants-per-pair` / `--service-variants-per-pair` | Number of sampled service variants per selected model/dataset pair. These variants keep the same base model and dataset, but vary training and workload knobs such as learning rate, optimizer, clients, and rounds. |
| `--dataset-split-variants-per-pair` | Number of dataset split variants emitted per selected model/dataset pair. Keep this at `1` if you want services to differ only by training setup. |
| `--total-runs` | Exact number of rows to emit, balanced across eligible requested task keys. |
| `--manifest-profile` | Workload profile: `test`, `balanced`, or `benchmark`. |
| `--avg-sample-size` | Target average `max_samples` across emitted rows. |
| `--input-json` | Optional Hugging Face audit metadata JSON for enrichment only. |
| `--output` | Destination `.xlsx` or `.csv` path. |
| `--sheet` | XLSX sheet name, default `runs`. |
| `--seed` | Reproducible sampling seed. |

Profiles:

| Profile | Use Case |
| --- | --- |
| `test` | Small, quick smoke manifests with low sample counts and short timeouts. |
| `balanced` | Default medium workload. |
| `benchmark` | Larger sample sizes, more clients/rounds, and longer timeouts. |

## Supported Manifest Task Keys

Hugging Face registry tasks:

```text
text_classification
token_classification
sentence_similarity
fill_mask
text_generation
text2text_generation
image_classification
object_detection
image_segmentation
image_captioning
text_image_retrieval
visual_question_answering
```

Generic manifest tasks:

```text
keras_image_classification
sklearn_image_classification
tabular_regression
clustering
```

## Manifest Columns

The builder emits all columns expected by `run-manifest`. The most useful columns to review or edit are:

| Column | Purpose |
| --- | --- |
| `enabled` | Set false to skip a row. Missing values default to enabled. |
| `external_run_id` | Stable run ID. If present, this becomes the database `run_id`. |
| `run_group_id` | Groups related manifest rows. Missing values are filled at runtime. |
| `case_name` | Human-readable model/dataset/regime label. |
| `dataset` | Dataset source, for example `hf`, `cifar10`, or `synthetic`. |
| `model_type` | Runner model family, for example `hf`, `hf_finetune`, `cnn`, `mlp`, `randomforest`, or `kmeans`. |
| `task_type` | Canonical task family such as `classification`, `regression`, `generation`, `detection`, `segmentation`, `retrieval`, `vqa`, or `clustering`. |
| `hf_task` | HF adapter task such as `sequence_classification`, `image_classification`, `causal_lm_generation`, or `visual_question_answering`. |
| `modality` | `text`, `image`, `multimodal`, or `tabular`. |
| `hf_model_id` | Hugging Face model repo ID. |
| `dataset_name` / `dataset_config` | Hugging Face dataset repo and config. |
| `train_split` / `test_split` | HF split names or slice expressions. |
| `text_column`, `image_column`, `label_column`, `mask_column` | Dataset schema mapping. |
| `task_tag` | Metric subtype such as `summarization`, `language-modeling`, `captioning`, `retrieval`, or `vqa`. |
| `run_regime` | `finetune_transfer`, `inference_only`, or `generic`. |
| `num_rounds`, `num_clients`, `local_epochs` | Federated workload size. |
| `batch_size`, `learning_rate`, `optimizer`, `weight_decay`, `momentum` | Training knobs. |
| `distribution`, `dirichlet_alpha`, `sample_size`, `max_samples` | Data partitioning and truncation knobs. |
| `timeout_s`, `device`, `mixed_precision`, `num_workers` | Runtime controls. |

For `inference_only` rows, training-only knobs such as `local_epochs`, `learning_rate`, `optimizer`, `weight_decay`, `momentum`, and `dirichlet_alpha` are emitted as `N/A`. `batch_size` remains numeric because the HF adapter still uses it for inference/evaluation batching.
For HF inference-only runs, `sample_size` is the per-client evaluation partition target. The runner partitions the resolved eval split, and `eval_sequence_count` is the actual number of examples forwarded through the model. If `sample_size * num_clients` exceeds the loaded eval rows, the run summary marks `split.resampled_with_replacement=true`.

CSV manifests can include a row where `external_run_id` is `defaults`; XLSX manifests can include a `defaults` sheet. Those values are applied before row-specific values.

## Manual Manifest Rows

You can create or edit a manifest manually. A minimal Hugging Face text classification row needs:

```csv
external_run_id,enabled,dataset,model_type,task_type,hf_task,hf_model_id,dataset_name,dataset_config,train_split,test_split,text_column,label_column,num_rounds,num_clients,local_epochs,batch_size,learning_rate,distribution,max_samples
demo_sst2,true,hf,hf_finetune,classification,sequence_classification,distilbert-base-uncased,glue,sst2,train,validation,sentence,label,1,2,1,8,0.00002,iid,128
```

Then run:

```bash
python -m mlaas_data_generator.cli.main run-manifest --file path/to/manifest.csv --dry_run
python -m mlaas_data_generator.cli.main run-manifest --file path/to/manifest.csv
```

## Generation and Multimodal Notes

Generation task mapping:

| Pipeline Tag | Manifest `hf_task` |
| --- | --- |
| `text-generation` | `causal_lm_generation` |
| `text2text-generation` | `seq2seq_generation` |

`task_tag` selects canonical generation metrics:

| `task_tag` | Eval Metrics |
| --- | --- |
| `language-modeling` | `loss`, `perplexity` |
| `summarization` | `rouge1`, `rouge2`, `rougeL` |
| `translation` | `sacrebleu` |
| `captioning` | `cider`, `bleu` |

For multimodal HF rows, set `modality=multimodal`, provide `image_column` and `text_column`, and optionally set `missing_pair_handling` to `drop` or `error`. The loader validates image/text pair integrity per split.

## Single-Run Examples

The legacy `generate` command is still useful for small local runs:

```bash
python -m mlaas_data_generator.cli.main generate ^
  --clients 5 ^
  --rounds 2 ^
  --dataset fashion_mnist ^
  --strategy iid ^
  --model-type CNN ^
  --output clients.csv
```

Distribution examples:

```bash
python -m mlaas_data_generator.cli.main generate --clients 5 --strategy quantity_skew --distribution-param 0.3 --output qskew.csv
python -m mlaas_data_generator.cli.main generate --clients 10 --strategy dirichlet --distribution-param 0.2 --output dirichlet.csv
python -m mlaas_data_generator.cli.main generate --clients 5 --strategy shard --distribution-param 2 --output shard.csv
python -m mlaas_data_generator.cli.main generate --clients 5 --strategy label_per_client --distribution-param 2 --output klabels.csv
python -m mlaas_data_generator.cli.main generate --clients 5 --strategy custom --distribution custom_distributions.json --output custom.csv
```

The supported split strategies are:

| Strategy | Meaning |
| --- | --- |
| `iid` | Independent and identically distributed split. |
| `quantity_skew` | Dirichlet allocation over client sample counts. |
| `dirichlet` | Dirichlet allocation over label distributions. |
| `shard` | Label-sorted shards assigned to clients. |
| `label_per_client` | Fixed number of labels per client. |
| `custom` | Per-client label counts from JSON. |

## Autogenerated Local Studies

`autogen` samples many local/generic experiments across classification, regression, and clustering:

```bash
python -m mlaas_data_generator.cli.main autogen ^
  --runs 50 ^
  --task-split 50,30,20 ^
  --seed 123
```

It writes per-run table exports under `outputs/` and a JSON study manifest at `outputs/study_manifest.json`.

## Merging CSV Outputs

```bash
python -m mlaas_data_generator.cli.main merge outputs/runs/*.csv --output merged.csv --dedupe
```

Merged files are written under `outputs/merged/`.

## Evaluating Federated Dynamics

```bash
python -m mlaas_data_generator.cli.main evaluate-dynamics ^
  --db outputs/federated.db ^
  --json
```

Add `--run-id <id>` to inspect one run.

## Output Layout

| Path | Contents |
| --- | --- |
| `outputs/federated.db` | Primary SQLite database for generated runs. |
| `outputs/run_manifest.xlsx` | Default generated manifest path. |
| `outputs/run_manifest_results.csv` | Per-manifest-row success, skipped, or failed status. |
| `outputs/run_failures.log` | Validation and runtime failure details. |
| `outputs/runs/` | CSV outputs from single-run/export workflows. |
| `outputs/merged/` | Merged CSV files. |
| `weights/` | Optional saved model weight JSON files. |

Set `MLAAS_OUTDIR` to redirect `outputs/runs/` and `outputs/merged/`. The default SQLite path remains `outputs/federated.db` unless the run config sets `db_path`.

## Tests

Run the test suite with:

```bash
python -m pytest mlaas_data_generator/test
```

Manifest-specific checks:

```bash
python -m pytest mlaas_data_generator/test/test_hf_manifest_builder_selection.py mlaas_data_generator/test/test_manifest_task_matrix.py
```

## Notes for Extending the Project

- Add HF models in `mlaas_data_generator/registry/models.py`.
- Add HF datasets in `mlaas_data_generator/registry/datasets.py`.
- The manifest builder only emits rows that pass compatibility validation for task, modality, required columns, and run regime.
- For high-risk multimodal pairs such as image captioning and image/text retrieval, registry entries must be explicitly marked as manifest validated.
- New metrics should be seeded in `mlaas_data_generator/storage/writer.py` before being written broadly.
