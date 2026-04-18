-- Client skip/error report.
--
-- To link a row back to outputs/run_manifest_results.csv, filter the manifest
-- by the run_id shown in manifest_join_key/run_id. If that run_id is not in
-- the manifest, the DB and CSV are from different run batches.

WITH metric_rows AS (
    SELECT
        m.run_id,
        m.round,
        m.client_id,
        md.name AS metric_name,
        m.value_bool,
        COALESCE(
            m.value_text,
            CAST(m.value_num AS TEXT),
            CAST(m.value_int AS TEXT),
            CAST(m.value_bool AS TEXT),
            m.value_json
        ) AS value
    FROM measurements m
    JOIN metrics md
      ON md.metric_id = m.metric_id
    WHERE md.name IN (
        'participated_flag',
        'fail_reason',
        'fail_reason_category',
        'perturbation_error'
    )
),

client_status AS (
    SELECT
        run_id,
        round,
        client_id,
        MAX(CASE WHEN metric_name = 'participated_flag' THEN value_bool END) AS participated_flag,
        MAX(CASE WHEN metric_name = 'fail_reason_category' THEN value END) AS fail_reason_category,
        MAX(CASE WHEN metric_name = 'fail_reason' THEN value END) AS fail_reason,
        MAX(CASE WHEN metric_name = 'perturbation_error' THEN value END) AS perturbation_error
    FROM metric_rows
    GROUP BY run_id, round, client_id
),

param_rows AS (
    SELECT
        run_id,
        scope,
        key,
        COALESCE(
            value_text,
            CAST(value_num AS TEXT),
            CAST(value_int AS TEXT),
            CAST(value_bool AS TEXT),
            value_json
        ) AS value
    FROM run_params
),

run_context AS (
    SELECT
        run_id,
        MAX(CASE WHEN key = 'hf_model_id' THEN value END) AS hf_model_id,
        MAX(CASE WHEN key = 'dataset_name' THEN value END) AS dataset_name,
        MAX(CASE WHEN key = 'hf_task' THEN value END) AS hf_task,
        MAX(CASE WHEN key = 'hf_url' THEN value END) AS hf_url,
        MAX(CASE WHEN key = 'train_split' THEN value END) AS train_split,
        MAX(CASE WHEN key = 'test_split' THEN value END) AS test_split,
        MAX(CASE WHEN key = 'hf_service_meta_json' THEN json_extract(value, '$.run_regime') END) AS run_regime,
        MAX(CASE WHEN key = 'hf_service_meta_json' THEN json_extract(value, '$.registry_task') END) AS registry_task,
        MAX(CASE WHEN key = 'hf_service_meta_json' THEN json_extract(value, '$.variant_index') END) AS variant_index,
        MAX(CASE WHEN key = 'hf_service_meta_json' THEN json_extract(value, '$.split_variant_index') END) AS split_variant_index
    FROM param_rows
    GROUP BY run_id
)

SELECT
    r.run_id,
    r.dataset,
    r.task_type,
    r.model_type,
    rc.hf_model_id,
    rc.dataset_name,
    rc.hf_task,
    cs.fail_reason_category,
    COALESCE(
        cs.fail_reason,
        cs.perturbation_error,
        cs.fail_reason_category,
        'skipped/no recorded error'
    ) AS reason_or_error
FROM client_status cs
LEFT JOIN runs r
  ON r.run_id = cs.run_id
LEFT JOIN run_context rc
  ON rc.run_id = cs.run_id
WHERE COALESCE(cs.participated_flag, 1) = 0
ORDER BY r.created_at DESC, cs.run_id, cs.round, cs.client_id;
