
WITH
m AS (
    SELECT
        run_id,
        round,
        client_id,
        metric_name,
        COALESCE(value_num, CAST(value_int AS REAL), CAST(value_bool AS REAL)) AS num_value,
        COALESCE(value_text, CAST(value_num AS TEXT), CAST(value_int AS TEXT), CAST(value_bool AS TEXT), value_json) AS text_value
    FROM v_measurements
),

p AS (
    SELECT
        run_id,
        scope,
        key,
        COALESCE(value_num, CAST(value_int AS REAL), CAST(value_bool AS REAL), CAST(value_text AS REAL)) AS num_value,
        COALESCE(value_text, CAST(value_num AS TEXT), CAST(value_int AS TEXT), CAST(value_bool AS TEXT), value_json) AS text_value
    FROM run_params
),
text_image_retrieval	retrieval		multimodal	jxie/flickr8k		openai/clip-vit-base-patch32

params AS (
    SELECT
        run_id,
        COALESCE(
            MAX(CASE WHEN scope = 'adapter' AND key = 'hf_model_id' THEN text_value END),
            MAX(CASE WHEN scope = 'dataset' AND key = 'hf_model_id' THEN text_value END)
        ) AS hf_model_id,
        MAX(CASE WHEN scope = 'dataset' AND key = 'dataset_name' THEN text_value END) AS dataset_name,
        COALESCE(
            MAX(CASE WHEN scope = 'dataset' AND key = 'hf_downloads' THEN num_value END),
            MAX(CASE WHEN scope = 'dataset' AND key = 'hf_service_meta_json' THEN CAST(json_extract(text_value, '$.downloads') AS REAL) END)
        ) AS downloads,
        COALESCE(
            MAX(CASE WHEN scope = 'dataset' AND key = 'hf_likes' THEN num_value END),
            MAX(CASE WHEN scope = 'dataset' AND key = 'hf_service_meta_json' THEN CAST(json_extract(text_value, '$.likes') AS REAL) END)
        ) AS likes,
        COALESCE(
            MAX(CASE WHEN scope = 'adapter' AND key IN ('lr', 'learning_rate', 'effective_learning_rate', 'learning_rate_adjusted') THEN num_value END),
            MAX(CASE WHEN scope = 'adapter' AND key IN ('requested_lr', 'requested_learning_rate') THEN num_value END),
            MAX(CASE WHEN scope = 'runner' AND key = 'learning_rate' THEN num_value END)
        ) AS learning_rate,
        COALESCE(
            MAX(CASE WHEN scope = 'adapter' AND key IN ('batch_size', 'effective_batch_size', 'requested_batch_size') THEN num_value END),
            MAX(CASE WHEN scope = 'runner' AND key = 'benchmark_batch' THEN num_value END)
        ) AS batch_size,
        MAX(CASE WHEN scope = 'adapter' AND key = 'inference_only' THEN num_value END) AS inference_only,
        MAX(CASE WHEN scope = 'splitter' AND key = 'distribution_type' THEN text_value END) AS data_distribution,
        COALESCE(
            MAX(CASE WHEN scope = 'splitter' AND key = 'effective_sample_size_total' THEN num_value END),
            MAX(CASE WHEN scope = 'splitter' AND key = 'sample_size' THEN num_value END),
            MAX(CASE WHEN scope = 'dataset' AND key = 'max_samples' THEN num_value END)
        ) AS dataset_size,
        MAX(CASE WHEN scope = 'runner' AND key = 'params_count' THEN num_value END) AS model_size
    FROM p
    GROUP BY run_id
),

run_level AS (
    SELECT
        run_id,
        MAX(CASE WHEN metric_name = 'metric_primary_name' THEN text_value END) AS primary_metric_name,
        MAX(CASE WHEN metric_name = 'metric_secondary_name' THEN text_value END) AS auxiliary_metric_name,
        MAX(CASE WHEN metric_name = 'run_total_runtime_s' THEN num_value END) AS run_total_runtime_s,
        MAX(CASE WHEN metric_name = 'train_set_size' THEN num_value END) AS train_set_size
    FROM m
    WHERE round IS NULL AND client_id IS NULL
    GROUP BY run_id
),

last_round AS (
    SELECT run_id, MAX(round) AS final_round
    FROM m
    WHERE round IS NOT NULL
    GROUP BY run_id
),

final_round AS (
    SELECT
        m.run_id,
        MAX(CASE WHEN metric_name = 'global_metric_score' THEN num_value END) AS global_metric_score,
        MAX(CASE WHEN metric_name = 'global_aux_metric' THEN num_value END) AS global_aux_metric,
        MAX(CASE WHEN metric_name = 'global_accuracy' THEN num_value END) AS global_accuracy,
        MAX(CASE WHEN metric_name = 'global_f1' THEN num_value END) AS global_f1,
        MAX(CASE WHEN metric_name = 'global_map' THEN num_value END) AS global_map,
        MAX(CASE WHEN metric_name = 'global_iou' THEN num_value END) AS global_iou,
        MAX(CASE WHEN metric_name = 'global_masked_accuracy' THEN num_value END) AS global_masked_accuracy,
        MAX(CASE WHEN metric_name = 'global_r@1' THEN num_value END) AS global_recall_at_1,
        MAX(CASE WHEN metric_name = 'global_token_accuracy' THEN num_value END) AS global_token_accuracy,
        MAX(CASE WHEN metric_name = 'global_pixel_accuracy' THEN num_value END) AS global_pixel_accuracy,
        MAX(CASE WHEN metric_name = 'global_silhouette' THEN num_value END) AS global_silhouette,
        MAX(CASE WHEN metric_name = 'global_rouge1' THEN num_value END) AS global_rouge1,
        MAX(CASE WHEN metric_name = 'global_rouge2' THEN num_value END) AS global_rouge2,
        MAX(CASE WHEN metric_name IN ('global_rougel', 'global_rouge-l', 'global_rouge_l') THEN num_value END) AS global_rougel,
        MAX(CASE WHEN metric_name = 'global_bleu' THEN num_value END) AS global_bleu,
        MAX(CASE WHEN metric_name = 'global_cider' THEN num_value END) AS global_cider,
        MAX(CASE WHEN metric_name = 'global_exact_match' THEN num_value END) AS global_exact_match,
        MAX(CASE WHEN metric_name = 'global_rmse' THEN num_value END) AS global_rmse,
        MAX(CASE WHEN metric_name = 'global_mae' THEN num_value END) AS global_mae,
        MAX(CASE WHEN metric_name = 'global_loss' THEN num_value END) AS global_loss,
        MAX(CASE WHEN metric_name = 'global_perplexity' THEN num_value END) AS global_perplexity,
        MAX(CASE WHEN metric_name = 'round_inference_latency_s_mean' THEN num_value END) AS latency,
        MAX(CASE WHEN metric_name = 'round_inference_latency_s_p95' THEN num_value END) AS tail_latency
    FROM m
    JOIN last_round lr
      ON lr.run_id = m.run_id
     AND lr.final_round = m.round
    WHERE m.client_id IS NULL
    GROUP BY m.run_id
),

client_agg AS (
    SELECT
        run_id,
        AVG(CASE WHEN metric_name = 'metric_score' THEN num_value END) AS metric_score,
        AVG(CASE WHEN metric_name = 'extra_metric' THEN num_value END) AS extra_metric,
        AVG(CASE WHEN metric_name = 'accuracy' THEN num_value END) AS accuracy,
        AVG(CASE WHEN metric_name = 'f1' THEN num_value END) AS f1,
        AVG(CASE WHEN metric_name IN ('map', 'mAP') THEN num_value END) AS map_value,
        AVG(CASE WHEN metric_name = 'map@0.5' THEN num_value END) AS map_at_50,
        AVG(CASE WHEN metric_name = 'map@0.75' THEN num_value END) AS map_at_75,
        AVG(CASE WHEN metric_name = 'iou' THEN num_value END) AS iou,
        AVG(CASE WHEN metric_name = 'dice' THEN num_value END) AS dice,
        AVG(CASE WHEN metric_name = 'pixel_accuracy' THEN num_value END) AS pixel_accuracy,
        AVG(CASE WHEN metric_name = 'masked_accuracy' THEN num_value END) AS masked_accuracy,
        AVG(CASE WHEN metric_name = 'token_accuracy' THEN num_value END) AS token_accuracy,
        AVG(CASE WHEN metric_name = 'r@1' THEN num_value END) AS recall_at_1,
        AVG(CASE WHEN metric_name = 'r@5' THEN num_value END) AS recall_at_5,
        AVG(CASE WHEN metric_name = 'silhouette' THEN num_value END) AS silhouette,
        AVG(CASE WHEN metric_name = 'rouge1' THEN num_value END) AS rouge1,
        AVG(CASE WHEN metric_name = 'rouge2' THEN num_value END) AS rouge2,
        AVG(CASE WHEN metric_name IN ('rougel', 'rouge-l', 'rouge_l') THEN num_value END) AS rougel,
        AVG(CASE WHEN metric_name = 'bleu' THEN num_value END) AS bleu,
        AVG(CASE WHEN metric_name = 'cider' THEN num_value END) AS cider,
        AVG(CASE WHEN metric_name = 'exact_match' THEN num_value END) AS exact_match,
        AVG(CASE WHEN metric_name = 'rmse' THEN num_value END) AS rmse,
        AVG(CASE WHEN metric_name = 'mae' THEN num_value END) AS mae,
        AVG(CASE WHEN metric_name IN ('loss', 'train_loss', 'cross_entropy_loss') THEN num_value END) AS loss,
        AVG(CASE WHEN metric_name = 'perplexity' THEN num_value END) AS perplexity,
        AVG(CASE WHEN metric_name = 'inference_latency_s' THEN num_value END) AS latency,
        AVG(CASE WHEN metric_name = 'inference_latency_s_p95' THEN num_value END) AS tail_latency,
        MAX(CASE WHEN metric_name = 'inference_latency_s' THEN num_value END) AS max_latency,
        AVG(CASE WHEN metric_name = 'compute_time_s' THEN num_value END) AS mean_compute_time,
        AVG(CASE WHEN metric_name = 'participated_flag' THEN num_value END) AS participation_rate,
        AVG(CASE WHEN metric_name = 'truncation_rate' THEN num_value END) AS truncation_rate,
        AVG(CASE WHEN metric_name = 'memory_used_mb' THEN num_value END) AS memory_used_mb,
        AVG(CASE WHEN metric_name = 'gpu_memory_used_mb' THEN num_value END) AS gpu_memory_used_mb,
        SUM(CASE WHEN metric_name IN ('comm_bytes_up', 'comm_bytes_down') THEN num_value END) AS comm_bytes,
        AVG(CASE WHEN metric_name = 'explainability_score' THEN num_value END) AS explainability_score
    FROM m
    WHERE client_id IS NOT NULL
    GROUP BY run_id
),

round_participation AS (
    SELECT
        run_id,
        AVG(CASE WHEN scheduled_clients > 0 THEN participating_clients * 1.0 / scheduled_clients END) AS participation_rate
    FROM rounds
    GROUP BY run_id
),

raw_base AS (
    SELECT
        r.run_id,
        r.created_at,
        r.dataset,
        r.task_type,
        r.model_type,
        params.hf_model_id,

        COALESCE(
            lower(rl.primary_metric_name),
            CASE
                WHEN fr.global_accuracy IS NOT NULL OR ca.accuracy IS NOT NULL THEN 'accuracy'
                WHEN fr.global_f1 IS NOT NULL OR ca.f1 IS NOT NULL THEN 'f1'
                WHEN fr.global_map IS NOT NULL OR ca.map_value IS NOT NULL THEN 'map'
                WHEN fr.global_iou IS NOT NULL OR ca.iou IS NOT NULL THEN 'iou'
                WHEN fr.global_masked_accuracy IS NOT NULL OR ca.masked_accuracy IS NOT NULL THEN 'masked_accuracy'
                WHEN fr.global_recall_at_1 IS NOT NULL OR ca.recall_at_1 IS NOT NULL THEN 'r@1'
                WHEN fr.global_token_accuracy IS NOT NULL OR ca.token_accuracy IS NOT NULL THEN 'token_accuracy'
                WHEN fr.global_pixel_accuracy IS NOT NULL OR ca.pixel_accuracy IS NOT NULL THEN 'pixel_accuracy'
                WHEN fr.global_silhouette IS NOT NULL OR ca.silhouette IS NOT NULL THEN 'silhouette'
                WHEN fr.global_rouge1 IS NOT NULL OR ca.rouge1 IS NOT NULL THEN 'rouge1'
                WHEN fr.global_rougel IS NOT NULL OR ca.rougel IS NOT NULL THEN 'rougel'
                WHEN fr.global_rmse IS NOT NULL OR ca.rmse IS NOT NULL THEN 'rmse'
                WHEN fr.global_loss IS NOT NULL OR ca.loss IS NOT NULL THEN 'loss'
                WHEN fr.global_perplexity IS NOT NULL OR ca.perplexity IS NOT NULL THEN 'perplexity'
                WHEN fr.global_metric_score IS NOT NULL OR ca.metric_score IS NOT NULL THEN 'metric_score'
            END
        ) AS primary_metric_name,

        COALESCE(
            lower(rl.auxiliary_metric_name),
            CASE
                WHEN fr.global_aux_metric IS NOT NULL OR ca.extra_metric IS NOT NULL THEN 'auxiliary_metric'
                WHEN fr.global_f1 IS NOT NULL OR ca.f1 IS NOT NULL THEN 'f1'
                WHEN fr.global_rmse IS NOT NULL OR ca.rmse IS NOT NULL THEN 'rmse'
                WHEN fr.global_perplexity IS NOT NULL OR ca.perplexity IS NOT NULL THEN 'perplexity'
                WHEN fr.global_loss IS NOT NULL OR ca.loss IS NOT NULL THEN 'loss'
            END
        ) AS auxiliary_metric_name,

        fr.global_metric_score,
        fr.global_aux_metric,
        fr.global_accuracy,
        fr.global_f1,
        fr.global_map,
        fr.global_iou,
        fr.global_masked_accuracy,
        fr.global_recall_at_1,
        fr.global_token_accuracy,
        fr.global_pixel_accuracy,
        fr.global_silhouette,
        fr.global_rouge1,
        fr.global_rouge2,
        fr.global_rougel,
        fr.global_bleu,
        fr.global_cider,
        fr.global_exact_match,
        fr.global_rmse,
        fr.global_mae,
        fr.global_loss,
        fr.global_perplexity,

        ca.metric_score,
        ca.extra_metric,
        ca.accuracy,
        ca.f1,
        ca.map_value,
        ca.map_at_50,
        ca.map_at_75,
        ca.iou,
        ca.dice,
        ca.pixel_accuracy,
        ca.masked_accuracy,
        ca.token_accuracy,
        ca.recall_at_1,
        ca.recall_at_5,
        ca.silhouette,
        ca.rouge1,
        ca.rouge2,
        ca.rougel,
        ca.bleu,
        ca.cider,
        ca.exact_match,
        ca.rmse,
        ca.mae,
        ca.loss,
        ca.perplexity,

        COALESCE(fr.latency, ca.latency) AS latency,
        COALESCE(fr.tail_latency, ca.tail_latency, ca.max_latency) AS tail_latency,
        COALESCE(
            ca.participation_rate,
            rp.participation_rate,
            CASE
                WHEN fr.global_metric_score IS NOT NULL
                  OR fr.global_accuracy IS NOT NULL
                  OR fr.global_map IS NOT NULL
                  OR fr.global_iou IS NOT NULL
                  OR fr.global_rmse IS NOT NULL
                  OR fr.global_loss IS NOT NULL
                THEN 1.0
            END
        ) AS participation_rate,
        ca.mean_compute_time,

        CASE
            WHEN rl.run_total_runtime_s IS NULL
             AND ca.mean_compute_time IS NULL
             AND ca.memory_used_mb IS NULL
             AND ca.gpu_memory_used_mb IS NULL
             AND ca.comm_bytes IS NULL
             AND params.model_size IS NULL
            THEN NULL
            ELSE
                COALESCE(rl.run_total_runtime_s, 0.0)
              + COALESCE(ca.mean_compute_time, 0.0)
              + COALESCE(ca.memory_used_mb, 0.0) / 1024.0
              + COALESCE(ca.gpu_memory_used_mb, 0.0) / 1024.0
              + COALESCE(ca.comm_bytes, 0.0) / 1073741824.0
              + CASE WHEN params.model_size IS NOT NULL AND params.model_size > 0 THEN params.model_size / 1000000.0 ELSE 0.0 END
        END AS raw_resource_cost,

        params.model_size,
        params.downloads,
        params.likes,
        CASE WHEN COALESCE(params.inference_only, 0.0) = 1.0 THEN NULL ELSE params.learning_rate END AS learning_rate,
        params.batch_size,
        params.data_distribution,
        COALESCE(rl.train_set_size, params.dataset_size) AS dataset_size,
        ca.explainability_score
    FROM runs r
    LEFT JOIN params ON params.run_id = r.run_id
    LEFT JOIN run_level rl ON rl.run_id = r.run_id
    LEFT JOIN final_round fr ON fr.run_id = r.run_id
    LEFT JOIN client_agg ca ON ca.run_id = r.run_id
    LEFT JOIN round_participation rp ON rp.run_id = r.run_id
),

raw AS (
    SELECT
        *,
        CASE replace(primary_metric_name, '-', '_')
            WHEN 'accuracy' THEN COALESCE(global_accuracy, accuracy, global_metric_score, metric_score)
            WHEN 'top1_accuracy' THEN COALESCE(global_accuracy, accuracy, global_metric_score, metric_score)
            WHEN 'f1' THEN COALESCE(global_f1, f1, global_metric_score, metric_score)
            WHEN 'macro_f1' THEN COALESCE(global_f1, f1, global_metric_score, metric_score)
            WHEN 'entity_f1' THEN COALESCE(global_f1, f1, global_metric_score, metric_score)
            WHEN 'map' THEN COALESCE(global_map, map_value, global_metric_score, metric_score)
            WHEN 'mean_average_precision' THEN COALESCE(global_map, map_value, global_metric_score, metric_score)
            WHEN 'map@0.5' THEN COALESCE(map_at_50, global_aux_metric, extra_metric)
            WHEN 'map@0.75' THEN COALESCE(map_at_75, global_aux_metric, extra_metric)
            WHEN 'iou' THEN COALESCE(global_iou, iou, global_metric_score, metric_score)
            WHEN 'miou' THEN COALESCE(global_iou, iou, global_metric_score, metric_score)
            WHEN 'mean_iou' THEN COALESCE(global_iou, iou, global_metric_score, metric_score)
            WHEN 'dice' THEN COALESCE(dice, global_aux_metric, extra_metric)
            WHEN 'pixel_accuracy' THEN COALESCE(global_pixel_accuracy, pixel_accuracy)
            WHEN 'masked_accuracy' THEN COALESCE(global_masked_accuracy, masked_accuracy, global_metric_score, metric_score)
            WHEN 'token_accuracy' THEN COALESCE(global_token_accuracy, token_accuracy)
            WHEN 'r@1' THEN COALESCE(global_recall_at_1, recall_at_1, global_metric_score, metric_score)
            WHEN 'silhouette' THEN COALESCE(global_silhouette, silhouette, global_metric_score, metric_score)
            WHEN 'rouge1' THEN COALESCE(global_rouge1, rouge1, global_metric_score, metric_score)
            WHEN 'rouge2' THEN COALESCE(global_rouge2, rouge2, global_aux_metric, extra_metric)
            WHEN 'rougel' THEN COALESCE(global_rougel, rougel, global_metric_score, metric_score)
            WHEN 'rouge_l' THEN COALESCE(global_rougel, rougel, global_metric_score, metric_score)
            WHEN 'bleu' THEN COALESCE(global_bleu, bleu, global_metric_score, metric_score)
            WHEN 'cider' THEN COALESCE(global_cider, cider, global_metric_score, metric_score)
            WHEN 'exact_match' THEN COALESCE(global_exact_match, exact_match, global_metric_score, metric_score)
            WHEN 'rmse' THEN COALESCE(global_rmse, rmse)
            WHEN 'mae' THEN COALESCE(global_mae, mae, global_aux_metric, extra_metric)
            WHEN 'loss' THEN COALESCE(global_loss, loss)
            WHEN 'cross_entropy_loss' THEN COALESCE(global_loss, loss)
            WHEN 'perplexity' THEN COALESCE(global_perplexity, perplexity)
            WHEN 'perplexity_proxy' THEN COALESCE(global_aux_metric, extra_metric, global_perplexity, perplexity)
            WHEN 'metric_score' THEN COALESCE(global_metric_score, metric_score)
            ELSE COALESCE(
                global_metric_score,
                metric_score,
                global_accuracy,
                accuracy,
                global_f1,
                f1,
                global_map,
                map_value,
                global_iou,
                iou,
                global_rmse,
                rmse,
                global_loss,
                loss
            )
        END AS primary_metric,

        CASE replace(auxiliary_metric_name, '-', '_')
            WHEN 'accuracy' THEN COALESCE(global_accuracy, accuracy, global_aux_metric, extra_metric)
            WHEN 'top1_accuracy' THEN COALESCE(global_accuracy, accuracy, global_aux_metric, extra_metric)
            WHEN 'f1' THEN COALESCE(global_f1, f1, global_aux_metric, extra_metric)
            WHEN 'macro_f1' THEN COALESCE(global_f1, f1, global_aux_metric, extra_metric)
            WHEN 'entity_f1' THEN COALESCE(global_f1, f1, global_aux_metric, extra_metric)
            WHEN 'map' THEN COALESCE(global_map, map_value, global_aux_metric, extra_metric)
            WHEN 'mean_average_precision' THEN COALESCE(global_map, map_value, global_aux_metric, extra_metric)
            WHEN 'map@0.5' THEN COALESCE(map_at_50, global_aux_metric, extra_metric)
            WHEN 'map@0.75' THEN COALESCE(map_at_75, global_aux_metric, extra_metric)
            WHEN 'iou' THEN COALESCE(global_iou, iou, global_aux_metric, extra_metric)
            WHEN 'miou' THEN COALESCE(global_iou, iou, global_aux_metric, extra_metric)
            WHEN 'mean_iou' THEN COALESCE(global_iou, iou, global_aux_metric, extra_metric)
            WHEN 'dice' THEN COALESCE(dice, global_aux_metric, extra_metric)
            WHEN 'pixel_accuracy' THEN COALESCE(global_pixel_accuracy, pixel_accuracy, global_aux_metric, extra_metric)
            WHEN 'masked_accuracy' THEN COALESCE(global_masked_accuracy, masked_accuracy, global_aux_metric, extra_metric)
            WHEN 'token_accuracy' THEN COALESCE(global_token_accuracy, token_accuracy, global_aux_metric, extra_metric)
            WHEN 'r@5' THEN COALESCE(recall_at_5, global_aux_metric, extra_metric)
            WHEN 'silhouette' THEN COALESCE(global_silhouette, silhouette, global_aux_metric, extra_metric)
            WHEN 'rouge1' THEN COALESCE(global_rouge1, rouge1, global_aux_metric, extra_metric)
            WHEN 'rouge2' THEN COALESCE(global_rouge2, rouge2, global_aux_metric, extra_metric)
            WHEN 'rougel' THEN COALESCE(global_rougel, rougel, global_aux_metric, extra_metric)
            WHEN 'rouge_l' THEN COALESCE(global_rougel, rougel, global_aux_metric, extra_metric)
            WHEN 'bleu' THEN COALESCE(global_bleu, bleu, global_aux_metric, extra_metric)
            WHEN 'cider' THEN COALESCE(global_cider, cider, global_aux_metric, extra_metric)
            WHEN 'exact_match' THEN COALESCE(global_exact_match, exact_match, global_aux_metric, extra_metric)
            WHEN 'rmse' THEN COALESCE(global_rmse, rmse, global_aux_metric, extra_metric)
            WHEN 'mae' THEN COALESCE(global_mae, mae, global_aux_metric, extra_metric)
            WHEN 'loss' THEN COALESCE(global_loss, loss, global_aux_metric, extra_metric)
            WHEN 'cross_entropy_loss' THEN COALESCE(global_loss, loss, global_aux_metric, extra_metric)
            WHEN 'perplexity' THEN COALESCE(global_perplexity, perplexity, global_aux_metric, extra_metric)
            WHEN 'perplexity_proxy' THEN COALESCE(global_aux_metric, extra_metric, global_perplexity, perplexity)
            WHEN 'auxiliary_metric' THEN COALESCE(global_aux_metric, extra_metric)
            ELSE COALESCE(
                global_aux_metric,
                extra_metric,
                global_f1,
                f1,
                global_rmse,
                rmse,
                global_perplexity,
                perplexity,
                global_loss,
                loss
            )
        END AS auxiliary_metric
    FROM raw_base
),

scored AS (
    SELECT
        *,
        COALESCE(
            global_metric_score,
            CASE
                WHEN replace(primary_metric_name, '-', '_') IN ('rmse', 'mae', 'loss', 'cross_entropy_loss', 'perplexity', 'perplexity_proxy')
                 AND primary_metric IS NOT NULL
                THEN 1.0 / (1.0 + primary_metric)
                ELSE primary_metric
            END
        ) AS primary_metric_higher_better,
        CASE
            WHEN replace(auxiliary_metric_name, '-', '_') IN ('rmse', 'mae', 'loss', 'cross_entropy_loss', 'perplexity', 'perplexity_proxy')
             AND auxiliary_metric IS NOT NULL
            THEN 1.0 / (1.0 + auxiliary_metric)
            ELSE auxiliary_metric
        END AS auxiliary_metric_higher_better
    FROM raw
),

final_values AS (
    SELECT
        *,
        CASE
            WHEN primary_metric_higher_better IS NULL THEN NULL
            WHEN MAX(primary_metric_higher_better) OVER () = MIN(primary_metric_higher_better) OVER () THEN 1.0
            ELSE max(0.0, min(1.0,
                (primary_metric_higher_better - MIN(primary_metric_higher_better) OVER ())
                / NULLIF(MAX(primary_metric_higher_better) OVER () - MIN(primary_metric_higher_better) OVER (), 0.0)
            ))
        END AS normalised_primary_metric,
        CASE
            WHEN auxiliary_metric_higher_better IS NULL THEN NULL
            WHEN MAX(auxiliary_metric_higher_better) OVER () = MIN(auxiliary_metric_higher_better) OVER () THEN 1.0
            ELSE max(0.0, min(1.0,
                (auxiliary_metric_higher_better - MIN(auxiliary_metric_higher_better) OVER ())
                / NULLIF(MAX(auxiliary_metric_higher_better) OVER () - MIN(auxiliary_metric_higher_better) OVER (), 0.0)
            ))
        END AS normalised_auxiliary_metric,
        CASE
            WHEN raw_resource_cost IS NULL THEN NULL
            WHEN MAX(raw_resource_cost) OVER () = MIN(raw_resource_cost) OVER () THEN 1.0
            ELSE max(0.0, min(1.0,
                1.0 - (
                    (raw_resource_cost - MIN(raw_resource_cost) OVER ())
                    / NULLIF(MAX(raw_resource_cost) OVER () - MIN(raw_resource_cost) OVER (), 0.0)
                )
            ))
        END AS resource_cost_score
    FROM scored
),

final_with_efficiency AS (
    SELECT
        *,
        CASE
            WHEN normalised_primary_metric IS NULL OR raw_resource_cost IS NULL THEN NULL
            ELSE normalised_primary_metric / (raw_resource_cost + 0.000000001)
        END AS cost_efficiency
    FROM final_values
)

SELECT
    run_id,
    created_at,
    dataset,
    task_type,
    model_type,

    COALESCE(primary_metric_name, 'Not Available') AS "Primary metric name",
    COALESCE(CAST(primary_metric AS TEXT), 'Not Available') AS "Primary metric",
    COALESCE(auxiliary_metric_name, 'Not Available') AS "Auxiliary metric name",
    COALESCE(CAST(auxiliary_metric AS TEXT), 'Not Available') AS "Auxiliary metric",
    COALESCE(CAST(latency AS TEXT), 'Not Available') AS "Latency",
    COALESCE(CAST(tail_latency AS TEXT), 'Not Available') AS "Tail latency",
    COALESCE(CAST(participation_rate AS TEXT), 'Not Available') AS "Participation rate",
    COALESCE(CAST(max(0.0, min(1.0, participation_rate)) AS TEXT), 'Not Available') AS "Reliability score",
    COALESCE(CAST(mean_compute_time AS TEXT), 'Not Available') AS "Mean compute time",
    COALESCE(CAST(resource_cost_score AS TEXT), 'Not Available') AS "Resource cost score",
    COALESCE(CAST(cost_efficiency AS TEXT), 'Not Available') AS "Cost efficiency",
    CASE
        WHEN model_size IS NULL OR model_size <= 0 THEN 'Not Available'
        ELSE CAST(model_size AS TEXT)
    END AS "Model size",
    COALESCE(CAST(downloads AS TEXT), 'Not Available') AS "Downloads",
    COALESCE(CAST(likes AS TEXT), 'Not Available') AS "Likes",
    COALESCE(CAST(learning_rate AS TEXT), 'Not Available') AS "Learning rate",
    COALESCE(CAST(batch_size AS TEXT), 'Not Available') AS "Batch size",
    COALESCE(data_distribution, 'Not Available') AS "Data distribution",
    COALESCE(CAST(dataset_size AS TEXT), 'Not Available') AS "Dataset size",
    COALESCE(CAST(explainability_score AS TEXT), 'Not Available') AS "Explainability score"
FROM final_with_efficiency
ORDER BY created_at DESC;
