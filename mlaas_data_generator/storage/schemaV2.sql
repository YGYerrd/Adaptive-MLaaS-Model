PRAGMA foreign_keys = ON;

-- =========================
-- 1) Runs
-- =========================
CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  dataset       TEXT,
  task_type     TEXT,
  model_type    TEXT,
  num_clients   INTEGER,
  num_rounds    INTEGER,
  db_version    INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);


-- =========================
-- 2) Rounds (dimension)
-- =========================
CREATE TABLE IF NOT EXISTS rounds (
  run_id                 TEXT NOT NULL,
  round                  INTEGER NOT NULL,
  scheduled_clients      INTEGER,
  attempted_clients      INTEGER,
  participating_clients  INTEGER,
  dropped_clients        INTEGER,
  created_at             TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (run_id, round),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  CHECK (round >= 1)
);

CREATE INDEX IF NOT EXISTS idx_rounds_run ON rounds(run_id);


-- =========================
-- 3) Clients (dimension)
-- =========================
CREATE TABLE IF NOT EXISTS clients (
  run_id                 TEXT NOT NULL,
  client_id              TEXT NOT NULL,
  created_at             TEXT NOT NULL DEFAULT (datetime('now')),
  data_distribution_json TEXT,
  samples_count          INTEGER,
  PRIMARY KEY (run_id, client_id),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_clients_run ON clients(run_id);


-- =========================
-- 4) Run Parameters (normalised config)
-- =========================
CREATE TABLE IF NOT EXISTS run_params (
  run_id       TEXT NOT NULL,
  scope        TEXT NOT NULL,  -- runner|dataset|adapter|aggregator|splitter
  key          TEXT NOT NULL,
  value_text   TEXT,
  value_num    REAL,
  value_int    INTEGER,
  value_bool   INTEGER,
  value_json   TEXT,
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (run_id, scope, key),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  CHECK (scope IN ('runner','dataset','adapter','aggregator','splitter')),
  CHECK (value_bool IS NULL OR value_bool IN (0,1)),
  CHECK (
    (value_text IS NOT NULL) +
    (value_num  IS NOT NULL) +
    (value_int  IS NOT NULL) +
    (value_bool IS NOT NULL) +
    (value_json IS NOT NULL)
    = 1
  )
);

CREATE INDEX IF NOT EXISTS idx_run_params_run_scope ON run_params(run_id, scope);


-- =========================
-- 5) Metric Definitions
-- =========================
CREATE TABLE IF NOT EXISTS metrics (
  metric_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL UNIQUE,
  domain      TEXT NOT NULL,   -- quality|performance|reliability|cost|resource
  unit        TEXT,
  direction   TEXT NOT NULL,   -- higher_better|lower_better|neutral
  data_type   TEXT NOT NULL,   -- num|int|bool|text|json
  description TEXT,
  CHECK (domain IN ('quality','performance','reliability','cost','resource')),
  CHECK (direction IN ('higher_better','lower_better','neutral')),
  CHECK (data_type IN ('num','int','bool','text','json'))
);

CREATE INDEX IF NOT EXISTS idx_metrics_domain ON metrics(domain);


-- =========================
-- 6) Measurements (universal fact table)
-- =========================
CREATE TABLE IF NOT EXISTS measurements (
  measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id         TEXT NOT NULL,
  round          INTEGER,
  client_id      TEXT,
  metric_id      INTEGER NOT NULL,
  value_num      REAL,
  value_int      INTEGER,
  value_bool     INTEGER,
  value_text     TEXT,
  value_json     TEXT,
  created_at     TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY (metric_id) REFERENCES metrics(metric_id) ON DELETE RESTRICT,
  CHECK (round IS NULL OR round >= 1),
  CHECK (value_bool IS NULL OR value_bool IN (0,1)),
  CHECK (
    (value_num  IS NOT NULL) +
    (value_int  IS NOT NULL) +
    (value_bool IS NOT NULL) +
    (value_text IS NOT NULL) +
    (value_json IS NOT NULL)
    = 1
  )
);

-- Prevent duplicate metric at same grain
CREATE UNIQUE INDEX IF NOT EXISTS uq_measurements_grain
ON measurements(run_id, round, client_id, metric_id);

CREATE INDEX IF NOT EXISTS idx_measurements_run_round ON measurements(run_id, round);
CREATE INDEX IF NOT EXISTS idx_measurements_run_client ON measurements(run_id, client_id);
CREATE INDEX IF NOT EXISTS idx_measurements_metric ON measurements(metric_id);


-- =========================
-- 7) Run Failures
-- =========================
CREATE TABLE IF NOT EXISTS run_failures (
  failure_id           INTEGER PRIMARY KEY AUTOINCREMENT,
  external_run_id      TEXT,
  row_index            INTEGER,
  case_name            TEXT,
  run_group_id         TEXT,
  failure_stage        TEXT NOT NULL,
  error_message        TEXT,
  resolved_config_json TEXT,
  traceback_text       TEXT,
  created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_run_failures_external_run_id ON run_failures(external_run_id);
CREATE INDEX IF NOT EXISTS idx_run_failures_run_group_id ON run_failures(run_group_id);
CREATE INDEX IF NOT EXISTS idx_run_failures_stage ON run_failures(failure_stage);


-- =========================
-- 8) Convenience View
-- =========================
CREATE VIEW IF NOT EXISTS v_measurements AS
SELECT
  m.measurement_id,
  m.run_id,
  m.round,
  m.client_id,
  md.name AS metric_name,
  md.domain,
  md.unit,
  md.direction,
  md.data_type,
  m.value_num,
  m.value_int,
  m.value_bool,
  m.value_text,
  m.value_json,
  m.created_at
FROM measurements m
JOIN metrics md ON md.metric_id = m.metric_id;
