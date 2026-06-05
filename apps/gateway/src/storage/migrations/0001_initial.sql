CREATE TABLE responses (
  id TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  previous_response_id TEXT NULL,
  status TEXT NOT NULL,
  input_json JSONB NOT NULL,
  output_json JSONB NOT NULL,
  request_json JSONB NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}',
  usage_json JSONB NULL,
  error_json JSONB NULL,
  tenant_id TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ NULL,
  deleted_at TIMESTAMPTZ NULL
);

CREATE TABLE response_items (
  id TEXT PRIMARY KEY,
  response_id TEXT NOT NULL REFERENCES responses(id),
  type TEXT NOT NULL,
  role TEXT NULL,
  content_json JSONB NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE tool_calls (
  id TEXT PRIMARY KEY,
  response_id TEXT NOT NULL REFERENCES responses(id),
  name TEXT NOT NULL,
  arguments_json JSONB NOT NULL,
  output_json JSONB NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ NULL
);

CREATE TABLE usage_records (
  id TEXT PRIMARY KEY,
  response_id TEXT NOT NULL REFERENCES responses(id),
  model TEXT NOT NULL,
  input_tokens INT NOT NULL DEFAULT 0,
  output_tokens INT NOT NULL DEFAULT 0,
  total_tokens INT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
