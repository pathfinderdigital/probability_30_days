-- Step 0: Declare variables
DECLARE page_paths ARRAY<STRING>;
DECLARE pivot_columns STRING;
DECLARE sql STRING;

-- Step 1: Extract base events from the past 6 months
CREATE OR REPLACE TABLE `mokosh.analytics_317847082.base_events` AS
SELECT
  user_pseudo_id,
  event_name,
  event_timestamp,
  (SELECT value.int_value FROM UNNEST(event_params) WHERE key = "engagement_time_msec") AS engagement_time_msec,
  REGEXP_EXTRACT((SELECT value.string_value FROM UNNEST(event_params) WHERE key = "page_location"), r'https?://[^/]+(/[^?#]*)') AS page_path,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = "device_category") AS device_category,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = "geo_country") AS geo_country,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = "source") AS source,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = "medium") AS medium
FROM
  `mokosh.analytics_317847082.events_*`
WHERE
  _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 180 DAY))
  AND FORMAT_DATE('%Y%m%d', CURRENT_DATE());

-- Step 2: Aggregate user-level features (expanded behavior)
CREATE OR REPLACE TABLE `mokosh.analytics_317847082.user_aggregates` AS
SELECT
  user_pseudo_id,
  COUNTIF(event_name = 'session_start') AS session_count,
  COUNTIF(event_name = 'view_item') AS view_item_count,
  COUNTIF(event_name = 'add_to_cart') AS add_to_cart_count,
  COUNTIF(event_name = 'begin_checkout') AS begin_checkout_count,
  COUNTIF(event_name = 'view_cart') AS view_cart_count,
  COUNTIF(event_name = 'remove_from_cart') AS remove_from_cart_count,
  COUNTIF(event_name = 'add_shipping_info') AS add_shipping_info_count,
  SUM(CAST(engagement_time_msec AS INT64)) AS total_engagement_time,
  DATE_DIFF(CURRENT_DATE(), DATE(TIMESTAMP_MICROS(MAX(event_timestamp))), DAY) AS days_since_last_event,
  MAX(device_category) AS device_category,
  MAX(geo_country) AS geo_country,
  MAX(source) AS traffic_source_source,
  MAX(medium) AS traffic_source_medium
FROM `mokosh.analytics_317847082.base_events`
GROUP BY user_pseudo_id;


-- Step 3: Count page views per user and page_path
CREATE OR REPLACE TABLE `mokosh.analytics_317847082.page_path_counts` AS
SELECT
  user_pseudo_id,
  page_path,
  COUNT(*) AS view_count
FROM `mokosh.analytics_317847082.base_events`
WHERE page_path IS NOT NULL
GROUP BY user_pseudo_id, page_path;

-- Step 4: Prepare dynamic SQL to pivot top 30 page paths
SET page_paths = (
  SELECT ARRAY_AGG(page_path)
  FROM `mokosh.analytics_317847082.top_30_pages`
  WHERE page_path IS NOT NULL
);

SET pivot_columns = (
  SELECT STRING_AGG(
    FORMAT("SUM(IF(page_path = '%s', view_count, 0)) AS `%s`",
      path,
      REGEXP_REPLACE(REPLACE(path, '/', '_'), r'[^a-zA-Z0-9_]', '_')
    ),
    ', '
  )
  FROM UNNEST(page_paths) AS path
);

SET sql = FORMAT("""
  CREATE OR REPLACE TABLE `mokosh.analytics_317847082.pivoted_page_views` AS
  SELECT
    user_pseudo_id,
    %s
  FROM `mokosh.analytics_317847082.page_path_counts`
  WHERE page_path IN UNNEST(@page_paths)
  GROUP BY user_pseudo_id;
""", pivot_columns);

EXECUTE IMMEDIATE sql USING page_paths AS page_paths;

-- Step 5: Combine features with null-safe categories
CREATE OR REPLACE TABLE `mokosh.analytics_317847082.user_features` AS
SELECT
  a.user_pseudo_id,
  a.session_count,
  a.view_item_count,
  a.add_to_cart_count,
  a.total_engagement_time,
  a.days_since_last_event,
  IFNULL(a.device_category, 'unknown') AS device_category,
  IFNULL(a.geo_country, 'unknown') AS geo_country,
  IFNULL(a.traffic_source_source, 'unknown') AS traffic_source_source,
  IFNULL(a.traffic_source_medium, 'unknown') AS traffic_source_medium,
  p.*
EXCEPT(user_pseudo_id)
FROM `mokosh.analytics_317847082.user_aggregates` a
LEFT JOIN `mokosh.analytics_317847082.pivoted_page_views` p
ON a.user_pseudo_id = p.user_pseudo_id;

-- Step 6: Label users based on purchase events in the past 30 days
CREATE OR REPLACE TABLE `mokosh.analytics_317847082.model_training_data` AS
SELECT
  uf.*,
  IF(p.user_pseudo_id IS NOT NULL, 1, 0) AS made_purchase
FROM `mokosh.analytics_317847082.user_features` uf
LEFT JOIN (
  SELECT DISTINCT user_pseudo_id
  FROM `mokosh.analytics_317847082.base_events`
  WHERE event_name IN ('purchase', 'in_app_purchase')
    AND DATE(TIMESTAMP_MICROS(event_timestamp)) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
) AS p
ON uf.user_pseudo_id = p.user_pseudo_id;

-- Step 7: Create a 3:1 balanced training dataset
CREATE OR REPLACE TABLE `mokosh.analytics_317847082.model_training_data_balanced` AS
SELECT * FROM `mokosh.analytics_317847082.model_training_data`
WHERE made_purchase = 1
UNION ALL
SELECT * FROM (
  SELECT * FROM `mokosh.analytics_317847082.model_training_data`
  WHERE made_purchase = 0
  ORDER BY RAND()
  LIMIT 651
);

-- Step 8: Train logistic regression model
CREATE OR REPLACE MODEL `mokosh.analytics_317847082.purchase_propensity_model_balanced`
OPTIONS(
  model_type='logistic_reg',
  input_label_cols=['made_purchase']
) AS
SELECT
  *
FROM
  `mokosh.analytics_317847082.model_training_data_balanced`;

-- Step 9: Evaluate the model
CREATE OR REPLACE TABLE `mokosh.analytics_317847082.model_evaluation_balanced` AS
SELECT *
FROM
  ML.EVALUATE(MODEL `mokosh.analytics_317847082.purchase_propensity_model_balanced`,
    (
      SELECT *
      FROM `mokosh.analytics_317847082.model_training_data_balanced`
    )
  );

-- Step 10: Score all users and rank by purchase probability + label
CREATE OR REPLACE TABLE `mokosh.analytics_317847082.user_scores_balanced` AS
SELECT
  user_pseudo_id,
  predicted_made_purchase AS predicted_label,
  predicted_made_purchase_probs[OFFSET(1)].prob AS purchase_probability,
  RANK() OVER (ORDER BY predicted_made_purchase_probs[OFFSET(1)].prob DESC) AS rank
FROM
  ML.PREDICT(MODEL `mokosh.analytics_317847082.purchase_propensity_model_balanced`,
    (
      SELECT * FROM `mokosh.analytics_317847082.user_features`
    )
  );

-- Step 11: Extract likely purchasers to "high-intent" table
CREATE OR REPLACE TABLE `mokosh.analytics_317847082.high_intent_users` AS
SELECT *
FROM `mokosh.analytics_317847082.user_scores_balanced`
WHERE predicted_label = 1
ORDER BY purchase_probability DESC
LIMIT 5000;
