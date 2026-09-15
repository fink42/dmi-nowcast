-- Tag rates over an annotation export, with the control group as contrast
-- and the drift check beside it (event review, WP4).
--
-- Input is the Parquet written by
-- `scripts/review_server.py --export parquet` (or POST /review-api/export):
-- one row per annotated event, `tags` as a list<string>, `mechanism_tags`
-- as the same list minus review_schema.NON_MECHANISM_TAGS, plus a boolean
-- `tag_<code>` column per vocabulary code. This query uses the list
-- columns rather than the booleans on purpose — unnesting is one line and
-- it never has to restate the vocabulary, so a tag added in
-- review_schema.py needs no edit here. The boolean columns are for the
-- ad-hoc cross-tabs that want a plain GROUP BY.
--
-- Two questions, one result set, distinguished by `view`:
--
--   view = 'class_verdict'
--     How often does each tag appear within (class × verdict), and how
--     often does the SAME tag appear in the `hit` control group? The
--     contrast is the whole point: a tag on 40 % of the false alarms that
--     is also on 40 % of the hits describes the weather, not the failure.
--     `lift` is share / control_share, null when the control group never
--     carries the tag (an infinite lift is not a number worth ranking).
--
--   view = 'decile'
--     The same rate by review-order decile within each class — the drift
--     check. Over 300 events across many sessions a reviewer's criteria
--     move, and a mechanism whose rate climbs monotonically from decile 1
--     to decile 10 is a finding about the reviewer, not about the
--     forecast. `review_decile` is computed at export time (ranked over
--     the judged rows by `review_seq`, the order each event was FIRST
--     saved) so that the Parquet, the markdown report and this query
--     cannot disagree about where a decile boundary falls.
--
-- Only rows with a verdict are counted: an event nobody has judged has no
-- tags and would only dilute every denominator.
--
-- Parameter: {corpus} — replaced with the quoted export Parquet path
-- (see sql/README.md).
--
--     sed "s|{corpus}|'review/exports/annotations-20260915T101500Z.parquet'|" \
--         sql/review_tags.sql | duckdb

WITH judged AS (
    SELECT
        event_id, event_class, verdict, review_decile, tags, mechanism_tags
    FROM read_parquet({corpus})
    WHERE verdict IS NOT NULL
),
exploded AS (
    SELECT
        event_id,
        event_class,
        verdict,
        review_decile,
        unnest(tags) AS tag,
        mechanism_tags
    FROM judged
),
tagged AS (
    SELECT
        event_id, event_class, verdict, review_decile, tag,
        list_contains(mechanism_tags, tag) AS is_mechanism
    FROM exploded
),
-- Denominators. A tag rate needs the count of JUDGED EVENTS in the group,
-- not the count of tag rows: events carry several tags each.
n_class_verdict AS (
    SELECT event_class, verdict, count(*) AS n_group
    FROM judged GROUP BY event_class, verdict
),
n_decile AS (
    SELECT event_class, review_decile, count(*) AS n_group
    FROM judged GROUP BY event_class, review_decile
),
-- The control group, pooled over its verdicts: it is a base rate, and
-- splitting it by verdict would leave single-digit denominators.
n_control AS (
    SELECT count(*) AS n FROM judged WHERE event_class = 'hit'
),
control_tags AS (
    SELECT tag, count(*) AS n_tagged
    FROM tagged WHERE event_class = 'hit' GROUP BY tag
),
by_class_verdict AS (
    SELECT
        t.event_class,
        t.verdict,
        NULL::INTEGER      AS review_decile,
        t.tag,
        any_value(t.is_mechanism) AS is_mechanism,
        count(*)           AS n_tagged,
        max(g.n_group)     AS n_group
    FROM tagged t
    JOIN n_class_verdict g
      ON g.event_class = t.event_class AND g.verdict = t.verdict
    GROUP BY t.event_class, t.verdict, t.tag
),
by_decile AS (
    SELECT
        t.event_class,
        NULL::VARCHAR      AS verdict,
        t.review_decile,
        t.tag,
        any_value(t.is_mechanism) AS is_mechanism,
        count(*)           AS n_tagged,
        max(g.n_group)     AS n_group
    FROM tagged t
    JOIN n_decile g
      ON g.event_class = t.event_class AND g.review_decile = t.review_decile
    GROUP BY t.event_class, t.review_decile, t.tag
),
combined AS (
    SELECT 'class_verdict' AS view, * FROM by_class_verdict
    UNION ALL
    SELECT 'decile'        AS view, * FROM by_decile
)
SELECT
    c.view,
    c.event_class,
    c.verdict,
    c.review_decile,
    c.tag,
    c.is_mechanism,
    c.n_tagged,
    c.n_group,
    c.n_tagged::DOUBLE / c.n_group                       AS tag_rate,
    COALESCE(ct.n_tagged, 0)                             AS control_n_tagged,
    nc.n                                                 AS control_n,
    CASE WHEN nc.n > 0
         THEN COALESCE(ct.n_tagged, 0)::DOUBLE / nc.n
    END                                                  AS control_rate,
    CASE WHEN nc.n > 0 AND COALESCE(ct.n_tagged, 0) > 0
         THEN (c.n_tagged::DOUBLE / c.n_group)
              / (ct.n_tagged::DOUBLE / nc.n)
    END                                                  AS lift
FROM combined c
CROSS JOIN n_control nc
LEFT JOIN control_tags ct ON ct.tag = c.tag
ORDER BY
    c.view,
    c.event_class,
    c.verdict NULLS LAST,
    c.review_decile NULLS LAST,
    c.n_tagged DESC,
    c.tag;
