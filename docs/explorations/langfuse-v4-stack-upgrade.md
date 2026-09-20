# Langfuse v4 stack upgrade

Research date: 2026-09-20

## Recommendation

Upgrade the arena's disposable infrastructure as one unit to Langfuse v4.38.0 and the component set below. This follows the current [Langfuse v4.38.0 Compose file](https://github.com/langfuse/langfuse/blob/v4.38.0/docker-compose.yml), while keeping every image immutable for CI.

| Service | Recommended image |
| --- | --- |
| Langfuse web | `docker.langfuse.com/langfuse/langfuse:4.38.0@sha256:47ef2f121e2959c8458c209d118949ade25778017e3f5b75b1e45f72276811c3` |
| Langfuse worker | `docker.langfuse.com/langfuse/langfuse-worker:4.38.0@sha256:8631cf429efc4a2981d4e6ced4c005b9f6f620a4e34625baf60152301ec6d006` |
| ClickHouse | `clickhouse/clickhouse-server:25.12@sha256:8a790dd3468db22b1d4e7b18a176f378ff5ff6053b9c48dd4ea1fa71a24c5ba6` |
| PostgreSQL | `postgres:17.11@sha256:f4c66b820c6f974249089d3d16d86a3698eae11e8746eb6644b2271031e91232` |
| Redis | `redis:7.4.11@sha256:c6eabf748fc7a61dbb5a705c78bcf3d6377b1127a97d0ce965c11c44ba46896f` |
| MinIO | `cgr.dev/chainguard/minio:latest@sha256:22bf0ad174612a0106dfbbb95818f307525dd32c5a6ec119ecd7273815abe456` |
| OTel Collector Contrib | `otel/opentelemetry-collector-contrib:0.161.0@sha256:fd328de2552466ad78385e1b1289c3f2402b1c45f265b252aab1955b42845ac1` |

The digests above are the multi-architecture manifest digests resolved from their official registries on the research date. Langfuse v4.38.0 was the [latest Langfuse release](https://github.com/langfuse/langfuse/releases/tag/v4.38.0), PostgreSQL 17.11 was the [current PostgreSQL 17 patch](https://www.postgresql.org/docs/17/release-17-11.html), and OTel Collector 0.161.0 was the [latest Collector distribution release](https://github.com/open-telemetry/opentelemetry-collector-releases/releases/tag/v0.161.0).

ClickHouse 25.12 is the conservative CI choice because it exactly matches Langfuse's current Compose deployment. Langfuse requires ClickHouse 25.12 or newer and recommends 26.4. ClickHouse 26.8 LTS is now security-supported and also satisfies that version constraint, so a security-first deployment can instead pin `clickhouse/clickhouse-server:26.8@sha256:cc7f5901580ec744d75c62817a226c61e67403c3b268d417193d212c843c0277`. The trade-off is using a version newer than Langfuse's checked-in Compose fixture. See [Langfuse's ClickHouse requirements](https://langfuse.com/self-hosting/deployment/infrastructure/clickhouse) and the [ClickHouse security support matrix](https://github.com/ClickHouse/ClickHouse/security/policy).

## Delta from this repository

| Service | Current | Required change |
| --- | --- | --- |
| Langfuse | v3 images from Docker Hub | Move web and worker together to v4.38.0 from `docker.langfuse.com`. |
| ClickHouse | 24.3 | Upgrade before Langfuse v4; 24.3 is below v4's 25.12 minimum. |
| PostgreSQL | 15 | Use 17.11 for a fresh stack; do not attach an existing PostgreSQL 15 data volume directly. |
| Redis | floating major `7` at an old digest | Pin the current upstream-resolved Redis 7 release, 7.4.11, and add `--maxmemory-policy noeviction`. |
| MinIO | archived/unavailable `minio/minio` Docker Hub path | Use the Chainguard image selected by upstream Langfuse and create the `langfuse` bucket at startup. |
| OTel Collector | 0.112.0 | Upgrade independently to 0.161.0; Langfuse's Compose file does not include a collector. |

The current OTel configuration still has all of its required components in the [0.161.0 Contrib distribution manifest](https://github.com/open-telemetry/opentelemetry-collector-releases/blob/v0.161.0/distributions/otelcol-contrib/manifest.yaml): OTLP receiver, batch processor, debug exporter, and health-check extension.

## Compose changes that matter

1. Reuse a single environment mapping for web and worker so database, ClickHouse, Redis, and object-storage settings cannot drift. The [official Compose file](https://github.com/langfuse/langfuse/blob/v4.38.0/docker-compose.yml) uses a YAML anchor for this.
2. Keep `CLICKHOUSE_CLUSTER_ENABLED=false` for this single-node deployment. The built-in `default` user is sufficient here; externally managed restricted users need the additional v4 DDL and system-table grants documented in the [v3-to-v4 guide](https://langfuse.com/self-hosting/upgrade/upgrade-guides/upgrade-v3-to-v4).
3. Add UTC settings to PostgreSQL (`TZ=UTC`, `PGTZ=UTC`). Langfuse requires PostgreSQL and ClickHouse infrastructure to run in UTC, as described in its [ClickHouse deployment guide](https://langfuse.com/self-hosting/deployment/infrastructure/clickhouse).
4. Change Redis to `--requirepass ... --maxmemory-policy noeviction`, matching upstream.
5. Replace the MinIO command with upstream's bucket-creating lifecycle:

   ```yaml
   entrypoint: sh
   command: -c 'mkdir -p /data/langfuse && minio server --address ":9000" --console-address ":9001" /data'
   ```

   The full-stack recommendation is therefore **Chainguard, not Quay**. `quay.io/minio/minio:RELEASE.2025-04-22T22-12-26Z` is a valid minimal repair for the failed Docker Hub pull, but current Langfuse deliberately uses `cgr.dev/chainguard/minio`. Chainguard documents that registry path in its [MinIO image catalog](https://images.chainguard.dev/directory/image/minio/overview).
6. Add `:-` empty defaults to optional `LANGFUSE_INIT_*` interpolation, for example `${LANGFUSE_INIT_ORG_ID:-}`. That is how upstream avoids Compose warnings when initialization is intentionally omitted.
7. For v4 migration monitoring, change the worker health check to `/api/health?failIfEventPropagationStuck=true`; Langfuse recommends this endpoint because it returns 503 when dual-write event propagation stops. See the [v4 migration guide](https://langfuse.com/self-hosting/upgrade/upgrade-guides/upgrade-v3-to-v4).

The arena's collector currently exports traces only to `debug`, not Langfuse. Upgrading the collector preserves that behavior; forwarding OTLP to Langfuse remains separate work.

## State and migration decision

GitHub Actions uses new tmpfs volumes, so it can start directly on v4 with the default `events_only` model. No v3 bridge or historic backfill is needed.

For this repository's developer stack, the simplest non-backward-compatible path is an explicit destructive reset of the four named volumes before first v4 startup. That avoids two independent in-place migrations:

- Langfuse requires ClickHouse to be upgraded first, the latest v3 background migrations to be complete, PostgreSQL and ClickHouse backups, then a controlled `legacy`, `dual`, or `events_only` transition. The v4 migration is not automatically reversible. See the complete [Langfuse v3-to-v4 migration guide](https://langfuse.com/self-hosting/upgrade/upgrade-guides/upgrade-v3-to-v4).
- PostgreSQL major-version data directories are not binary-compatible. PostgreSQL requires dump/restore, `pg_upgrade`, or logical replication to retain PostgreSQL 15 data when moving to 17. See [Upgrading a PostgreSQL cluster](https://www.postgresql.org/docs/17/upgrading.html).

If local Langfuse history must be retained, do not apply the direct stack replacement. First bridge to the latest v3 release, verify that `background_migrations` has no unfinished rows, back up both databases, upgrade ClickHouse, migrate PostgreSQL separately, and only then deploy v4 in `dual` or `legacy` mode. ClickHouse also recommends downtime or an intermediate version when jumping more than one year, which applies to 24.3 to 25.12/26.x; see its [self-managed upgrade guidance](https://clickhouse.com/docs/guides/oss/update).

## Validation targets

- `docker compose ... config --images` contains no `minio/minio` or unpinned image.
- CI's generated `.env` produces no missing-variable warnings.
- A clean-volume `docker compose up -d` makes PostgreSQL, ClickHouse, Redis, MinIO, Langfuse worker, and Langfuse web healthy.
- Langfuse startup logs show successful PostgreSQL and ClickHouse schema migrations.
- The MinIO `langfuse` bucket exists after startup.
- The OTel Collector accepts OTLP/HTTP on port 4318 and starts the existing debug trace pipeline.
