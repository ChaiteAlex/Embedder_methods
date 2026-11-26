from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SAMPLE_ROWS = 5
SAMPLE_DERIVED = 5
SAMPLE_NAMES_PER_TYPE = 20


# ---------------------------------------------------------------------------
# Общие утилиты
# ---------------------------------------------------------------------------


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json_value(v) for v in value]
    try:
        if hasattr(value, "as_py"):
            return safe_json_value(value.as_py())
    except Exception:
        pass
    return str(value)


def nonempty_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def add_sample(samples: list[dict[str, Any]], item: dict[str, Any], limit: int = SAMPLE_DERIVED) -> None:
    if len(samples) < limit:
        samples.append(safe_json_value(item))


def sample_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def empty_derived(note: str | None = None) -> dict[str, Any]:
    result = {
        "pairs": {"count": 0, "by_type": {}, "samples": []},
        "embedding_triplets": {"count": 0, "samples": []},
        "graph_triplets": {"count": 0, "by_relation": {}, "samples": []},
        "relationships": {"count": 0, "by_type": {}, "samples": []},
    }
    if note:
        result["note"] = note
    return result


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------


def parquet_schema_summary(parquet: Any) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    metadata = parquet.metadata
    schema_arrow = parquet.schema_arrow
    null_counts: dict[str, int | None] = {}
    columns: list[dict[str, Any]] = []

    for field_index, field in enumerate(schema_arrow):
        total_nulls = 0
        complete = True
        for rg_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(rg_index)
            if field_index >= row_group.num_columns:
                complete = False
                break
            try:
                stats = row_group.column(field_index).statistics
            except Exception:
                complete = False
                break
            if stats is None or stats.null_count is None:
                complete = False
                break
            total_nulls += int(stats.null_count)
        null_counts[field.name] = total_nulls if complete else None

    for field in schema_arrow:
        null_count = null_counts.get(field.name)
        null_percent = None
        if null_count is not None and metadata.num_rows:
            null_percent = round((null_count / metadata.num_rows) * 100, 4)
        columns.append({
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
            "null_count": null_count,
            "null_percent": null_percent,
        })
    return columns, null_counts


def iter_parquet_rows(parquet: Any, columns: list[str] | None = None, batch_size: int = 2048) -> Iterable[dict[str, Any]]:
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            yield row


def detect_cypher_columns(column_names: list[str]) -> dict[str, str | None] | None:
    names = set(column_names)
    if not {"query_cmd", "positive_cmd"}.issubset(names):
        return None
    negative_candidates = [
        "negative_index_list", "negative_indices", "negative_index",
        "negative_cmds", "negative_cmd", "negatives", "negative",
    ]
    negative_field = next((n for n in negative_candidates if n in names), None)
    idx_field = next((n for n in ["idx", "index", "id"] if n in names), None)
    return {"query": "query_cmd", "positive": "positive_cmd",
            "negative": negative_field, "idx": idx_field}


def detect_fstec_columns(column_names: list[str]) -> dict[str, str | None] | None:
    names = set(column_names)
    has_bdu = any("bdu" in n.lower() or n.lower() == "id" for n in names)
    has_desc = any("description" in n.lower() or "описание" in n.lower() for n in names)
    if not (has_bdu and has_desc):
        return None

    def pick(cands: list[str]) -> str | None:
        return next((c for c in cands if c in names), None)

    return {
        "id": pick(["id", "bdu_id", "BDU"]),
        "title": pick(["title", "name", "название"]),
        "description": pick(["description", "description_ru", "описание", "desc"]),
        "cve": pick(["cve_id", "cve", "cves"]),
        "year": pick(["year", "identify_year", "год"]),
        "severity": pick(["severity", "severity_level", "критичность"]),
    }


def analyze_cypher_parquet(parquet: Any, fields: dict[str, str | None]) -> dict[str, Any]:
    query_field = fields["query"]
    positive_field = fields["positive"]
    negative_field = fields.get("negative")
    idx_field = fields.get("idx")

    positive_by_idx: dict[int, str] = {}
    positive_by_position: list[str | None] = []

    first_cols = [positive_field] + ([idx_field] if idx_field else [])
    for position, row in enumerate(iter_parquet_rows(parquet, first_cols)):
        positive = nonempty_text(row.get(positive_field))
        positive_by_position.append(positive)
        if idx_field:
            try:
                idx = int(row.get(idx_field))
            except (TypeError, ValueError):
                idx = position
        else:
            idx = position
        if positive is not None:
            positive_by_idx[idx] = positive

    pair_count = 0
    pair_types: Counter[str] = Counter()
    pair_samples: list[dict[str, Any]] = []
    emb_count = 0
    emb_samples: list[dict[str, Any]] = []
    gt_count = 0
    gt_types: Counter[str] = Counter()
    gt_samples: list[dict[str, Any]] = []
    rel_count = 0
    rel_types: Counter[str] = Counter()
    rel_samples: list[dict[str, Any]] = []
    unresolved_neg = 0
    rows_with_neg = 0
    neg_total = 0

    req = [query_field, positive_field] + ([negative_field] if negative_field else []) + ([idx_field] if idx_field else [])
    for position, row in enumerate(iter_parquet_rows(parquet, req)):
        query = nonempty_text(row.get(query_field))
        positive = nonempty_text(row.get(positive_field))
        if idx_field:
            try:
                row_idx = int(row.get(idx_field))
            except (TypeError, ValueError):
                row_idx = position
        else:
            row_idx = position

        if query and positive:
            pair_count += 1
            pair_types["query_positive"] += 1
            add_sample(pair_samples, {"type": "query_positive", "row_idx": row_idx,
                                      "left": query, "right": positive})
            gt_count += 1
            gt_types["semantically_similar_to"] += 1
            add_sample(gt_samples, {"subject": query, "predicate": "semantically_similar_to",
                                    "object": positive, "row_idx": row_idx})
            rel_count += 1
            rel_types["positive_similarity"] += 1
            add_sample(rel_samples, {"source": {"kind": "query_cmd", "row_idx": row_idx, "text": query},
                                     "type": "positive_similarity",
                                     "target": {"kind": "positive_cmd", "row_idx": row_idx, "text": positive}})

        if not negative_field:
            continue

        negs = ensure_list(row.get(negative_field))
        if negs:
            rows_with_neg += 1
        neg_total += len(negs)

        for raw in negs:
            neg_text = None
            neg_idx = None
            if isinstance(raw, str):
                stripped = raw.strip()
                try:
                    neg_idx = int(stripped)
                except ValueError:
                    neg_text = stripped or None
            else:
                try:
                    neg_idx = int(raw)
                except (TypeError, ValueError):
                    neg_text = nonempty_text(raw)
            if neg_idx is not None:
                neg_text = positive_by_idx.get(neg_idx)
                if neg_text is None and 0 <= neg_idx < len(positive_by_position):
                    neg_text = positive_by_position[neg_idx]
            if not neg_text:
                unresolved_neg += 1
                continue
            if query and positive:
                emb_count += 1
                add_sample(emb_samples, {"anchor": query, "positive": positive, "negative": neg_text,
                                         "row_idx": row_idx, "negative_idx": neg_idx})
            if query:
                gt_count += 1
                gt_types["semantically_dissimilar_to"] += 1
                add_sample(gt_samples, {"subject": query, "predicate": "semantically_dissimilar_to",
                                        "object": neg_text, "row_idx": row_idx, "negative_idx": neg_idx})
                rel_count += 1
                rel_types["negative_similarity"] += 1
                add_sample(rel_samples, {"source": {"kind": "query_cmd", "row_idx": row_idx, "text": query},
                                         "type": "negative_similarity",
                                         "target": {"kind": "positive_cmd_from_negative_index",
                                                    "idx": neg_idx, "text": neg_text}})

    notes = []
    if negative_field is None:
        notes.append("Поле негативов не найдено: embedding-триплеты построить нельзя.")
    elif neg_total == 0:
        notes.append("Поле негативов найдено, но оно пустое во всех строках.")
    if unresolved_neg:
        notes.append(f"Не удалось разрешить ссылок на негативы: {unresolved_neg}.")

    return {
        "dataset_kind": "cypher_command_similarity",
        "detected_fields": fields,
        "pairs": {"count": pair_count, "by_type": sample_counter(pair_types), "samples": pair_samples},
        "embedding_triplets": {"count": emb_count, "samples": emb_samples},
        "graph_triplets": {"count": gt_count, "by_relation": sample_counter(gt_types), "samples": gt_samples},
        "relationships": {"count": rel_count, "by_type": sample_counter(rel_types), "samples": rel_samples},
        "negative_processing": {
            "field": negative_field,
            "rows_with_negatives": rows_with_neg,
            "negative_references_total": neg_total,
            "resolved_negative_references": neg_total - unresolved_neg,
            "unresolved_negative_references": unresolved_neg,
        },
        "notes": notes,
    }


def analyze_fstec_parquet(parquet: Any, fields: dict[str, str | None]) -> dict[str, Any]:
    id_f = fields["id"]; title_f = fields["title"]; desc_f = fields["description"]
    cve_f = fields.get("cve"); year_f = fields.get("year")

    pair_count = 0
    pair_types: Counter[str] = Counter()
    pair_samples: list[dict[str, Any]] = []
    rows_total = 0
    rows_with_cve = 0
    cve_total = 0
    years: Counter[str] = Counter()

    req = [c for c in [id_f, title_f, desc_f, cve_f, year_f] if c]
    for row in iter_parquet_rows(parquet, req):
        rows_total += 1
        title = nonempty_text(row.get(title_f)) if title_f else None
        desc = nonempty_text(row.get(desc_f)) if desc_f else None
        cve_raw = row.get(cve_f) if cve_f else None
        year_raw = row.get(year_f) if year_f else None
        if title and desc:
            pair_count += 1
            pair_types["title_description"] += 1
            add_sample(pair_samples, {"type": "title_description",
                                      "bdu_id": row.get(id_f) if id_f else None,
                                      "left": title, "right": desc[:300] + ("…" if len(desc) > 300 else "")})
        if cve_raw:
            cves = [c for c in str(cve_raw).split(",") if c.strip().startswith("CVE-")]
            if cves:
                rows_with_cve += 1
                cve_total += len(cves)
        if year_raw:
            years[str(year_raw)] += 1

    return {
        "dataset_kind": "fstec_bdu",
        "detected_fields": fields,
        "rows_total": rows_total,
        "rows_with_cve": rows_with_cve,
        "cve_references_total": cve_total,
        "year_distribution": dict(sorted(years.items())),
        "pairs": {"count": pair_count, "by_type": sample_counter(pair_types), "samples": pair_samples},
        "embedding_triplets": {"count": 0, "samples": [],
                               "note": "Для ФСТЭК негативы добываются отдельно (другая CWE/тип уязвимости)."},
        "graph_triplets": {"count": 0, "by_relation": {}, "samples": []},
        "relationships": {"count": 0, "by_type": {}, "samples": []},
        "notes": [],
    }


def analyze_parquet(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Нужен pyarrow: pip install pyarrow") from exc

    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    schema = parquet.schema_arrow
    columns, _ = parquet_schema_summary(parquet)

    samples: list[dict[str, Any]] = []
    if metadata.num_rows > 0:
        for row in iter_parquet_rows(parquet, batch_size=SAMPLE_ROWS):
            samples.append({str(k): safe_json_value(v) for k, v in row.items()})
            if len(samples) >= SAMPLE_ROWS:
                break

    names = list(schema.names)
    cypher = detect_cypher_columns(names)
    fstec = detect_fstec_columns(names) if not cypher else None

    if cypher:
        derived = analyze_cypher_parquet(parquet, cypher)
    elif fstec:
        derived = analyze_fstec_parquet(parquet, fstec)
    else:
        derived = empty_derived("Parquet не похож ни на CyPHER, ни на ФСТЭК.")

    return {
        "format": "parquet",
        "rows": int(metadata.num_rows),
        "columns_count": len(names),
        "row_groups": int(metadata.num_row_groups),
        "columns": columns,
        "sample_rows": samples,
        "derived": derived,
    }


# ---------------------------------------------------------------------------
# SQLite (ФСТЭК BDU)
# ---------------------------------------------------------------------------


def analyze_sqlite(path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        if not tables:
            return {"format": "sqlite", "tables": [], "derived": empty_derived("В базе нет таблиц.")}

        main_table = None
        for t in tables:
            cur.execute(f"PRAGMA table_info({t})")
            cols = [r["name"] for r in cur.fetchall()]
            if any("bdu" in c.lower() for c in cols) and any("desc" in c.lower() for c in cols):
                main_table = t
                break
        if main_table is None:
            return {"format": "sqlite", "tables": tables,
                    "derived": empty_derived("Нет таблицы с BDU ID и описанием.")}

        cur.execute(f"PRAGMA table_info({main_table})")
        cols_info = [{"name": r["name"], "type": r["type"]} for r in cur.fetchall()]
        names = [c["name"] for c in cols_info]
        fields = detect_fstec_columns(names)
        if not fields:
            return {"format": "sqlite", "tables": tables, "main_table": main_table,
                    "derived": empty_derived("Колонки не распознаны как ФСТЭК.")}

        id_f = fields["id"]; title_f = fields["title"]; desc_f = fields["description"]
        cve_f = fields.get("cve"); year_f = fields.get("year")
        sel = [c for c in [id_f, title_f, desc_f, cve_f, year_f] if c]
        cur.execute(f"SELECT {', '.join(sel)} FROM {main_table}")

        pair_count = 0
        pair_types: Counter[str] = Counter()
        pair_samples: list[dict[str, Any]] = []
        rows_total = 0
        rows_with_cve = 0
        cve_total = 0
        years: Counter[str] = Counter()

        for row in cur:
            rows_total += 1
            d = dict(row)
            title = nonempty_text(d.get(title_f)) if title_f else None
            desc = nonempty_text(d.get(desc_f)) if desc_f else None
            cve_raw = d.get(cve_f) if cve_f else None
            year_raw = d.get(year_f) if year_f else None
            if title and desc:
                pair_count += 1
                pair_types["title_description"] += 1
                add_sample(pair_samples, {"type": "title_description",
                                          "bdu_id": d.get(id_f),
                                          "left": title,
                                          "right": desc[:300] + ("…" if len(desc) > 300 else "")})
            if cve_raw:
                cves = [c for c in str(cve_raw).split(",") if c.strip().startswith("CVE-")]
                if cves:
                    rows_with_cve += 1
                    cve_total += len(cves)
            if year_raw:
                years[str(year_raw)] += 1

        return {
            "format": "sqlite",
            "tables": tables,
            "main_table": main_table,
            "columns": cols_info,
            "derived": {
                "dataset_kind": "fstec_bdu_sqlite",
                "detected_fields": fields,
                "rows_total": rows_total,
                "rows_with_cve": rows_with_cve,
                "cve_references_total": cve_total,
                "year_distribution": dict(sorted(years.items())),
                "pairs": {"count": pair_count, "by_type": sample_counter(pair_types), "samples": pair_samples},
                "embedding_triplets": {"count": 0, "samples": [],
                                       "note": "Негативы для ФСТЭК добываются отдельно."},
                "graph_triplets": {"count": 0, "by_relation": {}, "samples": []},
                "relationships": {"count": 0, "by_type": {}, "samples": []},
                "notes": [],
            },
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# JSON / STIX
# ---------------------------------------------------------------------------


def is_stix_bundle(data: Any) -> bool:
    return (isinstance(data, dict) and isinstance(data.get("objects"), list)
            and (data.get("type") == "bundle"
                 or any(isinstance(o, dict) and "type" in o for o in data["objects"][:10])))


def get_external_id(obj: dict[str, Any]) -> str | None:
    refs = obj.get("external_references")
    if not isinstance(refs, list):
        return None
    for ref in refs:
        if isinstance(ref, dict):
            ext = nonempty_text(ref.get("external_id"))
            if ext:
                return ext
    return None


def object_display_name(obj: dict[str, Any]) -> str:
    name = nonempty_text(obj.get("name"))
    ext = get_external_id(obj)
    if name and ext:
        return f"{ext} — {name}"
    return name or ext or str(obj.get("id", "<unknown>"))


def object_embedding_text(obj: dict[str, Any]) -> str | None:
    parts = []
    if ext := get_external_id(obj):
        parts.append(ext)
    if name := nonempty_text(obj.get("name")):
        parts.append(name)
    if desc := nonempty_text(obj.get("description")):
        parts.append(desc)
    return " — ".join(parts) if parts else None


def get_tactics(obj: dict[str, Any]) -> list[str]:
    tactics = []
    for phase in ensure_list(obj.get("kill_chain_phases")):
        if isinstance(phase, dict):
            t = nonempty_text(phase.get("phase_name"))
            if t:
                tactics.append(t)
    return tactics


def analyze_stix_bundle(data: dict[str, Any]) -> dict[str, Any]:
    objects = [o for o in data.get("objects", []) if isinstance(o, dict)]
    by_id = {str(o.get("id")): o for o in objects if o.get("id")}

    type_counts: Counter[str] = Counter()
    revoked = 0
    deprecated = 0
    names_by_type: dict[str, list[str]] = defaultdict(list)
    ap_ext_ids: list[str] = []

    for obj in objects:
        t = str(obj.get("type", "<unknown>"))
        type_counts[t] += 1
        if obj.get("revoked") is True:
            revoked += 1
        if obj.get("x_mitre_deprecated") is True:
            deprecated += 1
        name = nonempty_text(obj.get("name"))
        if name and len(names_by_type[t]) < SAMPLE_NAMES_PER_TYPE:
            names_by_type[t].append(name)
        if t == "attack-pattern":
            if ext := get_external_id(obj):
                ap_ext_ids.append(ext)

    pair_count = 0
    pair_types: Counter[str] = Counter()
    pair_samples: list[dict[str, Any]] = []
    pair_seen: set[tuple[str, str, str]] = set()

    def record_pair(ptype: str, left: str | None, right: str | None, meta: dict | None = None) -> None:
        nonlocal pair_count
        if not left or not right:
            return
        key = (ptype, left, right)
        if key in pair_seen:
            return
        pair_seen.add(key)
        pair_count += 1
        pair_types[ptype] += 1
        item = {"type": ptype, "left": left, "right": right}
        if meta:
            item.update(meta)
        add_sample(pair_samples, item)

    for obj in objects:
        name = nonempty_text(obj.get("name"))
        desc = nonempty_text(obj.get("description"))
        if name and desc:
            record_pair("name_description", name, desc,
                        {"object_id": str(obj.get("id", "")), "object_type": str(obj.get("type", ""))})

    gt_count = 0
    gt_types: Counter[str] = Counter()
    gt_samples: list[dict[str, Any]] = []
    rel_count = 0
    rel_types: Counter[str] = Counter()
    rel_samples: list[dict[str, Any]] = []
    unresolved_src = 0
    unresolved_tgt = 0

    for rel in objects:
        if rel.get("type") != "relationship":
            continue
        src_ref = nonempty_text(rel.get("source_ref"))
        tgt_ref = nonempty_text(rel.get("target_ref"))
        rel_type = nonempty_text(rel.get("relationship_type")) or "related_to"
        src_obj = by_id.get(src_ref or "")
        tgt_obj = by_id.get(tgt_ref or "")
        if src_ref and src_obj is None:
            unresolved_src += 1
        if tgt_ref and tgt_obj is None:
            unresolved_tgt += 1

        src_label = object_display_name(src_obj) if src_obj else (src_ref or "<missing>")
        tgt_label = object_display_name(tgt_obj) if tgt_obj else (tgt_ref or "<missing>")

        gt_count += 1
        gt_types[rel_type] += 1
        add_sample(gt_samples, {"subject": src_label, "predicate": rel_type,
                                "object": tgt_label, "relationship_id": rel.get("id")})

        rel_count += 1
        rel_types[rel_type] += 1
        add_sample(rel_samples, {"id": rel.get("id"), "source_ref": src_ref, "source": src_label,
                                 "type": rel_type, "target_ref": tgt_ref, "target": tgt_label,
                                 "description": nonempty_text(rel.get("description"))})

        procedure = nonempty_text(rel.get("description"))
        if procedure and tgt_obj and tgt_obj.get("type") == "attack-pattern":
            record_pair("procedure_technique", procedure, object_embedding_text(tgt_obj), {
                "relationship_id": rel.get("id"), "relationship_type": rel_type, "target_ref": tgt_ref,
            })

    # Negative mining по тактикам: для attack-pattern из тактики X берём attack-pattern из тактики Y.
    emb_count = 0
    emb_samples: list[dict[str, Any]] = []
    attack_patterns = [o for o in objects
                       if o.get("type") == "attack-pattern"
                       and not o.get("revoked") and not o.get("x_mitre_deprecated")]
    by_tactic: dict[str, list[dict]] = defaultdict(list)
    for ap in attack_patterns:
        for t in get_tactics(ap):
            by_tactic[t].append(ap)

    random.seed(42)
    for ap in attack_patterns:
        tactics = get_tactics(ap)
        if not tactics:
            continue
        anchor_text = object_embedding_text(ap)
        if not anchor_text:
            continue
        other_tactics = [t for t in by_tactic if t not in tactics]
        if not other_tactics:
            continue
        neg_tactic = random.choice(other_tactics)
        neg_ap = random.choice(by_tactic[neg_tactic])
        neg_text = object_embedding_text(neg_ap)
        if neg_text and neg_text != anchor_text:
            emb_count += 1
            add_sample(emb_samples, {"anchor": anchor_text, "positive": anchor_text,
                                     "negative": neg_text,
                                     "note": "anchor=positive=техника; negative=техника из другой тактики"})

    return {
        "format": "stix_bundle",
        "bundle_id": data.get("id"),
        "total_objects": len(objects),
        "object_type_counts": dict(sorted(type_counts.items())),
        "revoked_objects": revoked,
        "deprecated_objects": deprecated,
        "active_estimated_objects": max(0, len(objects) - revoked - deprecated),
        "attack_pattern_external_ids_count": len(ap_ext_ids),
        "attack_pattern_external_ids_sample": ap_ext_ids[:SAMPLE_NAMES_PER_TYPE],
        "sample_names_by_type": dict(sorted(names_by_type.items())),
        "derived": {
            "dataset_kind": "mitre_attack_stix",
            "pairs": {"count": pair_count, "by_type": sample_counter(pair_types), "samples": pair_samples},
            "embedding_triplets": {"count": emb_count, "samples": emb_samples},
            "graph_triplets": {"count": gt_count, "by_relation": sample_counter(gt_types), "samples": gt_samples},
            "relationships": {"count": rel_count, "by_type": sample_counter(rel_types),
                              "samples": rel_samples,
                              "unresolved_source_refs": unresolved_src,
                              "unresolved_target_refs": unresolved_tgt},
        },
    }


def analyze_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if is_stix_bundle(data):
        return analyze_stix_bundle(data)
    return {"format": "json", "structure": {"type": type(data).__name__},
            "derived": empty_derived("JSON не STIX Bundle.")}


# ---------------------------------------------------------------------------
# JSONL (MITRE-пары и ФСТЭК-пары)
# ---------------------------------------------------------------------------


def analyze_jsonl(path: Path) -> dict[str, Any]:
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not rows:
        return {"format": "jsonl", "rows": 0, "derived": empty_derived("Пустой JSONL.")}

    first = rows[0]

    # --- MITRE-формат: event_text + technique_description + label ---
    if "event_text" in first and "technique_description" in first:
        event_to_labels: dict[str, set] = defaultdict(set)
        for r in rows:
            ev = nonempty_text(r.get("event_text"))
            lbl = nonempty_text(r.get("label"))
            if ev and lbl:
                event_to_labels[ev].add(lbl)

        pair_count = 0
        pair_types: Counter[str] = Counter()
        pair_samples: list[dict[str, Any]] = []
        labels: Counter[str] = Counter()
        tactics: Counter[str] = Counter()

        for r in rows:
            ev = nonempty_text(r.get("event_text"))
            tech = nonempty_text(r.get("technique_description"))
            lbl = nonempty_text(r.get("label"))
            tac = nonempty_text(r.get("tactics"))
            if ev and tech:
                pair_count += 1
                pair_types["event_technique"] += 1
                add_sample(pair_samples, {"type": "event_technique", "left": ev[:200],
                                          "right": tech[:200], "label": lbl, "tactics": tac})
            if lbl:
                labels[lbl] += 1
            if tac:
                tactics[tac] += 1

        multi_label = sum(1 for v in event_to_labels.values() if len(v) > 1)

        return {
            "format": "jsonl",
            "rows": len(rows),
            "dataset_kind": "mitre_event_technique",
            "derived": {
                "pairs": {"count": pair_count, "by_type": sample_counter(pair_types), "samples": pair_samples},
                "label_distribution": sample_counter(labels),
                "tactic_distribution": sample_counter(tactics),
                "multi_label_events": multi_label,
                "embedding_triplets": {"count": 0, "samples": [],
                                       "note": "Строятся отдельным скриптом с учётом tactics и parent_label."},
                "graph_triplets": {"count": 0, "by_relation": {}, "samples": []},
                "relationships": {"count": 0, "by_type": {}, "samples": []},
            },
        }

    # --- ФСТЭК-формат: description_ru + description_en + bdu_id + cve_id ---
    if "description_ru" in first and "bdu_id" in first:
        pair_count = 0
        pair_types: Counter[str] = Counter()
        pair_samples: list[dict[str, Any]] = []
        rows_with_null_cve = 0
        rows_with_empty_en = 0
        unique_bdu: set[str] = set()

        for r in rows:
            ru = nonempty_text(r.get("description_ru"))
            en = nonempty_text(r.get("description_en"))
            bdu = nonempty_text(r.get("bdu_id"))
            cve = r.get("cve_id")
            if bdu:
                unique_bdu.add(bdu)
            if ru and en:
                pair_count += 1
                pair_types["ru_en"] += 1
                add_sample(pair_samples, {"type": "ru_en", "left": ru[:200],
                                          "right": en[:200], "bdu_id": bdu, "cve_id": cve})
            if cve is None:
                rows_with_null_cve += 1
            if not en:
                rows_with_empty_en += 1

        return {
            "format": "jsonl",
            "rows": len(rows),
            "dataset_kind": "fstec_ru_en",
            "derived": {
                "pairs": {"count": pair_count, "by_type": sample_counter(pair_types), "samples": pair_samples},
                "unique_bdu_count": len(unique_bdu),
                "rows_with_null_cve": rows_with_null_cve,
                "rows_with_empty_en": rows_with_empty_en,
                "embedding_triplets": {"count": 0, "samples": [],
                                       "note": "Негативы для ФСТЭК: другая CWE/тип уязвимости."},
                "graph_triplets": {"count": 0, "by_relation": {}, "samples": []},
                "relationships": {"count": 0, "by_type": {}, "samples": []},
            },
        }

    return {"format": "jsonl", "rows": len(rows),
            "derived": empty_derived("Формат JSONL не распознан (нет event_text/technique_description и нет description_ru/bdu_id).")}


# ---------------------------------------------------------------------------
# Общий анализ
# ---------------------------------------------------------------------------


def analyze_file(path: Path) -> dict[str, Any]:
    stat = path.stat()
    base = {
        "file_name": path.name,
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "size_human": human_size(stat.st_size),
        "extension": path.suffix.lower(),
        "status": "ok",
    }
    try:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            details = analyze_parquet(path)
        elif suffix == ".json":
            details = analyze_json(path)
        elif suffix in (".jsonl", ".ndjson") or suffix == ".txt":
            details = analyze_jsonl(path)
        elif suffix in (".sqlite", ".db", ".sqlite3"):
            details = analyze_sqlite(path)
        else:
            details = {"format": "unsupported",
                       "message": f"Формат {suffix or '<без расширения>'} не поддерживается.",
                       "derived": empty_derived("Неподдерживаемый формат.")}
            base["status"] = "unsupported"
        base["analysis"] = details
        return base
    except Exception as exc:
        base["status"] = "error"
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base


def build_report(paths: list[Path]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in paths:
        print(f"Анализ: {path.name}")
        if not path.exists():
            files.append({"file_name": path.name, "path": str(path),
                          "status": "error", "error": "FileNotFoundError: файл не найден"})
            continue
        item = analyze_file(path)
        files.append(item)
        if item.get("status") == "ok":
            d = item.get("analysis", {}).get("derived", {})
            print("  пары={p}; embedding={e}; граф={g}; связи={r}".format(
                p=d.get("pairs", {}).get("count", 0),
                e=d.get("embedding_triplets", {}).get("count", 0),
                g=d.get("graph_triplets", {}).get("count", 0),
                r=d.get("relationships", {}).get("count", 0),
            ))

    statuses = Counter(i.get("status", "unknown") for i in files)
    total_pairs = total_emb = total_gt = total_rel = 0
    for i in files:
        d = i.get("analysis", {}).get("derived", {})
        total_pairs += int(d.get("pairs", {}).get("count", 0) or 0)
        total_emb += int(d.get("embedding_triplets", {}).get("count", 0) or 0)
        total_gt += int(d.get("graph_triplets", {}).get("count", 0) or 0)
        total_rel += int(d.get("relationships", {}).get("count", 0) or 0)

    return {
        "report_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "files_total": len(files),
            "files_ok": statuses.get("ok", 0),
            "files_with_errors": statuses.get("error", 0),
            "files_unsupported": statuses.get("unsupported", 0),
            "derived_totals": {"pairs": total_pairs, "embedding_triplets": total_emb,
                               "graph_triplets": total_gt, "relationships": total_rel},
        },
        "files": files,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def md_escape(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def add_derived_markdown(lines: list[str], derived: dict[str, Any]) -> None:
    lines.extend(["### Пары, триплеты и связи", ""])
    pairs = derived.get("pairs", {})
    emb = derived.get("embedding_triplets", {})
    gt = derived.get("graph_triplets", {})
    rel = derived.get("relationships", {})

    lines.extend(["| Результат | Количество |", "|---|---:|",
                  f"| Текстовые пары | {pairs.get('count', 0)} |",
                  f"| Embedding-триплеты | {emb.get('count', 0)} |",
                  f"| Графовые триплеты | {gt.get('count', 0)} |",
                  f"| Связи / рёбра | {rel.get('count', 0)} |", ""])

    if pairs.get("by_type"):
        lines.extend(["#### Пары по типам", "", "| Тип | Количество |", "|---|---:|"])
        for k, v in pairs["by_type"].items():
            lines.append(f"| {md_escape(k)} | {v} |")
        lines.append("")

    if gt.get("by_relation"):
        lines.extend(["#### Графовые отношения", "", "| Отношение | Количество |", "|---|---:|"])
        for k, v in gt["by_relation"].items():
            lines.append(f"| {md_escape(k)} | {v} |")
        lines.append("")

    if rel.get("by_type"):
        lines.extend(["#### Связи по типам", "", "| Тип связи | Количество |", "|---|---:|"])
        for k, v in rel["by_type"].items():
            lines.append(f"| {md_escape(k)} | {v} |")
        lines.append("")

    # Доп. поля (label_distribution, tactic_distribution, rows_with_null_cve и т.п.)
    extras = [(k, v) for k, v in derived.items()
              if k not in ("pairs", "embedding_triplets", "graph_triplets", "relationships",
                           "dataset_kind", "detected_fields", "notes", "note")]
    if extras:
        lines.extend(["#### Дополнительные метрики", ""])
        for k, v in extras:
            if isinstance(v, dict) and len(v) <= 25:
                lines.append(f"- **{md_escape(k)}**: " + ", ".join(f"{md_escape(kk)}={vv}" for kk, vv in v.items()))
            elif isinstance(v, (int, float, str)):
                lines.append(f"- **{md_escape(k)}**: {md_escape(v)}")
        lines.append("")

    notes = ensure_list(derived.get("notes"))
    if derived.get("note"):
        notes.append(derived.get("note"))
    if emb.get("note"):
        notes.append(emb.get("note"))
    if notes:
        lines.extend(["#### Примечания", ""])
        for n in notes:
            if n:
                lines.append(f"- {md_escape(n)}")
        lines.append("")

    for title, samples in [("Примеры пар", pairs.get("samples", [])),
                           ("Примеры embedding-триплетов", emb.get("samples", [])),
                           ("Примеры графовых триплетов", gt.get("samples", [])),
                           ("Примеры связей", rel.get("samples", []))]:
        if samples:
            lines.extend([f"#### {title}", "", "```json",
                          json.dumps(samples, ensure_ascii=False, indent=2), "```", ""])


def report_to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    s = report.get("summary", {})
    t = s.get("derived_totals", {})
    lines.extend(["# Анализ датасетов по информационной безопасности", "",
                  f"Сформировано: `{report.get('created_at_utc', '')}`", "",
                  "## Общая сводка", "",
                  f"- Всего файлов: **{s.get('files_total', 0)}**",
                  f"- Успешно: **{s.get('files_ok', 0)}**",
                  f"- Ошибок: **{s.get('files_with_errors', 0)}**",
                  f"- Неподдерживаемых: **{s.get('files_unsupported', 0)}**", "",
                  "### Итог по производным структурам", "",
                  f"- Текстовые пары: **{t.get('pairs', 0)}**",
                  f"- Embedding-триплеты: **{t.get('embedding_triplets', 0)}**",
                  f"- Графовые триплеты: **{t.get('graph_triplets', 0)}**",
                  f"- Связи: **{t.get('relationships', 0)}**", ""])

    for item in report.get("files", []):
        lines.extend(["---", "", f"## {md_escape(item.get('file_name', 'Без имени'))}", "",
                      f"- Статус: **{md_escape(item.get('status', 'unknown'))}**",
                      f"- Путь: `{md_escape(item.get('path', ''))}`"])
        if item.get("size_human"):
            lines.append(f"- Размер: **{md_escape(item['size_human'])}**")
        if item.get("status") == "error":
            lines.extend(["", "### Ошибка", "", f"`{md_escape(item.get('error', ''))}`", ""])
            continue

        analysis = item.get("analysis", {})
        fmt = analysis.get("format", "unknown")
        lines.extend([f"- Формат: **{md_escape(fmt)}**", ""])

        if fmt == "parquet":
            lines.extend(["### Структура Parquet", "",
                          f"- Строк: **{analysis.get('rows', 0)}**",
                          f"- Столбцов: **{analysis.get('columns_count', 0)}**",
                          f"- Row groups: **{analysis.get('row_groups', 0)}**", "",
                          "| Поле | Тип | Nullable | Null count | Null % |",
                          "|---|---|---:|---:|---:|"])
            for c in analysis.get("columns", []):
                lines.append(f"| {md_escape(c.get('name'))} | {md_escape(c.get('type'))} | "
                             f"{md_escape(c.get('nullable'))} | {md_escape(c.get('null_count'))} | "
                             f"{md_escape(c.get('null_percent'))} |")
            lines.append("")

        elif fmt == "stix_bundle":
            lines.extend(["### MITRE / STIX Bundle", "",
                          f"- Bundle ID: `{md_escape(analysis.get('bundle_id'))}`",
                          f"- Всего объектов: **{analysis.get('total_objects', 0)}**",
                          f"- Revoked: **{analysis.get('revoked_objects', 0)}**",
                          f"- Deprecated: **{analysis.get('deprecated_objects', 0)}**",
                          f"- Активных (оценка): **{analysis.get('active_estimated_objects', 0)}**", "",
                          "#### Объекты по типам", "", "| Тип | Количество |", "|---|---:|"])
            for k, v in analysis.get("object_type_counts", {}).items():
                lines.append(f"| {md_escape(k)} | {v} |")
            lines.append("")

        elif fmt == "sqlite":
            lines.extend(["### SQLite", "",
                          f"- Таблиц: **{len(analysis.get('tables', []))}**",
                          f"- Основная таблица: `{md_escape(analysis.get('main_table'))}`", ""])

        elif fmt == "jsonl":
            lines.extend(["### JSONL", "",
                          f"- Строк: **{analysis.get('rows', 0)}**",
                          f"- Тип датасета: `{md_escape(analysis.get('dataset_kind'))}`", ""])

        add_derived_markdown(lines, analysis.get("derived", empty_derived()))

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------


def write_text(path: Path, text: str, direct_write: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if direct_write:
        path.write_text(text, encoding="utf-8", newline="\n")
        return
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent), text=True)
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Анализирует Parquet (CyPHER / ФСТЭК), JSON (STIX), JSONL и SQLite ФСТЭК.")
    p.add_argument("inputs", nargs="+", help="Файлы для анализа")
    p.add_argument("-o", "--output", default="dataset_analysis.json",
                   help="Путь к JSON-отчёту. Markdown создаётся рядом с .md")
    p.add_argument("--direct-write", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [Path(p).expanduser() for p in args.inputs]
    json_path = Path(args.output).expanduser()
    if json_path.suffix.lower() != ".json":
        json_path = json_path.with_suffix(".json")
    md_path = json_path.with_suffix(".md")

    report = build_report(inputs)
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    md_text = report_to_markdown(report)

    write_text(json_path, json_text, args.direct_write)
    write_text(md_path, md_text, args.direct_write)

    print()
    print("Готово.")
    print(f"JSON: {json_path.resolve()}")
    print(f"Markdown: {md_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nПрервано.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nОшибка: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)