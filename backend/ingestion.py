"""
GTFS Data Ingestion Module
==========================
Fetches and parses GTFS-RT (protobuf) and static GTFS data from:
  - Local parquet files (feed_data/ directory, downloaded from MinIO)
  - Live feed URLs (gtfs.ovapi.nl)
  - Static GTFS zip (gtfs.ovapi.nl/gtfs-nl.zip)

Produces merged DataFrames compatible with DisruptionFeatureBuilder.
"""
import io
import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_FEED_URLS = {
    "alerts": "http://gtfs.ovapi.nl/nl/alerts.pb",
    "vehiclePositions": "http://gtfs.ovapi.nl/nl/vehiclePositions.pb",
    "tripUpdates": "http://gtfs.ovapi.nl/nl/tripUpdates.pb",
}
DEFAULT_STATIC_GTFS_URL = "http://gtfs.ovapi.nl/gtfs-nl.zip"
DEFAULT_LOCAL_DIR = "feed_data"

ALERT_CAUSE_MAP = {
    1: "UNKNOWN_CAUSE", 2: "OTHER_CAUSE", 3: "TECHNICAL_PROBLEM",
    4: "STRIKE", 5: "DEMONSTRATION", 6: "ACCIDENT", 7: "HOLIDAY",
    8: "WEATHER", 9: "MAINTENANCE", 10: "CONSTRUCTION",
    11: "POLICE_ACTIVITY", 12: "MEDICAL_EMERGENCY",
}

ALERT_EFFECT_MAP = {
    1: "NO_SERVICE", 2: "REDUCED_SERVICE", 3: "SIGNIFICANT_DELAYS",
    4: "DETOUR", 5: "ADDITIONAL_SERVICE", 6: "MODIFIED_SERVICE",
    7: "OTHER_EFFECT", 8: "UNKNOWN_EFFECT", 9: "STOP_MOVED",
    10: "NO_EFFECT", 11: "ACCESSIBILITY_ISSUE",
}


# ---------------------------------------------------------------------------
# Protobuf parsers
# ---------------------------------------------------------------------------

def _parse_vehicle_positions(feed: gtfs_realtime_pb2.FeedMessage) -> pd.DataFrame:
    """Parse a GTFS-RT VehiclePositions feed into a DataFrame."""
    rows = []
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle
        trip = v.trip
        pos = v.position
        rows.append({
            "entity_id": entity.id,
            "trip_id": trip.trip_id or None,
            "route_id": trip.route_id or None,
            "direction_id": trip.direction_id if trip.direction_id else np.nan,
            "start_time": trip.start_time or None,
            "start_date": trip.start_date or None,
            "schedule_relationship": int(trip.schedule_relationship),
            "vehicle_id": v.vehicle.id or None,
            "vehicle_label": v.vehicle.label or None,
            "license_plate": v.vehicle.license_plate or None,
            "wheelchair_accessible": None,
            "latitude": pos.latitude if pos.latitude else np.nan,
            "longitude": pos.longitude if pos.longitude else np.nan,
            "bearing": pos.bearing if pos.bearing else np.nan,
            "odometer": pos.odometer if pos.odometer else np.nan,
            "speed": pos.speed if pos.speed else np.nan,
            "current_stop_sequence": float(v.current_stop_sequence) if v.current_stop_sequence else np.nan,
            "stop_id": v.stop_id or None,
            "current_status": int(v.current_status),
            "timestamp": int(v.timestamp) if v.timestamp else int(feed.header.timestamp),
            "congestion_level": str(v.congestion_level) if v.congestion_level else None,
            "occupancy_status": str(v.occupancy_status) if v.occupancy_status else None,
            "occupancy_percentage": str(v.occupancy_percentage) if v.occupancy_percentage else None,
            "multi_carriage_details": None,
            "retrieved_at": pd.Timestamp.now(),
        })
    return pd.DataFrame(rows)


def _parse_trip_updates(feed: gtfs_realtime_pb2.FeedMessage) -> pd.DataFrame:
    """Parse a GTFS-RT TripUpdates feed into a flattened DataFrame."""
    rows = []
    header_ts = feed.header.timestamp
    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        trip = tu.trip
        for stu in tu.stop_time_update:
            row = {
                "entity_id": entity.id,
                "trip_id": trip.trip_id or None,
                "route_id": trip.route_id or None,
                "direction_id": trip.direction_id if trip.direction_id else np.nan,
                "start_time": trip.start_time or None,
                "start_date": trip.start_date or None,
                "schedule_relationship": int(trip.schedule_relationship),
                "stop_sequence": int(stu.stop_sequence),
                "stop_id": stu.stop_id or None,
                "arrival_delay": int(stu.arrival.delay) if stu.arrival.HasField("delay") else None,
                "arrival_time": int(stu.arrival.time) if stu.arrival.HasField("time") else None,
                "departure_delay": int(stu.departure.delay) if stu.departure.HasField("delay") else None,
                "departure_time": int(stu.departure.time) if stu.departure.HasField("time") else None,
                "schedule_relationship_stu": int(stu.schedule_relationship),
                "timestamp": int(header_ts),
                "retrieved_at": pd.Timestamp.now(),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _parse_alerts(feed: gtfs_realtime_pb2.FeedMessage) -> pd.DataFrame:
    """Parse a GTFS-RT Alerts feed into a DataFrame."""
    rows = []
    header_ts = feed.header.timestamp
    for entity in feed.entity:
        if not entity.HasField("alert"):
            continue
        a = entity.alert
        cause = ALERT_CAUSE_MAP.get(int(a.cause), "UNKNOWN_CAUSE")
        effect = ALERT_EFFECT_MAP.get(int(a.effect), "UNKNOWN_EFFECT")
        header = ""
        if a.header_text.translation:
            header = a.header_text.translation[0].text
        description = ""
        if a.description_text.translation:
            description = a.description_text.translation[0].text
        url = ""
        if a.url.translation:
            url = a.url.translation[0].text
        
        # Extract informed entities
        informed_entities = []
        for ie in a.informed_entity:
            ie_dict = {
                "agency_id": ie.agency_id or None,
                "route_id": ie.route_id or None,
                "route_type": ie.route_type if ie.route_type else None,
                "trip_id": ie.trip.trip_id or None,
                "stop_id": ie.stop_id or None,
            }
            informed_entities.append(ie_dict)
        
        row = {
            "entity_id": entity.id,
            "cause": cause,
            "effect": effect,
            "header": header,
            "description": description,
            "url": url,
            "active_period_start": int(a.active_period[0].start) if a.active_period else None,
            "active_period_end": int(a.active_period[0].end) if a.active_period else None,
            "informed_entities": json.dumps(informed_entities),
            "timestamp": int(header_ts),
            "retrieved_at": pd.Timestamp.now(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _read_parquet_from_zip(zip_path: str, max_files: Optional[int] = None) -> pd.DataFrame:
    """Read parquet files from a zip archive and concatenate them.

    Parameters
    ----------
    zip_path : str
        Path to the zip file.
    max_files : int, optional
        Maximum number of parquet files to read (for memory control).
        Files are sampled evenly across the sorted list to maximise
        temporal coverage.  If None, reads all files.
    """
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        parquet_names = sorted(n for n in z.namelist() if n.endswith(".parquet"))
        if max_files is not None and max_files < len(parquet_names):
            # Sample evenly across the full list for temporal spread
            indices = np.linspace(0, len(parquet_names) - 1, max_files, dtype=int)
            parquet_names = [parquet_names[i] for i in indices]
        for name in parquet_names:
            frames.append(pd.read_parquet(io.BytesIO(z.read(name))))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_local_feeds(
    local_dir: str = DEFAULT_LOCAL_DIR,
    max_files: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load GTFS data from local zip files in *local_dir*.

    Expected zip files (in filename-alphabetical order):
      1st zip -> vehicle positions
      2nd zip -> trip updates
      3rd zip -> alerts

    Parameters
    ----------
    local_dir : str
        Path to directory containing *_files_list.zip files.
    max_files : int, optional
        Max parquet files to read per zip (for memory control).

    Returns
    -------
    Dict with keys 'vehicle_positions', 'trip_updates', 'alerts'.
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Local feed directory not found: {local_dir}")

    zip_files = sorted(local_path.glob("*_files_list.zip"))
    if len(zip_files) < 3:
        raise ValueError(
            f"Expected at least 3 *_files_list.zip files in {local_dir}, "
            f"found {len(zip_files)}"
        )

    labels = ["vehicle_positions", "trip_updates", "alerts"]
    result = {}
    for label, zf in zip(labels, zip_files[:3]):
        logger.info(f"Loading {label} from {zf.name}...")
        df = _read_parquet_from_zip(str(zf), max_files=max_files)
        logger.info(f"  Loaded {len(df)} rows x {len(df.columns)} cols")
        result[label] = df
    
    # Auto-load static GTFS data
    logger.info("Auto-loading static GTFS data...")
    try:
        # Try to load from cached zip or data directory
        import os
        static_path = "gtfs-nl.zip"
        if not os.path.exists(static_path):
            static_path = os.path.join("..", "data", "gtfs-nl.zip")
        if os.path.exists(static_path):
            static_data = load_static_gtfs_from_zip(static_path)
            result.update(static_data)
            logger.info("  Static GTFS data loaded from zip")
        else:
            logger.warning("  Static GTFS zip not found, skipping")
    except Exception as e:
        logger.warning(f"  Failed to load static GTFS: {e}")
    
    return result


def fetch_all_live_feeds(urls: Dict[str, str] = DEFAULT_FEED_URLS) -> Dict[str, pd.DataFrame]:
    """
    Fetch and parse all live GTFS-RT feeds.

    Parameters
    ----------
    urls : dict
        URLs for vehiclePositions, tripUpdates, alerts.

    Returns
    -------
    Dict with keys 'vehicle_positions', 'trip_updates', 'alerts'.
    """
    result = {}
    for feed_type, url in urls.items():
        logger.info(f"Fetching {feed_type} from {url}...")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(resp.content)
            
            if feed_type == "vehiclePositions":
                df = _parse_vehicle_positions(feed)
            elif feed_type == "tripUpdates":
                df = _parse_trip_updates(feed)
            elif feed_type == "alerts":
                df = _parse_alerts(feed)
            else:
                continue
            
            logger.info(f"  Parsed {len(df)} records")
            result[feed_type] = df
        except Exception as e:
            logger.warning(f"  Failed to fetch {feed_type}: {e}")
            result[feed_type] = pd.DataFrame()
    
    return result


def ingest_live(feed_type: str, url: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch and parse a single live GTFS-RT feed.
    
    Parameters
    ----------
    feed_type : str
        One of 'vehicle_positions', 'trip_updates', 'alerts'.
    url : str, optional
        Custom URL. If None, uses default.
        
    Returns
    -------
    pd.DataFrame
    """
    if url is None:
        url = DEFAULT_FEED_URLS.get(feed_type, "")
    if not url:
        raise ValueError(f"No URL for feed_type: {feed_type}")
    
    logger.info(f"Fetching {feed_type} from {url}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)
    
    if feed_type == "vehicle_positions":
        return _parse_vehicle_positions(feed)
    elif feed_type == "trip_updates":
        return _parse_trip_updates(feed)
    elif feed_type == "alerts":
        return _parse_alerts(feed)
    else:
        raise ValueError(f"Unknown feed_type: {feed_type}")


def ingest_local(feed_type: str, local_dir: str = DEFAULT_LOCAL_DIR, max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Load a single GTFS feed from local zip files.
    
    Parameters
    ----------
    feed_type : str
        One of 'vehicle_positions', 'trip_updates', 'alerts'.
    local_dir : str
        Directory with zip files.
    max_files : int, optional
        Max parquet files per zip.
        
    Returns
    -------
    pd.DataFrame
    """
    feeds = load_local_feeds(local_dir, max_files)
    return feeds.get(feed_type, pd.DataFrame())


def ingest_combined(feed_type: str, local_dir: str = DEFAULT_LOCAL_DIR, 
                   live_url: Optional[str] = None, max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Combine local and live data for a feed type.
    
    Parameters
    ----------
    feed_type : str
        One of 'vehicle_positions', 'trip_updates', 'alerts'.
    local_dir : str
        Local data directory.
    live_url : str, optional
        Live feed URL.
    max_files : int, optional
        Max files for local data.
        
    Returns
    -------
    pd.DataFrame
    """
    local_df = ingest_local(feed_type, local_dir, max_files)
    live_df = ingest_live(feed_type, live_url)
    
    if local_df.empty:
        return live_df
    if live_df.empty:
        return local_df
    
    # Combine and sort by timestamp
    combined = pd.concat([local_df, live_df], ignore_index=True)
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def merge_feed_data(
    vehicle_positions: pd.DataFrame,
    trip_updates: pd.DataFrame,
    alerts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge vehicle positions, trip updates, and alerts into a single DataFrame.

    The primary join key is (trip_id, route_id).  Trip-update stop-level
    columns are prefixed ``tu_`` to avoid collisions.  Alert columns are
    prefixed ``alert_`` where necessary.
    """
    logger.info("Merging feed data sources...")

    vp = vehicle_positions.copy()
    tu = trip_updates.copy()
    al = alerts.copy()

    # --- standardise key columns ---
    for df in (vp, tu):
        for col in ("trip_id", "route_id", "stop_id"):
            if col in df.columns:
                df[col] = df[col].astype(str)

    if "trip_id" in al.columns:
        al["trip_id"] = al["trip_id"].astype(str)

    # --- merge VP + TU ---
    try:
        # Build subset of TU columns not present in VP (but ensure trip_id is kept if present)
        tu_extra = [c for c in tu.columns if (c not in vp.columns) or c == "trip_id"] if not tu.empty else []
        # If trip_id exists, prefer joining on trip_id; otherwise fall back to route_id
        if "trip_id" in vp.columns and "trip_id" in tu.columns:
            if tu_extra:
                # Keep trip_id for dedupe key if present
                if "trip_id" not in tu_extra:
                    tu_extra.append("trip_id")
                tu_sub = tu[tu_extra].drop_duplicates(subset=["trip_id"]) if "trip_id" in tu.columns else tu
            else:
                # No extra columns, dedupe on trip_id across full TU
                tu_sub = tu.drop_duplicates(subset=["trip_id"]) if "trip_id" in tu.columns else tu
            merged = vp.merge(tu_sub, on="trip_id", how="left", suffixes=("", "_tu"))
        elif "route_id" in vp.columns and "route_id" in tu.columns:
            # Fall back to merging on route_id when trip_id is not available
            tu_extra = [c for c in tu.columns if (c not in vp.columns) or c == "route_id"] if not tu.empty else []
            if tu_extra and "route_id" not in tu_extra:
                tu_extra.append("route_id")
            tu_sub = tu[tu_extra].drop_duplicates(subset=["route_id"]) if "route_id" in tu.columns else tu
            merged = vp.merge(tu_sub, on="route_id", how="left", suffixes=("", "_tu"))
        else:
            # Nothing sensible to join on; attach TU columns as empty columns to VP
            logger.warning("No trip_id or route_id available in both feeds; returning vehicle positions with placeholder TU columns")
            merged = vp.copy()
            for c in tu.columns:
                merged[c] = np.nan
    except Exception as e:
        logger.exception(f"Failed to merge VP and TU: {e}")
        # Fallback to returning vehicle positions alone
        merged = vp.copy()

    # --- derive feed_timestamp ---
    if "retrieved_at" in merged.columns:
        merged["feed_timestamp"] = pd.to_datetime(merged["retrieved_at"])
    elif "timestamp" in merged.columns:
        merged["feed_timestamp"] = pd.to_datetime(merged["timestamp"], unit="s", errors="coerce")
    else:
        merged["feed_timestamp"] = pd.Timestamp.now()

    # --- compute delay_sec from trip updates ---
    if "arrival_delay" in merged.columns:
        merged["delay_sec"] = merged["arrival_delay"].astype(float)
    elif "delay" in merged.columns:
        merged["delay_sec"] = merged["delay"].astype(float)
    else:
        merged["delay_sec"] = np.nan

    # --- derive scheduled / actual time columns ---
    def _parse_gtfs_time(t):
        """Parse GTFS time string (HH:MM:SS) to seconds from midnight."""
        if pd.isna(t):
            return np.nan
        if isinstance(t, (int, float)):
            return float(t)
        try:
            parts = str(t).split(":")
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except:
            return np.nan
    
    if "arrival_time" in merged.columns:
        if merged["arrival_time"].dtype == object:
            merged["actual_time_sec"] = merged["arrival_time"].apply(_parse_gtfs_time)
        else:
            merged["actual_time_sec"] = merged["arrival_time"].astype(float)
    if "departure_time" in merged.columns:
        # use departure as actual if arrival missing
        if "actual_time_sec" not in merged.columns:
            if merged["departure_time"].dtype == object:
                merged["actual_time_sec"] = merged["departure_time"].apply(_parse_gtfs_time)
            else:
                merged["actual_time_sec"] = merged["departure_time"].astype(float)

    # --- merge alerts by route_id ---
    # Live alerts store route info in informed_entities JSON; local parquets may
    # have a direct route_id column.  Try both approaches.
    if "route_id" in al.columns and "route_id" in merged.columns:
        # Direct merge on route_id
        al_sub = al[["route_id", "cause", "effect", "header", "description", "url"]].drop_duplicates(subset=["route_id"], keep="first")
        merged = merged.merge(al_sub, on="route_id", how="left", suffixes=("", "_alert"))
    elif "informed_entities" in al.columns:
        # Expand JSON and merge on route_id
        al_expanded = []
        for _, row in al.iterrows():
            entities = json.loads(row["informed_entities"])
            for entity in entities:
                al_expanded.append({
                    "route_id": entity.get("route_id"),
                    "cause": row["cause"],
                    "effect": row["effect"],
                    "header": row["header"],
                    "description": row["description"],
                    "url": row["url"],
                })
        al_df = pd.DataFrame(al_expanded).dropna(subset=["route_id"])
        al_sub = al_df.drop_duplicates(subset=["route_id"], keep="first")
        merged = merged.merge(al_sub, on="route_id", how="left", suffixes=("", "_alert"))
    else:
        logger.warning("Could not merge alerts - no route_id or informed_entities column")

    # --- deduplicate ---
    # Remove exact duplicates, preferring most recent retrieved_at
    if "retrieved_at" in merged.columns:
        merged = merged.sort_values("retrieved_at").drop_duplicates(
            subset=[c for c in merged.columns if c != "retrieved_at"], keep="last"
        )
    else:
        merged = merged.drop_duplicates()

    logger.info(f"Merged data: {len(merged)} rows x {len(merged.columns)} cols")
    return merged


def load_static_gtfs_from_zip(zip_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load static GTFS data from a zip file.
    
    Returns dict with keys: 'routes', 'stops', 'trips', 'stop_times', 'agency', etc.
    """
    result = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".txt"):
                continue
            table_name = name.replace(".txt", "")
            df = pd.read_csv(io.BytesIO(z.read(name)), dtype=str)
            result[table_name] = df
    return result