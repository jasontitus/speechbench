"""Thin gcloud / gsutil wrappers used by the orchestrator.

We deliberately shell out to `gcloud` and `gsutil` instead of using the python
client libraries on the orchestrator side — it keeps local install lightweight
and matches whatever credentials the user already has set up. The runner
(VM-side) also uses gsutil for results upload.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from typing import Optional


class GCloudError(RuntimeError):
    pass


def _run(cmd: list[str], *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=capture, text=True)
    if check and proc.returncode != 0:
        raise GCloudError(
            f"command failed (rc={proc.returncode}): {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


# ─── Quotas ────────────────────────────────────────────────────────────────────


def regional_quota(project: str, region: str, metric: str) -> int:
    """Return the regional quota limit for a metric, e.g. PREEMPTIBLE_NVIDIA_T4_GPUS."""
    proc = _run(
        [
            "gcloud",
            "compute",
            "regions",
            "describe",
            region,
            f"--project={project}",
            "--format=json",
        ]
    )
    data = json.loads(proc.stdout)
    for q in data.get("quotas", []):
        if q.get("metric") == metric:
            return int(q.get("limit", 0))
    return 0


def regional_quota_usage(project: str, region: str, metric: str) -> int:
    proc = _run(
        [
            "gcloud",
            "compute",
            "regions",
            "describe",
            region,
            f"--project={project}",
            "--format=json",
        ]
    )
    data = json.loads(proc.stdout)
    for q in data.get("quotas", []):
        if q.get("metric") == metric:
            return int(q.get("usage", 0))
    return 0


def project_quota(project: str, metric: str) -> int:
    """Return the project-wide quota limit for a metric, e.g. GPUS_ALL_REGIONS.

    Prefers the new Cloud Quotas API (`gcloud beta quotas info describe`)
    which reflects post-2024 quota raises correctly. Falls back to the
    legacy `compute project-info describe` view if the beta API isn't
    available — but the legacy view often shows stale 0s for projects with
    granted quota and should not be trusted on its own.
    """
    # Map old-style metric name to new-style quota id
    new_quota_ids = {
        "GPUS_ALL_REGIONS": "GPUS-ALL-REGIONS-per-project",
        "CPUS_ALL_REGIONS": "CPUS-ALL-REGIONS-per-project",
    }
    new_id = new_quota_ids.get(metric)
    if new_id:
        proc = subprocess.run(
            [
                "gcloud",
                "beta",
                "quotas",
                "info",
                "describe",
                new_id,
                "--service=compute.googleapis.com",
                f"--project={project}",
                "--format=json",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            try:
                data = json.loads(proc.stdout)
                # Find the global dimension's value
                for di in data.get("dimensionsInfos", []):
                    val = (di.get("details") or {}).get("value")
                    if val is not None:
                        return int(val)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        # else fall through to legacy

    # Legacy fallback
    proc = _run(
        [
            "gcloud",
            "compute",
            "project-info",
            "describe",
            f"--project={project}",
            "--format=json",
        ]
    )
    data = json.loads(proc.stdout)
    for q in data.get("quotas", []):
        if q.get("metric") == metric:
            return int(q.get("limit", 0))
    return 0


# ─── GCS bucket ────────────────────────────────────────────────────────────────


def bucket_uri(name: str) -> str:
    return f"gs://{name}" if not name.startswith("gs://") else name


def ensure_bucket(name: str, project: str, region: str) -> None:
    """Create the bucket if it doesn't already exist."""
    uri = bucket_uri(name)
    proc = subprocess.run(
        ["gsutil", "ls", "-b", uri],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return
    _run(
        [
            "gsutil",
            "mb",
            f"-p", project,
            f"-l", region,
            f"-b", "on",
            uri,
        ]
    )


def gs_upload(local: str, remote: str) -> None:
    _run(["gsutil", "-q", "cp", local, remote])


def gs_upload_string(content: str, remote: str) -> None:
    import tempfile

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(content)
        local = f.name
    try:
        gs_upload(local, remote)
    finally:
        import os as _os
        try:
            _os.unlink(local)
        except OSError:
            pass


def gs_list(prefix: str) -> list[str]:
    proc = subprocess.run(
        ["gsutil", "ls", prefix],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def gs_download(remote: str, local: str) -> None:
    _run(["gsutil", "-q", "cp", remote, local])


def gs_download_string(remote: str) -> Optional[str]:
    proc = subprocess.run(
        ["gsutil", "-q", "cat", remote], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


# ─── VM creation ───────────────────────────────────────────────────────────────


def create_spot_vm(
    *,
    project: str,
    zone: str,
    name: str,
    machine_type: str,
    accelerator_type: str,
    image_project: str,
    image_family: str,
    startup_script_path: str,
    metadata: dict[str, str],
    boot_disk_gb: int = 200,
    service_account: Optional[str] = None,
) -> dict:
    """Create a 1-GPU spot VM. Returns parsed gcloud JSON output."""
    md_pairs = ",".join(f"{k}={v}" for k, v in metadata.items())
    cmd = [
        "gcloud", "compute", "instances", "create", name,
        f"--project={project}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--image-family={image_family}",
        f"--image-project={image_project}",
        f"--boot-disk-size={boot_disk_gb}GB",
        f"--boot-disk-type=pd-balanced",
        f"--accelerator=type={accelerator_type},count=1",
        "--maintenance-policy=TERMINATE",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        f"--metadata-from-file=startup-script={startup_script_path}",
        f"--metadata={md_pairs}",
        "--scopes=cloud-platform",
        "--no-shielded-secure-boot",
        "--format=json",
    ]
    if service_account:
        cmd.append(f"--service-account={service_account}")
    proc = _run(cmd)
    return json.loads(proc.stdout)


def list_instances(project: str, zone: Optional[str] = None) -> list[dict]:
    cmd = ["gcloud", "compute", "instances", "list", f"--project={project}", "--format=json"]
    if zone:
        cmd.append(f"--zones={zone}")
    proc = _run(cmd)
    return json.loads(proc.stdout)


def delete_instance(project: str, zone: str, name: str) -> None:
    _run(
        [
            "gcloud",
            "compute",
            "instances",
            "delete",
            name,
            f"--project={project}",
            f"--zone={zone}",
            "--quiet",
        ]
    )
