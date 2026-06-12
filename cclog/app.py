"""FastAPI web application for cclog."""

import io
import json
import zipfile
from pathlib import Path
from typing import Optional

import markdown
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from . import analytics as analytics_mod
from .metadata import MetadataStore
from .parser import (
    discover_all_sessions,
    extract_text_content,
    get_claude_home,
    parse_session_file,
)
from .search import build_index, search


class MetaUpdate(BaseModel):
    name: Optional[str] = None
    tags: Optional[list[str]] = None
    favorite: Optional[bool] = None
    notes: Optional[str] = None


def render_md(text: str) -> str:
    if not text:
        return ""
    return markdown.markdown(text, extensions=["fenced_code", "tables", "nl2br"])


def format_tokens_short(n: int) -> str:
    if not n:
        return "0"
    if n > 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n > 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def short_project(path: str) -> str:
    if not path:
        return "unknown"
    parts = [p for p in path.rstrip("/").split("/") if p]
    # Strip /home/<user> or /Users/<user> prefix
    if len(parts) >= 2 and parts[0] in ("home", "Users"):
        parts = parts[2:]  # drop root + username
    # Strip common noise directories
    while parts and parts[0] in ("repos", "src", "projects", "code", "dev", "work"):
        parts = parts[1:]
    # Show last 3 segments max
    if len(parts) > 3:
        parts = parts[-3:]
    return "/".join(parts) if parts else path.rstrip("/").split("/")[-1]


def short_model(model: str) -> str:
    if not model:
        return ""
    return model.replace("claude-", "").replace("-4-6", " 4.6").replace("-4-5", " 4.5")


def format_date(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d, %H:%M")
    except (ValueError, TypeError):
        return iso_str[:16] if iso_str else ""


def format_date_long(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d, %Y %H:%M")
    except (ValueError, TypeError):
        return iso_str[:16] if iso_str else ""


def get_display_name(summary: dict, meta: dict) -> str:
    name = meta.get("custom_name")
    if name:
        return name
    name = summary.get("auto_summary")
    if name:
        return name
    preview = summary.get("first_message_preview", "")
    if preview:
        return preview[:60]
    return "Untitled conversation"


def create_app(claude_home: Path | None = None) -> FastAPI:
    claude_home = claude_home or get_claude_home()

    from . import __version__

    app = FastAPI(title="cclog", version=__version__)

    base_dir = Path(__file__).parent
    app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
    templates = Jinja2Templates(directory=str(base_dir / "templates"))

    # Register template filters/globals
    templates.env.globals["version"] = __version__
    templates.env.filters["format_tokens"] = format_tokens_short
    templates.env.filters["short_project"] = short_project
    templates.env.filters["short_model"] = short_model
    templates.env.filters["format_date"] = format_date
    templates.env.filters["format_date_long"] = format_date_long
    templates.env.filters["render_md"] = render_md
    templates.env.globals["get_display_name"] = get_display_name

    # Load data
    print(f"Reading conversations from {claude_home}")
    summaries = discover_all_sessions(claude_home)
    print(f"Found {len(summaries)} conversations")

    metadata = MetadataStore()

    print("Building search index...")
    try:
        build_index(summaries, claude_home)
        print("Search index built successfully")
    except Exception as e:
        print(f"Warning: Failed to build search index: {e}")

    def _rescan_sessions():
        """Rescan claude_home and rebuild search index. Returns (total, new) counts."""
        old_count = len(summaries)
        new_summaries = discover_all_sessions(claude_home)
        summaries.clear()
        summaries.extend(new_summaries)
        try:
            build_index(summaries, claude_home)
        except Exception as e:
            print(f"Warning: Failed to rebuild search index: {e}")
        return len(summaries), len(summaries) - old_count

    def _count_session_files():
        """Cheap check: count .jsonl files without parsing them."""
        projects_dir = claude_home / "projects"
        if not projects_dir.exists():
            return 0
        return sum(1 for d in projects_dir.iterdir() if d.is_dir() and not d.name.startswith(".") for f in d.glob("*.jsonl"))

    # ── Pages ────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def landing(request: Request):
        active = [s for s in summaries if not metadata.get(s["session_id"]).get("deleted")]
        overview = analytics_mod.compute_overview(active)
        total_tok = overview["total_input_tokens"] + overview["total_output_tokens"]
        return templates.TemplateResponse(request, "landing.html", {
            "total_sessions": overview["total_sessions"],
            "total_tokens": format_tokens_short(total_tok),
            "total_projects": overview["total_projects"],
        })

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        active = [s for s in summaries if not metadata.get(s["session_id"]).get("deleted")]
        range_param = request.query_params.get("range", "all")
        selected_range = range_param if range_param in analytics_mod.TIME_RANGES else "all"
        range_options = [
            {"key": key, "label": option["label"]}
            for key, option in analytics_mod.TIME_RANGES.items()
        ]
        ranged_active = analytics_mod.filter_summaries_by_range(active, selected_range)
        overview = analytics_mod.compute_overview(ranged_active)
        recent = ranged_active[:8]
        recent_with_meta = [(s, metadata.get(s["session_id"])) for s in recent]
        return templates.TemplateResponse(request, "dashboard.html", {
            "overview": overview,
            "recent": recent_with_meta,
            "daily_json": json.dumps(overview["daily_activity"]),
            "heatmap_json": json.dumps(overview["heatmap"]),
            "range_options": range_options,
            "selected_range": selected_range,
            "selected_range_label": analytics_mod.TIME_RANGES[selected_range]["label"],
            "active_nav": "dashboard",
        })

    @app.get("/projects", response_class=HTMLResponse)
    async def projects_page(request: Request):
        projects = {}
        for s in summaries:
            if metadata.get(s["session_id"]).get("deleted"):
                continue
            enc = s["project_encoded"]
            if enc not in projects:
                projects[enc] = {"path": s["project_path"], "encoded": enc, "session_count": 0, "total_tokens": 0, "last_activity": None}
            projects[enc]["session_count"] += 1
            projects[enc]["total_tokens"] += s.get("total_input_tokens", 0) + s.get("total_output_tokens", 0)
            la = s.get("last_activity")
            if la and (projects[enc]["last_activity"] is None or la > projects[enc]["last_activity"]):
                projects[enc]["last_activity"] = la
        sorted_projects = sorted(projects.values(), key=lambda p: p["total_tokens"], reverse=True)
        return templates.TemplateResponse(request, "projects.html", {
            "projects": sorted_projects, "active_nav": "projects",
        })

    @app.get("/projects/{encoded:path}", response_class=HTMLResponse)
    async def project_sessions_page(request: Request, encoded: str):
        sessions = [
            (s, metadata.get(s["session_id"]))
            for s in summaries
            if s["project_encoded"] == encoded and not metadata.get(s["session_id"]).get("deleted")
        ]
        project_name = sessions[0][0]["project_path"] if sessions else encoded
        return templates.TemplateResponse(request, "project_sessions.html", {
            "sessions": sessions, "project_name": project_name,
            "encoded": encoded, "active_nav": "projects",
        })

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    async def conversation_page(request: Request, session_id: str):
        return templates.TemplateResponse(request, "conversation.html", {
            "session_id": session_id, "active_nav": "projects",
        })

    @app.get("/search", response_class=HTMLResponse)
    async def search_page(request: Request, q: str = ""):
        return templates.TemplateResponse(request, "search.html", {
            "query": q, "active_nav": "search",
        })

    @app.get("/tags", response_class=HTMLResponse)
    async def tags_page(request: Request):
        tags = metadata.all_tags()
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        return templates.TemplateResponse(request, "tags.html", {
            "tags": sorted_tags, "active_nav": "tags",
        })

    @app.get("/import", response_class=HTMLResponse)
    async def import_page(request: Request):
        return templates.TemplateResponse(request, "import.html", {
            "active_nav": "import",
        })

    # ── API ──────────────────────────────────────

    @app.get("/api/v1/projects")
    async def api_projects():
        projects = {}
        for s in summaries:
            if metadata.get(s["session_id"]).get("deleted"):
                continue
            enc = s["project_encoded"]
            if enc not in projects:
                projects[enc] = {"path": s["project_path"], "encoded": enc, "session_count": 0, "total_tokens": 0, "last_activity": None}
            projects[enc]["session_count"] += 1
            projects[enc]["total_tokens"] += s.get("total_input_tokens", 0) + s.get("total_output_tokens", 0)
            la = s.get("last_activity")
            if la and (projects[enc]["last_activity"] is None or la > projects[enc]["last_activity"]):
                projects[enc]["last_activity"] = la
        return sorted(projects.values(), key=lambda p: p["total_tokens"], reverse=True)

    @app.get("/api/v1/sessions/{session_id}")
    async def api_get_session(session_id: str):
        summary = next((s for s in summaries if s["session_id"] == session_id), None)
        if not summary:
            raise HTTPException(status_code=404, detail="Not found")

        session_file = claude_home / "projects" / summary["project_encoded"] / f"{session_id}.jsonl"
        if not session_file.exists():
            raise HTTPException(status_code=404, detail="Session file not found")

        messages = parse_session_file(session_file)
        meta = metadata.get(session_id)

        msgs_out = []
        for entry in messages:
            msg = entry.get("message", {})
            content = msg.get("content", "")
            content_html = ""
            tool_uses = []
            thinking = None

            if isinstance(content, str):
                content_html = render_md(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        content_html += render_md(block.get("text", ""))
                    elif btype == "thinking":
                        thinking = block.get("thinking", "")[:500]
                    elif btype == "tool_use":
                        inp = json.dumps(block.get("input", {}))
                        tool_uses.append({"name": block.get("name", ""), "input_preview": inp[:200]})

            usage = msg.get("usage", {})
            msgs_out.append({
                "uuid": entry.get("uuid"),
                "role": msg.get("role", ""),
                "content_html": content_html,
                "model": msg.get("model"),
                "input_tokens": (usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)) if usage else None,
                "output_tokens": usage.get("output_tokens") if usage else None,
                "timestamp": entry.get("timestamp"),
                "is_sidechain": entry.get("isSidechain", False),
                "tool_uses": tool_uses,
                "thinking": thinking,
            })

        return {
            "session_id": session_id,
            "display_name": get_display_name(summary, meta),
            "project_path": summary["project_path"],
            "git_branch": summary.get("git_branch"),
            "started_at": summary.get("started_at"),
            "tags": meta.get("tags", []),
            "favorite": meta.get("favorite", False),
            "messages": msgs_out,
        }

    @app.patch("/api/v1/sessions/{session_id}/meta")
    async def api_update_meta(session_id: str, update: MetaUpdate):
        updates = {}
        if update.name is not None:
            updates["custom_name"] = update.name if update.name else None
        if update.tags is not None:
            updates["tags"] = update.tags
        if update.favorite is not None:
            updates["favorite"] = update.favorite
        if update.notes is not None:
            updates["notes"] = update.notes if update.notes else None
        metadata.update(session_id, **updates)
        return {"ok": True}

    @app.post("/api/v1/sessions/{session_id}/delete")
    async def api_delete_session(session_id: str):
        metadata.update(session_id, deleted=True)
        return {"ok": True}

    @app.post("/api/v1/sessions/{session_id}/undelete")
    async def api_undelete_session(session_id: str):
        metadata.update(session_id, deleted=False)
        return {"ok": True}

    @app.get("/api/v1/search")
    async def api_search(q: str = "", limit: int = 50):
        if not q or len(q) < 2:
            return {"results": []}
        results = search(q, limit=limit)
        return {"results": results}

    @app.get("/api/v1/tags")
    async def api_tags():
        tags = metadata.all_tags()
        tag_list = [{"tag": t, "count": c} for t, c in tags.items()]
        tag_list.sort(key=lambda x: x["count"], reverse=True)
        return {"tags": tag_list}

    @app.post("/api/v1/refresh")
    async def api_refresh():
        """Rescan ~/.claude/ for new sessions and rebuild search index."""
        # Cheap check: skip full rescan if file count hasn't changed
        if _count_session_files() == len(summaries):
            return {"ok": True, "sessions": len(summaries), "new": 0}
        total, new = _rescan_sessions()
        return {"ok": True, "sessions": total, "new": new}

    @app.get("/api/v1/analytics/overview")
    async def api_analytics():
        active = [s for s in summaries if not metadata.get(s["session_id"]).get("deleted")]
        return analytics_mod.compute_overview(active)

    @app.get("/api/v1/version")
    async def api_version():
        latest = __version__
        try:
            from urllib.request import urlopen
            resp = urlopen("https://pypi.org/pypi/claude-log/json", timeout=3)
            data = json.loads(resp.read())
            latest = data["info"]["version"]
        except Exception:
            pass
        return {"current": __version__, "latest": latest, "update_available": latest != __version__}

    # ── Export / Import ──────────────────────────

    @app.get("/api/v1/export/session/{session_id}")
    async def api_export_session(session_id: str):
        """Export a single conversation as a zip."""
        summary = next((s for s in summaries if s["session_id"] == session_id), None)
        if not summary:
            raise HTTPException(status_code=404, detail="Not found")

        session_file = claude_home / "projects" / summary["project_encoded"] / f"{session_id}.jsonl"
        if not session_file.exists():
            raise HTTPException(status_code=404, detail="Session file not found")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(session_file, f"projects/{summary['project_encoded']}/{session_id}.jsonl")
            meta = metadata.get(session_id)
            if any(v for k, v in meta.items() if k != "deleted"):
                zf.writestr("cclog_metadata.json", json.dumps({"sessions": {session_id: meta}}, indent=2))
            zf.writestr("manifest.json", json.dumps({"version": "1", "total_sessions": 1, "exported_from": str(claude_home)}, indent=2))

        buf.seek(0)
        name = short_project(summary["project_path"]).replace("/", "-")
        return StreamingResponse(buf, media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=cclog-{name}-{session_id[:8]}.zip"})

    @app.get("/api/v1/export/project/{encoded:path}")
    async def api_export_project(encoded: str):
        """Export all conversations from a project as a zip."""
        project_sessions = [s for s in summaries if s["project_encoded"] == encoded]
        if not project_sessions:
            raise HTTPException(status_code=404, detail="Project not found")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            project_dir = claude_home / "projects" / encoded
            idx = project_dir / "sessions-index.json"
            if idx.exists():
                zf.write(idx, f"projects/{encoded}/sessions-index.json")
            for s in project_sessions:
                f = project_dir / f"{s['session_id']}.jsonl"
                if f.exists():
                    zf.write(f, f"projects/{encoded}/{s['session_id']}.jsonl")
            # Include metadata for these sessions
            meta_export = {}
            for s in project_sessions:
                m = metadata.get(s["session_id"])
                if any(v for k, v in m.items() if k != "deleted"):
                    meta_export[s["session_id"]] = m
            if meta_export:
                zf.writestr("cclog_metadata.json", json.dumps({"sessions": meta_export}, indent=2))
            zf.writestr("manifest.json", json.dumps({"version": "1", "total_sessions": len(project_sessions), "exported_from": str(claude_home)}, indent=2))

        buf.seek(0)
        name = short_project(project_sessions[0]["project_path"]).replace("/", "-")
        return StreamingResponse(buf, media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=cclog-{name}.zip"})

    @app.get("/api/v1/export")
    async def api_export():
        buf = io.BytesIO()
        projects_dir = claude_home / "projects"

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if projects_dir.exists():
                for project_dir in sorted(projects_dir.iterdir()):
                    if not project_dir.is_dir() or project_dir.name.startswith("."):
                        continue
                    idx = project_dir / "sessions-index.json"
                    if idx.exists():
                        zf.write(idx, f"projects/{project_dir.name}/sessions-index.json")
                    for f in sorted(project_dir.glob("*.jsonl")):
                        zf.write(f, f"projects/{project_dir.name}/{f.name}")

            history = claude_home / "history.jsonl"
            if history.exists():
                zf.write(history, "history.jsonl")

            if metadata.path.exists():
                zf.write(metadata.path, "cclog_metadata.json")

            zf.writestr("manifest.json", json.dumps({
                "version": "1",
                "total_sessions": len(summaries),
                "exported_from": str(claude_home),
            }, indent=2))

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=cclog-export.zip"},
        )

    @app.post("/api/v1/import")
    async def api_import(file: UploadFile = File(...)):
        if not file.filename or not file.filename.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Must be a .zip file")

        content = await file.read()
        try:
            with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
                names = zf.namelist()
                if "manifest.json" not in names:
                    raise HTTPException(status_code=400, detail="Invalid export: missing manifest.json")

                imported_sessions = 0
                imported_projects = set()

                for name in names:
                    if name.startswith("projects/") and name.endswith(".jsonl"):
                        parts = name.split("/")
                        if len(parts) == 3:
                            project_name = parts[1]
                            dest_dir = claude_home / "projects" / project_name
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            dest = dest_dir / parts[2]
                            if not dest.exists():
                                dest.write_bytes(zf.read(name))
                                imported_sessions += 1
                                imported_projects.add(project_name)

                    elif name.startswith("projects/") and name.endswith("sessions-index.json"):
                        parts = name.split("/")
                        if len(parts) == 3:
                            dest_dir = claude_home / "projects" / parts[1]
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            dest = dest_dir / "sessions-index.json"
                            if not dest.exists():
                                dest.write_bytes(zf.read(name))

                    elif name == "history.jsonl":
                        history_dest = claude_home / "history.jsonl"
                        new_lines = zf.read(name).decode("utf-8", errors="replace")
                        if history_dest.exists():
                            existing_set = set(history_dest.read_text().strip().split("\n"))
                            new_entries = [l for l in new_lines.strip().split("\n") if l and l not in existing_set]
                            if new_entries:
                                with open(history_dest, "a") as hf:
                                    hf.write("\n" + "\n".join(new_entries))
                        else:
                            history_dest.write_text(new_lines)

                    elif name == "cclog_metadata.json":
                        imported_meta = json.loads(zf.read(name))
                        for sid, meta_val in imported_meta.get("sessions", {}).items():
                            if sid not in metadata.sessions:
                                metadata.sessions[sid] = meta_val
                        metadata.save()

                _rescan_sessions()

                return {"success": True, "imported_sessions": imported_sessions, "imported_projects": len(imported_projects)}

        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file")

    return app
