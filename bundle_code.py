#!/usr/bin/env python3
# bundle_copy.py — Zero-config project bundler (run in the project folder)
# - CWD = declared_base_dir (auto)
# - detected_root = git root if exists else CWD
# - Supports .bundleignore (glob patterns)
# - Safe excludes: .venv/venv/node_modules, caches, binaries, images, >120KB

import os, sys, time, pathlib, mimetypes, subprocess, argparse, json, fnmatch

SAFE_EXCLUDE_DIRS = {
    ".git","git","venv",".venv","env",".env",
    "node_modules","__pycache__","dist","build",
    ".pytest_cache",".mypy_cache",".ruff_cache",".cache",
    ".playwright",".idea",".vscode","coverage","site-packages",
    ".DS_Store",".sass-cache",".parcel-cache"
}
SAFE_EXCLUDE_EXT = {
    ".pyc",".pyo",".pyd",".so",".dll",".dylib",
    ".zip",".tar",".tgz",".gz",".xz",".bz2",".7z",
    ".pdf",".png",".jpg",".jpeg",".webp",".tif",".tiff",".bmp",".ico",
    ".db",".sqlite",".sqlite3",".parquet",".feather",".pickle",".pkl",
    ".mp4",".mov",".mkv",".avi",".mp3",".wav",".flac"
}
SAFE_INCLUDE_EXT = {
    ".py",".js",".jsx",".ts",".tsx",
    ".json",".yml",".yaml",".toml",".ini",".conf",
    ".md",".sql",".sh",".bat",".ps1",".dockerfile",
    ".html",".css",".env.example",".env.template"
}
LANG_MAP = {
    ".py":"python",".js":"javascript",".jsx":"jsx",".ts":"ts",".tsx":"tsx",
    ".sql":"sql",".sh":"bash",".yml":"yaml",".yaml":"yaml",".toml":"toml",
    ".ini":"ini",".md":"md",".json":"json",".ps1":"powershell",
}

def lang_for(path: pathlib.Path):
    if path.name.lower()=="dockerfile": return "dockerfile"
    return LANG_MAP.get(path.suffix.lower(), "")

def looks_binary(p: pathlib.Path) -> bool:
    typ, _ = mimetypes.guess_type(str(p))
    if typ and (typ.startswith("text/") or "json" in typ or "xml" in typ):
        try:
            with p.open("rb") as f:
                chunk = f.read(4096)
            return b"\x00" in chunk
        except:
            return True
    try:
        with p.open("rb") as f:
            chunk = f.read(4096)
        return b"\x00" in chunk
    except:
        return True

def safe_run(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, cwd=cwd).decode().strip()
    except Exception:
        return ""

def detect_git_root(cwd: pathlib.Path) -> pathlib.Path:
    root = safe_run(["git","rev-parse","--show-toplevel"], cwd=str(cwd))
    return pathlib.Path(root).resolve() if root else cwd

def load_bundleignore(root: pathlib.Path):
    """Return list of glob patterns from .bundleignore if present."""
    p = root / ".bundleignore"
    patterns = []
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"): 
                continue
            patterns.append(s)
    return patterns

def matches_any(path_rel: str, patterns):
    return any(fnmatch.fnmatch(path_rel, pat) for pat in patterns)

def gather_files(root: pathlib.Path, include_ext, exclude_ext, exclude_dirs, max_bytes: int, bundleignore_patterns):
    files = []
    for r, dirs, names in os.walk(root):
        rp = pathlib.Path(r)
        # prune excluded dirs
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for n in names:
            p = rp / n
            rel = p.relative_to(root).as_posix()
            # .bundleignore patterns
            if matches_any(rel, bundleignore_patterns):
                continue
            # directory components exclude
            if any(part in exclude_dirs for part in p.parts):
                continue
            # ext filters
            ext = p.suffix.lower()
            if ext in exclude_ext: 
                continue
            if p.name.lower()!="dockerfile" and include_ext and ext not in include_ext:
                continue
            # size + binary
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except:
                continue
            if looks_binary(p):
                continue
            files.append(p)
    files.sort()
    return files

def main():
    ap = argparse.ArgumentParser(description="Bundle project into a single Markdown snapshot (zero-config).")
    ap.add_argument("--max-bytes", type=int, default=int(os.environ.get("BUNDLE_MAX_FILE_BYTES","200000")),
                    help="Per-file size limit (bytes). Default 200000.")
    ap.add_argument("--outfile", default="", help="Output .md name. Default: <folder>_bundle_YYYYMMDD_HHMM.md")
    ap.add_argument("--no-git", action="store_true", help="Skip git info lookup.")
    # 고급 사용자용(필요할 때만)
    ap.add_argument("--include-ext", nargs="*", default=None, help="Override include extensions")
    ap.add_argument("--exclude-ext", nargs="*", default=None, help="Override exclude extensions")
    ap.add_argument("--exclude-dir", nargs="*", default=None, help="Override exclude directory names")
    args = ap.parse_args()

    # CWD가 곧 'declared_base_dir'
    declared_base = pathlib.Path.cwd().resolve()
    # git root 있으면 거기로, 없으면 CWD
    detected_root = detect_git_root(declared_base)

    project_name = detected_root.name
    ts = time.strftime("%Y%m%d_%H%M")
    out = pathlib.Path(args.outfile or f"{project_name}_bundle_{ts}.md").resolve()

    include_ext = set(args.include_ext) if args.include_ext is not None else set(SAFE_INCLUDE_EXT)
    exclude_ext = set(args.exclude_ext) if args.exclude_ext is not None else set(SAFE_EXCLUDE_EXT)
    exclude_dirs = set(args.exclude_dir) if args.exclude_dir is not None else set(SAFE_EXCLUDE_DIRS)
    max_bytes = int(args.max_bytes)

    git_info = {}
    if not args.no_git and (detected_root / ".git").exists():
        git_info = {
            "branch": safe_run(["git","rev-parse","--abbrev-ref","HEAD"], cwd=str(detected_root)),
            "commit": safe_run(["git","rev-parse","--short","HEAD"], cwd=str(detected_root)),
            "root":   str(detected_root),
            "status": safe_run(["git","status","--porcelain"], cwd=str(detected_root)),
        }

    bundleignore = load_bundleignore(detected_root)

    files = gather_files(detected_root, include_ext, exclude_ext, exclude_dirs, max_bytes, bundleignore)

    meta = {
        "project_name": project_name,
        "declared_base_dir": str(declared_base),
        "detected_root": str(detected_root),
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "git": git_info or None,
        "config": {
            "include_ext": sorted(include_ext),
            "exclude_ext": sorted(exclude_ext),
            "exclude_dirs": sorted(exclude_dirs),
            "max_file_bytes": max_bytes,
            ".bundleignore": bundleignore
        }
    }

    total_bytes = 0
    with out.open("w", encoding="utf-8", errors="ignore") as f:
        f.write(f"# Project Bundle ({meta['project_name']})\n\n")
        f.write("```json\n" + json.dumps(meta, indent=2, ensure_ascii=False) + "\n```\n")
        for p in files:
            rel = p.relative_to(detected_root)
            fence = lang_for(p) or ""
            f.write(f"\n\n---\n\n## {rel}\n\n```{fence}\n")
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                total_bytes += len(txt.encode("utf-8", errors="ignore"))
                f.write(txt)
            except Exception as e:
                f.write(f"<<READ_ERROR: {e}>>")
            f.write("\n```\n")

    print(f"[OK] Wrote: {out}")
    print(f"Files: {len(files)}, approx text bytes: {total_bytes:,}")

if __name__ == "__main__":
    main()
