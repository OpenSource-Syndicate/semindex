import os
import re
import time
import hashlib
from typing import Iterable, List, Optional, Tuple
import importlib.metadata as im

import requests
from bs4 import BeautifulSoup

from .embed import Embedder
from .store import (
    DB_NAME,
    DOCS_FAISS_INDEX,
    add_doc_vectors,
    db_conn,
    ensure_db,
)


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()


def _read_requirements(req_path: str) -> List[str]:
    if not os.path.exists(req_path):
        return []
    names: List[str] = []
    with open(req_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # simple parse: pkg[extra]==ver | pkg>=ver | pkg
            name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
            if name:
                names.append(name.replace('_', '-'))
    return names


def _candidate_doc_urls_from_pypi(name: str, version: Optional[str] = None) -> List[str]:
    urls: List[str] = []
    try:
        url = f"https://pypi.org/pypi/{name}/{version}/json" if version else f"https://pypi.org/pypi/{name}/json"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return urls
        data = resp.json()
        info = data.get('info', {})
        project_urls = info.get('project_urls') or {}
        for key in [
            'Documentation', 'Read the Docs', 'Docs', 'Homepage', 'Home', 'Source', 'Repository'
        ]:
            u = project_urls.get(key)
            if u:
                urls.append(u)
        if info.get('docs_url'):
            urls.append(info['docs_url'])
        if info.get('home_page'):
            urls.append(info['home_page'])
    except Exception:
        pass
    # de-dup while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq


def _fetch_url(url: str) -> Optional[Tuple[bytes, str]]:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and r.content:
            ctype = r.headers.get('Content-Type', '')
            return r.content, ctype
    except Exception:
        return None
    return None


def _html_to_text(content: bytes) -> Tuple[str, str]:
    html = content.decode('utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')
    # remove nav/footer/aside
    for tag in soup(['nav', 'footer', 'aside', 'script', 'style']):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else '')
    text = soup.get_text("\n", strip=True)
    return title, text


def _discover_local_doc_dirs() -> List[Tuple[str, str, str]]:
    """
    Returns list of (package, version, path_to_docs_dir)
    """
    found: List[Tuple[str, str, str]] = []
    for dist in im.distributions():
        try:
            name = dist.metadata['Name'] or dist.metadata.get('name') or dist.metadata.get('Summary')
            if not name:
                continue
            name = name.replace('_', '-').lower()
            version = dist.version
            loc = getattr(dist, 'locate_file', None)
            if loc:
                base = str(dist.locate_file(''))
            else:
                base = os.path.dirname(dist._path) if hasattr(dist, '_path') else ''
            if not base:
                continue
            for candidate in ['docs', 'doc', 'documentation']:
                p = os.path.join(base, candidate)
                if os.path.isdir(p):
                    found.append((name, version, p))
        except Exception:
            continue
    return found


def _iter_local_doc_pages(doc_dir: str) -> Iterable[Tuple[str, bytes, str]]:
    """
    Yield (relative_path, content_bytes, content_type)
    """
    for root, _dirs, files in os.walk(doc_dir):
        for fn in files:
            if fn.lower().endswith(('.md', '.markdown')):
                p = os.path.join(root, fn)
                try:
                    with open(p, 'rb') as f:
                        yield os.path.relpath(p, doc_dir).replace('\\', '/'), f.read(), 'text/markdown'
                except Exception:
                    continue
            elif fn.lower().endswith(('.html', '.htm')):
                p = os.path.join(root, fn)
                try:
                    with open(p, 'rb') as f:
                        yield os.path.relpath(p, doc_dir).replace('\\', '/'), f.read(), 'text/html'
                except Exception:
                    continue


def index_docs(index_dir: str, repo_root: str, embedder: Optional[Embedder] = None, verbose: bool = False, page_limit: int = 50):
    """
    Index external library documentation based on requirements.txt in repo_root and local site-packages docs.
    """
    os.makedirs(index_dir, exist_ok=True)
    db_path = os.path.join(index_dir, DB_NAME)
    docs_index_path = os.path.join(index_dir, DOCS_FAISS_INDEX)
    ensure_db(db_path)

    if embedder is None:
        embedder = Embedder()

    # Collect deps from requirements.txt
    reqs = _read_requirements(os.path.join(repo_root, 'requirements.txt'))
    req_set = {r.lower() for r in reqs}

    # Add local docs regardless of requirements to capture environment packages
    local_docs = _discover_local_doc_dirs()

    # Prepare containers
    page_rows: List[Tuple[str, str, str, str, str, float]] = []  # (package, version, url, title, checksum, ts)
    texts: List[str] = []

    # Remote docs via PyPI
    for pkg in list(req_set)[:50]:
        urls = _candidate_doc_urls_from_pypi(pkg)
        if verbose:
            print(f"[docs] {pkg}: candidates={len(urls)}")
        for u in urls[:3]:
            fetched = _fetch_url(u)
            if not fetched:
                continue
            content, ctype = fetched
            title, text = ("", "")
            if 'html' in ctype.lower() or u.endswith(('.html', '/')):
                title, text = _html_to_text(content)
            else:
                try:
                    text = content.decode('utf-8', errors='ignore')
                except Exception:
                    continue
            if not text or len(text) < 200:
                continue
            checksum = _sha256_bytes(content)
            page_rows.append((pkg, '', u, title or u, checksum, time.time()))
            texts.append(text)
            if len(texts) >= page_limit:
                break
        if len(texts) >= page_limit:
            break

    # Local docs
    for pkg, version, doc_dir in local_docs:
        count = 0
        for rel, content, ctype in _iter_local_doc_pages(doc_dir):
            if 'html' in ctype:
                title, text = _html_to_text(content)
            else:
                try:
                    text = content.decode('utf-8', errors='ignore')
                    title = rel
                except Exception:
                    continue
            if not text or len(text) < 200:
                continue
            checksum = _sha256_bytes(content)
            url = f"file://{os.path.join(doc_dir, rel).replace('\\', '/')}"
            page_rows.append((pkg, version, url, title, checksum, time.time()))
            texts.append(text)
            count += 1
            if count >= 20 or len(texts) >= page_limit:
                break
        if len(texts) >= page_limit:
            break

    if not texts:
        if verbose:
            print("[docs] No documentation pages discovered.")
        return

    # Write to DB and FAISS
    with db_conn(db_path) as con:
        cur = con.cursor()
        # Upsert packages
        for pkg, ver, *_rest in page_rows:
            cur.execute(
                "INSERT OR IGNORE INTO doc_packages(name, version) VALUES (?, ?);",
                (pkg, ver or ''),
            )
        # Avoid reinserting identical pages by checksum
        existing = {row[0] for row in cur.execute("SELECT checksum FROM doc_pages;").fetchall()}
        new_rows = [r for r in page_rows if r[4] not in existing]
        if not new_rows:
            if verbose:
                print("[docs] No new/changed doc pages.")
            return
        cur.executemany(
            """
            INSERT INTO doc_pages(package, version, url, title, checksum, last_indexed)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            new_rows,
        )
        last_id = cur.execute("SELECT ifnull(MAX(id), 0) FROM doc_pages;").fetchone()[0]
        count = len(new_rows)
        first_id = last_id - count + 1 if count > 0 else 0
        page_ids = list(range(first_id, last_id + 1)) if count > 0 else []

        vecs = embedder.encode(texts, batch_size=8)
        add_doc_vectors(docs_index_path, con, page_ids, vecs)

    if verbose:
        print(f"[docs] Indexed {len(texts)} documentation pages")
