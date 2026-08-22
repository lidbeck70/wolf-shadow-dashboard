"""
storage.py — persistent lagring mot GitHub Contents API.

Ersätter gist_storage för panelens inmatade data. Varje flik får en egen fil
under data/ i repot, vilket ger versionshistorik på köpet: varje sparning är
en commit, så du kan se exakt när ett betyg ändrades och till vad.

VARFÖR DET HÄR BEHÖVDES
Den gamla vägen skrev till en Gist via save_blob(), som returnerar False vid
fel i stället för att kasta — och samtliga anropare kastade bort returvärdet.
En saknad eller obehörig token blev därmed omöjlig att skilja från en lyckad
sparning: data låg kvar i session_state under sessionen och var borta efter
en omladdning. Därför kastar det här modulen StorageError i stället, och
UI-lagret visar felet. Tyst dataförlust är förbjuden.

LAGRAS: endast inmatningar. Beräknade fält räknas vid rendering — samma
princip som blå mot svarta celler i Excel-arken.

KONFIGURATION (Streamlit Cloud → Settings → Secrets):

    [github]
    token  = "github_pat_..."      # fine-grained PAT, Contents: Read & write
    repo   = "lidbeck70/wolf-shadow-dashboard"
    branch = "panel-data"          # SE VARNINGEN NEDAN

VARNING OM BRANCH: Streamlit Cloud startar om appen när den deployade branchen
får en ny commit. Sparar du till den branchen startar appen om mitt i arbetet
och osparade ändringar i sessionen går förlorade — alltså exakt den bugg det
här ska laga. Peka därför branch på en gren som INTE deployas, förslagsvis
"panel-data". storage.py vägrar inte, men varnar (se check_config).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import requests
import streamlit as st

API = "https://api.github.com"
DATA_DIR = "data"
TIMEOUT = 15

# Grenen appen deployas från. Sparar man hit startar Streamlit Cloud om appen.
DEPLOY_BRANCH = "main"
DEFAULT_DATA_BRANCH = "panel-data"

SECRETS_TEMPLATE = """[github]
token  = "github_pat_..."      # fine-grained PAT, Contents: Read & write
repo   = "lidbeck70/wolf-shadow-dashboard"
branch = "panel-data\""""


class StorageError(RuntimeError):
    """Lagringen misslyckades. Meddelandet är skrivet för att visas i UI:t."""


@dataclass(frozen=True)
class Config:
    token: str
    repo: str
    branch: str

    @property
    def deploys_the_app(self) -> bool:
        return self.branch == DEPLOY_BRANCH


@dataclass(frozen=True)
class SaveResult:
    name: str
    commit_sha: str
    commit_url: str
    when: str          # ISO-tid i UTC

    @property
    def short_sha(self) -> str:
        return self.commit_sha[:7]


# ── Konfiguration ────────────────────────────────────────────────────────────
def _secrets_section() -> Optional[dict]:
    try:
        section = st.secrets["github"]
    except Exception:
        return None
    try:
        return dict(section)
    except Exception:
        return None


def get_config() -> Config:
    """Läser st.secrets["github"]. Kastar StorageError med exakt åtgärd."""
    section = _secrets_section()
    if not section:
        raise StorageError(
            "Ingen [github]-sektion i Streamlit-secrets. Lägg in detta under "
            "Settings → Secrets:\n\n" + SECRETS_TEMPLATE)

    missing = [k for k in ("token", "repo", "branch") if not str(
        section.get(k, "")).strip()]
    if missing:
        raise StorageError(
            f"Secrets saknar: {', '.join(missing)}. Hela sektionen ska se ut "
            f"så här:\n\n" + SECRETS_TEMPLATE)

    repo = str(section["repo"]).strip()
    if repo.count("/") != 1:
        raise StorageError(
            f"repo ska vara 'ägare/namn', inte {repo!r}. Exempel: "
            f"lidbeck70/wolf-shadow-dashboard")

    return Config(token=str(section["token"]).strip(), repo=repo,
                  branch=str(section["branch"]).strip())


def _headers(cfg: Config) -> dict:
    # Fine-grained PAT vill ha Bearer, klassisk token vill ha token.
    prefix = "Bearer" if cfg.token.startswith("github_pat_") else "token"
    return {"Authorization": f"{prefix} {cfg.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def path_for(name: str) -> str:
    """data/<flik>.json. Namnet saneras — det bygger en URL."""
    clean = "".join(c for c in str(name) if c.isalnum() or c in "-_")
    if not clean:
        raise StorageError(f"Ogiltigt lagringsnamn: {name!r}")
    return f"{DATA_DIR}/{clean}.json"


# ── Fel som går att åtgärda ──────────────────────────────────────────────────
def _explain(resp, cfg: Config, path: str, writing: bool) -> str:
    code = resp.status_code
    if code == 401:
        return ("GitHub svarade 401 — token är ogiltig eller har gått ut. "
                "Skapa en ny fine-grained PAT och uppdatera secrets.")
    if code == 403:
        return ("GitHub svarade 403 — token saknar skrivrätt till "
                f"{cfg.repo}. Kontrollera att PAT:en har behörigheten "
                "Contents: Read and write och att den omfattar just det "
                "repot.")
    if code == 404:
        if writing:
            return (f"GitHub svarade 404 för {cfg.repo}@{cfg.branch}. "
                    "Antingen finns inte repot/branchen, eller så ser token "
                    "det inte — en fine-grained PAT måste uttryckligen ge "
                    "åtkomst till repot.")
        return f"Filen {path} finns inte i {cfg.repo}@{cfg.branch}."
    if code == 409:
        return ("Konflikt: filen ändrades av någon annan sedan den lästes. "
                "Ladda om fliken och spara igen.")
    if code == 422:
        return (f"GitHub svarade 422 — branchen {cfg.branch!r} finns "
                "troligen inte. Skapa den, eller peka branch i secrets på en "
                "som finns.")
    detail = ""
    try:
        detail = f" {resp.json().get('message', '')}"
    except Exception:
        pass
    return f"GitHub svarade {code}.{detail}"


# ── Läsning ──────────────────────────────────────────────────────────────────
def _get_file(cfg: Config, path: str) -> Optional[dict]:
    """Filens JSON-metadata från Contents API, eller None om den saknas."""
    url = f"{API}/repos/{cfg.repo}/contents/{path}"
    try:
        r = requests.get(url, headers=_headers(cfg),
                         params={"ref": cfg.branch}, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise StorageError(f"Nådde inte GitHub: {exc}") from exc
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise StorageError(_explain(r, cfg, path, writing=False))
    return r.json()


def load_json(name: str, default: Any = None) -> Any:
    """Läser data/<name>.json. Saknas filen returneras default.

    Kastar StorageError vid allt annat än "filen finns inte" — ett läsfel som
    tyst ger tom data ser ut som att allt är borta.
    """
    cfg = get_config()
    path = path_for(name)
    meta = _get_file(cfg, path)
    if meta is None:
        return default
    try:
        raw = base64.b64decode(meta.get("content", "")).decode("utf-8")
        return json.loads(raw) if raw.strip() else default
    except (ValueError, UnicodeDecodeError) as exc:
        raise StorageError(
            f"{path} går inte att tolka som JSON: {exc}. Filen har blivit "
            f"korrupt — återställ den ur historiken i GitHub.") from exc


def load_sha(name: str) -> Optional[str]:
    """Filens nuvarande sha, eller None om den inte finns."""
    cfg = get_config()
    meta = _get_file(cfg, path_for(name))
    return meta.get("sha") if meta else None


# ── Skrivning ────────────────────────────────────────────────────────────────
def save_json(name: str, data: Any, sha: Optional[str] = None) -> SaveResult:
    """Skriver data/<name>.json och returnerar committen.

    sha hämtas automatiskt när den inte skickas med — Contents API kräver den
    för att uppdatera en existerande fil.
    """
    cfg = get_config()
    path = path_for(name)
    if sha is None:
        meta = _get_file(cfg, path)
        sha = meta.get("sha") if meta else None

    when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = json.dumps(data, indent=2, ensure_ascii=False, default=str,
                      sort_keys=True)
    payload = {
        "message": f"panel: {name} {when[:10]}",
        "content": base64.b64encode(body.encode("utf-8")).decode("ascii"),
        "branch": cfg.branch,
    }
    if sha:
        payload["sha"] = sha

    url = f"{API}/repos/{cfg.repo}/contents/{path}"
    try:
        r = requests.put(url, headers=_headers(cfg), json=payload,
                         timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise StorageError(f"Nådde inte GitHub: {exc}") from exc
    if r.status_code not in (200, 201):
        raise StorageError(_explain(r, cfg, path, writing=True))

    commit = (r.json() or {}).get("commit", {})
    return SaveResult(name=name, commit_sha=commit.get("sha", ""),
                      commit_url=commit.get("html_url", ""), when=when)


# ── Diagnos ──────────────────────────────────────────────────────────────────
def check_config() -> list:
    """[(nivå, text)] — 'ok', 'warning' eller 'error'. Kastar aldrig."""
    out = []
    try:
        cfg = get_config()
    except StorageError as exc:
        return [("error", str(exc))]

    out.append(("ok", f"Repo {cfg.repo}, branch {cfg.branch}."))
    if cfg.deploys_the_app:
        out.append((
            "warning",
            f"branch = {cfg.branch!r} är grenen appen deployas från. Varje "
            f"sparning blir en commit och Streamlit Cloud startar då om "
            f"appen — osparade ändringar i sessionen försvinner. Peka branch "
            f"på {DEFAULT_DATA_BRANCH!r} i stället."))

    try:
        r = requests.get(f"{API}/repos/{cfg.repo}", headers=_headers(cfg),
                         timeout=TIMEOUT)
        if r.status_code == 200:
            perms = (r.json() or {}).get("permissions", {})
            if perms and not perms.get("push"):
                out.append(("error", f"Token kan läsa {cfg.repo} men inte "
                                     f"skriva. Sätt Contents: Read and write."))
            else:
                out.append(("ok", "Token har skrivrätt."))
        else:
            out.append(("error", _explain(r, cfg, "", writing=True)))
    except requests.RequestException as exc:
        out.append(("error", f"Nådde inte GitHub: {exc}"))
    return out


# ── Sessionsmönstret ─────────────────────────────────────────────────────────
# Ladda EN gång per session; därefter äger session_state sanningen. Att läsa
# om från GitHub ovanpå osparade ändringar vore att kasta bort dem.

def _fingerprint(data: Any) -> str:
    try:
        return json.dumps(data, sort_keys=True, default=str)
    except Exception:
        return repr(data)


def _clean_key(name: str) -> str:
    return f"_storage_clean_{name}"


def _meta_key(name: str) -> str:
    return f"_storage_meta_{name}"


def legacy_gist(filename: str) -> Any:
    """Läser den gamla Gist-filen — engångsmigrering, aldrig skrivning.

    Panelen sparade tidigare till en Gist. Den datan får inte försvinna bara
    för att lagringen bytt plats, så första gången data/<flik>.json saknas
    hämtas den därifrån och skrivs till repot vid nästa sparning.
    """
    try:
        from gist_storage import load_blob
        return load_blob(filename, None)
    except Exception:
        return None


def session_load(name: str, default: Any = None,
                 legacy_file: Optional[str] = None) -> Any:
    """Laddar en gång per session till st.session_state[name].

    Fel sväljs inte: de läggs i _storage_meta_<name> så UI:t kan visa dem, och
    fliken får default att jobba mot i stället för att krascha.

    legacy_file pekar på den gamla Gist-filen och används bara när repofilen
    saknas — annars vinner alltid repot.
    """
    if name in st.session_state:
        return st.session_state[name]

    data, error, migrated = default, None, False
    try:
        data = load_json(name, None)
        if data is None:
            if legacy_file:
                old = legacy_gist(legacy_file)
                if old:
                    data, migrated = old, True
            if data is None:
                data = default
    except StorageError as exc:
        error = str(exc)

    st.session_state[name] = data
    # Migrerad data räknas som OSPARAD: den ligger fortfarande bara i Gisten,
    # och först ett sparklick flyttar den till repot.
    st.session_state[_clean_key(name)] = (
        "<omigrerad>" if migrated else _fingerprint(data))
    st.session_state[_meta_key(name)] = {"error": error, "saved": None,
                                         "migrated": migrated}
    return data


def is_dirty(name: str) -> bool:
    """True när sessionen avviker från senast sparade version."""
    if name not in st.session_state:
        return False
    return _fingerprint(st.session_state[name]) != st.session_state.get(
        _clean_key(name))


def mark_saved(name: str, result: Optional[SaveResult] = None) -> None:
    st.session_state[_clean_key(name)] = _fingerprint(
        st.session_state.get(name))
    meta = st.session_state.setdefault(_meta_key(name), {})
    meta["error"] = None
    if result is not None:
        meta["saved"] = {"when": result.when, "sha": result.short_sha,
                         "url": result.commit_url}


def save_session(name: str) -> SaveResult:
    """Sparar st.session_state[name] och markerar den som ren."""
    result = save_json(name, st.session_state.get(name))
    mark_saved(name, result)
    return result


def meta(name: str) -> dict:
    return st.session_state.get(_meta_key(name), {}) or {}


def last_saved(name: str) -> Optional[dict]:
    return meta(name).get("saved")


def load_error(name: str) -> Optional[str]:
    return meta(name).get("error")


def dirty_stores() -> list:
    """Alla laddade lager med osparade ändringar — för sidfoten."""
    prefix = "_storage_clean_"
    names = [k[len(prefix):] for k in st.session_state if k.startswith(prefix)]
    return sorted(n for n in names if is_dirty(n))


def saved_stores() -> list:
    """[(namn, sparinfo)] för de lager som sparats i den här sessionen."""
    out = []
    for name in sorted(st.session_state):
        if not name.startswith("_storage_meta_"):
            continue
        store = name[len("_storage_meta_"):]
        info = last_saved(store)
        if info:
            out.append((store, info))
    return out


def differs(new, old, default=None) -> bool:
    """Har widgeten faktiskt ändrat värdet — eller visar den bara sin default?

    Nummerfälten ritas med value=float(old eller 0.0): ett lagrat None VISAS
    som 0,0. Jämför man sedan widgetens 0,0 mot det lagrade None läses blotta
    öppnandet av kortet som en ändring, och osparat-varningen tänds av att man
    tittar. Här jämförs i stället mot samma default som widgeten fick.

    default är widgetens vilovärde: 0.0 för nummerfält, första alternativet
    för en selectbox, None för textfält (där "" och None är samma tomhet).
    """
    def _norm(v):
        return default if v is None or v == "" else v

    a, b = _norm(new), _norm(old)
    if a is None and b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) != bool(b)
    try:
        return float(a) != float(b)
    except (TypeError, ValueError):
        return a != b
