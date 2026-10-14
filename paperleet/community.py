"""Persistent community actions, with a single public profile privacy boundary."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
import uuid

from .accounts import public_user
from .catalog import problem, problems


CATEGORIES = ("papers", "implementation", "questions", "career", "lounge")
ACTIVITY_TABS = ("posts", "replies", "likes", "reposts")


def now():
    return datetime.now(timezone.utc).isoformat()


def _text(payload, field, limit, required=True):
    value = payload.get(field, "")
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{field} must be text of at most {limit} characters")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{field} is required")
    if "\x00" in value:
        raise ValueError(f"{field} contains an invalid character")
    return value


def _payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected an object")
    return payload


def _page(limit, offset):
    if isinstance(limit, bool) or isinstance(offset, bool):
        raise ValueError("Invalid pagination")
    try:
        limit, offset = int(limit), int(offset)
    except (TypeError, ValueError):
        raise ValueError("Invalid pagination") from None
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("limit must be between 1 and 100; offset must be non-negative")
    return limit, offset


class CommunityService:
    """Accounts must initialize the shared database before this service is used."""

    def __init__(self, db_path, institutions=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.institutions = institutions
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS community_posts (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    category TEXT NOT NULL CHECK(category IN ('papers','implementation','questions','career','lounge')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS community_replies (
                    id TEXT PRIMARY KEY,
                    post_id TEXT NOT NULL REFERENCES community_posts(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    parent_id TEXT REFERENCES community_replies(id) ON DELETE SET NULL,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS community_reactions (
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    post_id TEXT NOT NULL REFERENCES community_posts(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL CHECK(kind IN ('like','repost')),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, post_id, kind)
                );
                CREATE TABLE IF NOT EXISTS community_follows (
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    target_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, target_id),
                    CHECK(user_id != target_id)
                );
                CREATE INDEX IF NOT EXISTS community_posts_recent ON community_posts(created_at DESC, id);
                CREATE INDEX IF NOT EXISTS community_posts_author ON community_posts(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS community_replies_post ON community_replies(post_id, created_at);
                CREATE INDEX IF NOT EXISTS community_replies_author ON community_replies(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS community_reactions_post ON community_reactions(post_id, kind);
                CREATE INDEX IF NOT EXISTS community_reactions_author ON community_reactions(user_id, kind, created_at DESC);
                CREATE INDEX IF NOT EXISTS community_follows_target ON community_follows(target_id, created_at DESC);
            """)
            # Serialize schema upgrades across processes; keep existing posts
            # unassociated rather than inferring a topic from their text.
            db.execute("BEGIN IMMEDIATE")
            if "problem_id" not in {row[1] for row in db.execute("PRAGMA table_info(community_posts)")}:
                db.execute("ALTER TABLE community_posts ADD COLUMN problem_id TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS community_posts_problem ON community_posts(problem_id, created_at DESC, id)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _user(self, db, user_id):
        row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            raise KeyError("Unknown user")
        return row

    def _require_user(self, db, user_id):
        # Hold a write transaction through authorization and mutation. Concurrent
        # changes cannot invalidate ownership, parent-reply or account checks.
        if not db.in_transaction:
            db.execute("BEGIN IMMEDIATE")
        if not user_id:
            raise PermissionError("Institutional sign-in is required")
        row = self._user(db, user_id)
        if row["status"] != "active":
            raise PermissionError("This account is not active")
        return row

    def _by_username(self, db, username):
        if not isinstance(username, str):
            raise ValueError("Invalid username")
        row = db.execute("SELECT * FROM users WHERE username = ?", (username.lower(),)).fetchone()
        if row is None:
            raise KeyError("Unknown user")
        return row

    def _public_user(self, db, row):
        return public_user(db, row, self.institutions)

    def _post_row(self, db, post_id):
        row = db.execute("SELECT * FROM community_posts WHERE id = ?", (post_id,)).fetchone()
        if row is None:
            raise KeyError("Unknown post")
        return row

    def _can_delete(self, db, user_id, owner_id):
        if not user_id:
            return False
        user = db.execute("SELECT status, role FROM users WHERE id = ?", (user_id,)).fetchone()
        return bool(user and user["status"] == "active" and (user_id == owner_id or user["role"] == "admin"))

    def _post(self, db, row, viewer_id=None, problem_index=None):
        result = {field: row[field] for field in ("id", "title", "body", "category", "created_at", "updated_at", "problem_id")}
        result["problem"] = None
        if row["problem_id"]:
            if problem_index is None:
                problem_index = {entry["id"]: entry for entry in problems()}
            entry = problem_index.get(row["problem_id"])
            # Retired modules must not make historical posts unreadable or
            # redirect to a different module which reused their number.
            result["problem"] = ({key: entry.get(key) for key in ("id", "number", "title", "title_en")}
                                 if entry else {"id": row["problem_id"], "title": row["problem_id"], "retired": True})
        result["author"] = self._public_user(db, self._user(db, row["user_id"]))
        reactions = dict(db.execute("SELECT kind, COUNT(*) FROM community_reactions WHERE post_id = ? GROUP BY kind", (row["id"],)))
        result["counts"] = {
            "likes": reactions.get("like", 0),
            "reposts": reactions.get("repost", 0),
            "replies": db.execute("SELECT COUNT(*) FROM community_replies WHERE post_id = ?", (row["id"],)).fetchone()[0],
        }
        mine = {r[0] for r in db.execute("SELECT kind FROM community_reactions WHERE post_id = ? AND user_id = ?", (row["id"], viewer_id))}
        result["viewer"] = {"liked": "like" in mine, "reposted": "repost" in mine,
                            "can_delete": self._can_delete(db, viewer_id, row["user_id"])}
        return result

    def _reply(self, db, row, viewer_id=None):
        result = {field: row[field] for field in ("id", "post_id", "parent_id", "body", "created_at")}
        result["author"] = self._public_user(db, self._user(db, row["user_id"]))
        result["can_delete"] = self._can_delete(db, viewer_id, row["user_id"])
        return result

    def list_posts(self, viewer_id=None, category=None, query="", following=False, limit=30, offset=0, problem_id=None):
        limit, offset = _page(limit, offset)
        if category and category not in CATEGORIES:
            raise ValueError("Unknown category")
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("Search must be at most 200 characters")
        where, params = [], []
        if problem_id is not None:
            if not isinstance(problem_id, str) or len(problem_id) > 200:
                raise ValueError("Invalid problem_id")
            # Canonical IDs also keep historical threads accessible after a
            # module is retired. Creation still requires a current module.
            if not re.fullmatch(r"[a-z0-9_\-]+/[a-z0-9_\-]+", problem_id):
                raise ValueError("problem_id must be a canonical module ID")
            where.append("p.problem_id = ?")
            params.append(problem_id)
        if category:
            where.append("p.category = ?")
            params.append(category)
        if query.strip():
            # Literal substring search: user '%' and '_' are not SQL wildcards.
            where.append("(instr(lower(p.title), lower(?)) > 0 OR instr(lower(p.body), lower(?)) > 0)")
            params.extend((query.strip(), query.strip()))
        if following:
            if not viewer_id:
                raise PermissionError("Sign in to see posts from people you follow")
            where.append("EXISTS (SELECT 1 FROM community_follows f WHERE f.user_id = ? AND f.target_id = p.user_id)")
            params.append(viewer_id)
        clause = " WHERE " + " AND ".join(where) if where else ""
        with self.connection() as db:
            total = db.execute("SELECT COUNT(*) FROM community_posts p" + clause, params).fetchone()[0]
            rows = db.execute("SELECT p.* FROM community_posts p" + clause + " ORDER BY p.created_at DESC, p.id DESC LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
            problem_index = {entry["id"]: entry for entry in problems()} if any(row["problem_id"] for row in rows) else {}
            return {"items": [self._post(db, row, viewer_id, problem_index) for row in rows], "total": total, "limit": limit, "offset": offset}

    def get_post(self, post_id, viewer_id=None):
        with self.connection() as db:
            result = self._post(db, self._post_row(db, post_id), viewer_id)
            result["replies"] = [self._reply(db, row, viewer_id) for row in db.execute(
                "SELECT * FROM community_replies WHERE post_id = ? ORDER BY created_at, id", (post_id,)).fetchall()]
            return result

    def create_post(self, user_id, payload):
        _payload(payload)
        title = _text(payload, "title", 180)
        body = _text(payload, "body", 20000)
        category = payload.get("category", "questions")
        if category not in CATEGORIES:
            raise ValueError("Unknown category")
        problem_id = payload.get("problem_id")
        if problem_id is not None:
            if not isinstance(problem_id, str) or len(problem_id) > 200:
                raise ValueError("Invalid problem_id")
            if problem(problem_id)["id"] != problem_id:
                raise ValueError("problem_id must be a canonical module ID")
        post_id, timestamp = uuid.uuid4().hex, now()
        with self.connection() as db:
            self._require_user(db, user_id)
            db.execute("INSERT INTO community_posts(id,user_id,title,body,category,created_at,updated_at,problem_id) VALUES(?,?,?,?,?,?,?,?)",
                       (post_id, user_id, title, body, category, timestamp, timestamp, problem_id))
            return self._post(db, self._post_row(db, post_id), user_id)

    def reply(self, user_id, post_id, payload):
        _payload(payload)
        body = _text(payload, "body", 10000)
        parent_id = payload.get("parent_id") or None
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError("Invalid parent reply")
        reply_id, timestamp = uuid.uuid4().hex, now()
        with self.connection() as db:
            self._require_user(db, user_id)
            self._post_row(db, post_id)
            if parent_id:
                parent = db.execute("SELECT post_id FROM community_replies WHERE id = ?", (parent_id,)).fetchone()
                if parent is None or parent["post_id"] != post_id:
                    raise ValueError("Parent reply must belong to this post")
            db.execute("INSERT INTO community_replies(id,post_id,user_id,parent_id,body,created_at) VALUES(?,?,?,?,?,?)",
                       (reply_id, post_id, user_id, parent_id, body, timestamp))
            row = db.execute("SELECT * FROM community_replies WHERE id = ?", (reply_id,)).fetchone()
            return self._reply(db, row, user_id)

    def set_reaction(self, user_id, post_id, kind, active):
        if kind not in ("like", "repost") or not isinstance(active, bool):
            raise ValueError("Reaction requires like/repost and a boolean active flag")
        with self.connection() as db:
            self._require_user(db, user_id)
            row = self._post_row(db, post_id)
            if active:
                db.execute("INSERT OR IGNORE INTO community_reactions(user_id,post_id,kind,created_at) VALUES(?,?,?,?)", (user_id, post_id, kind, now()))
            else:
                db.execute("DELETE FROM community_reactions WHERE user_id = ? AND post_id = ? AND kind = ?", (user_id, post_id, kind))
            return self._post(db, row, user_id)

    def _profile(self, db, row, viewer_id=None):
        result = self._public_user(db, row)
        uid = row["id"]
        counts = {}
        for label, table, field in (("posts", "community_posts", "user_id"), ("replies", "community_replies", "user_id"),
                                     ("following", "community_follows", "user_id"), ("followers", "community_follows", "target_id")):
            counts[label] = db.execute(f"SELECT COUNT(*) FROM {table} WHERE {field} = ?", (uid,)).fetchone()[0]
        for label, kind in (("likes", "like"), ("reposts", "repost")):
            counts[label] = db.execute("SELECT COUNT(*) FROM community_reactions WHERE user_id = ? AND kind = ?", (uid, kind)).fetchone()[0]
            counts[label + "_received"] = db.execute("""
                SELECT COUNT(*) FROM community_reactions r
                JOIN community_posts p ON p.id=r.post_id
                WHERE p.user_id=? AND r.user_id!=p.user_id AND r.kind=?
            """, (uid, kind)).fetchone()[0]
        result["counts"] = counts
        result["viewer"] = self._relationship(db, uid, viewer_id)
        return result

    def _relationship(self, db, user_id, viewer_id):
        return {"following": bool(db.execute("SELECT 1 FROM community_follows WHERE user_id = ? AND target_id = ?", (viewer_id, user_id)).fetchone()),
                "is_self": viewer_id == user_id}

    def profile(self, username, viewer_id=None):
        with self.connection() as db:
            return self._profile(db, self._by_username(db, username), viewer_id)

    def practice_progress(self, username, viewer_id=None, *, year=None, timezone_name="UTC"):
        from .practice_progress import profile_progress
        with self.connection() as db:
            # The visibility check and data read share one consistent snapshot.
            db.execute("BEGIN")
            user = self._by_username(db, username)
            if not user["show_progress"] and viewer_id != user["id"]:
                return {"visible": False}
            return profile_progress(db, user["id"], year=year, timezone_name=timezone_name)

    def follow(self, user_id, target_username, active):
        if not isinstance(active, bool):
            raise ValueError("Follow requires a boolean active flag")
        with self.connection() as db:
            self._require_user(db, user_id)
            target = self._by_username(db, target_username)
            if user_id == target["id"]:
                raise ValueError("You cannot follow yourself")
            if target["status"] != "active" and active:
                raise ValueError("This account is not available to follow")
            if active:
                db.execute("INSERT OR IGNORE INTO community_follows(user_id,target_id,created_at) VALUES(?,?,?)", (user_id, target["id"], now()))
            else:
                db.execute("DELETE FROM community_follows WHERE user_id = ? AND target_id = ?", (user_id, target["id"]))
            return self._profile(db, target, user_id)

    def activity(self, username, tab, viewer_id=None, limit=30, offset=0):
        if tab not in ACTIVITY_TABS:
            raise ValueError("Unknown activity tab")
        limit, offset = _page(limit, offset)
        with self.connection() as db:
            user = self._by_username(db, username)
            if tab == "posts":
                table, condition, params = "community_posts", "user_id = ?", [user["id"]]
            elif tab == "replies":
                table, condition, params = "community_replies", "user_id = ?", [user["id"]]
            else:
                table, condition, params = "community_reactions", "user_id = ? AND kind = ?", [user["id"], "like" if tab == "likes" else "repost"]
            total = db.execute(f"SELECT COUNT(*) FROM {table} WHERE {condition}", params).fetchone()[0]
            tie_breaker = "post_id" if tab in ("likes", "reposts") else "id"
            rows = db.execute(f"SELECT * FROM {table} WHERE {condition} ORDER BY created_at DESC, {tie_breaker} DESC LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
            items = []
            post_rows = [row if tab == "posts" else self._post_row(db, row["post_id"]) for row in rows]
            problem_index = {entry["id"]: entry for entry in problems()} if any(post["problem_id"] for post in post_rows) else {}
            for row, post in zip(rows, post_rows):
                item = {"kind": tab, "created_at": row["created_at"], "post": self._post(db, post, viewer_id, problem_index)}
                if tab == "replies":
                    item["reply"] = self._reply(db, row, viewer_id)
                items.append(item)
            return {"items": items, "total": total, "limit": limit, "offset": offset}

    def connections(self, username, kind, viewer_id=None, limit=30, offset=0):
        if kind not in ("following", "followers"):
            raise ValueError("Unknown connection kind")
        limit, offset = _page(limit, offset)
        with self.connection() as db:
            user = self._by_username(db, username)
            source, target = ("user_id", "target_id") if kind == "following" else ("target_id", "user_id")
            total = db.execute(f"SELECT COUNT(*) FROM community_follows WHERE {source} = ?", (user["id"],)).fetchone()[0]
            rows = db.execute(f"SELECT u.* FROM users u JOIN community_follows f ON f.{target} = u.id WHERE f.{source} = ? ORDER BY f.created_at DESC, u.id LIMIT ? OFFSET ?", (user["id"], limit, offset)).fetchall()
            items = []
            for row in rows:
                item = self._public_user(db, row)
                item["viewer"] = self._relationship(db, row["id"], viewer_id)
                items.append(item)
            return {"items": items, "total": total, "limit": limit, "offset": offset}

    def delete_post(self, user_id, post_id):
        with self.connection() as db:
            self._require_user(db, user_id)
            row = self._post_row(db, post_id)
            if not self._can_delete(db, user_id, row["user_id"]):
                raise PermissionError("Only the author or an administrator can delete this post")
            db.execute("DELETE FROM community_posts WHERE id = ?", (post_id,))
            return {"deleted": True, "id": post_id}

    def delete_reply(self, user_id, reply_id):
        with self.connection() as db:
            self._require_user(db, user_id)
            row = db.execute("SELECT * FROM community_replies WHERE id = ?", (reply_id,)).fetchone()
            if row is None:
                raise KeyError("Unknown reply")
            if not self._can_delete(db, user_id, row["user_id"]):
                raise PermissionError("Only the author or an administrator can delete this reply")
            db.execute("DELETE FROM community_replies WHERE id = ?", (reply_id,))
            return {"deleted": True, "id": reply_id}
