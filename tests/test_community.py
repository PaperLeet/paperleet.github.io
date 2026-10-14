"""Community persistence, authorization and public-profile regressions."""
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

import pytest

from paperleet.community import CommunityService


@pytest.fixture
def community(tmp_path):
    path = tmp_path / "community.sqlite3"
    # Authentication is tested separately. These represent already verified users.
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE users (
            id TEXT PRIMARY KEY, username TEXT UNIQUE, display_name TEXT NOT NULL,
            avatar_color TEXT NOT NULL, bio TEXT NOT NULL DEFAULT '', institution_id TEXT,
            country TEXT NOT NULL DEFAULT '', show_institution INTEGER DEFAULT 0,
            show_country INTEGER DEFAULT 0, show_progress INTEGER DEFAULT 1,
            role TEXT DEFAULT 'member', status TEXT DEFAULT 'active', created_at TEXT NOT NULL
        )""")
        for user_id, username, name, role in (("a", "alice", "Alice", "member"),
                                               ("b", "bob", "Bob", "member"),
                                               ("c", "carol", "Carol", "member"),
                                               ("admin", "moderator", "Moderator", "admin")):
            db.execute("INSERT INTO users(id,username,display_name,avatar_color,institution_id,country,role,created_at) VALUES(?,?,?,?,?,?,?,?)",
                       (user_id, username, name, "#527ad9", "test-university", "US", role, "2026-01-01T00:00:00+00:00"))
    return CommunityService(path, institutions={"test-university": {
        "id": "test-university", "name": "测试大学", "name_en": "Test University"}})


def make_post(community, user="a", **extra):
    return community.create_post(user, {"title": "Residual connections", "body": "How does the **gradient** flow?", "category": "papers", **extra})


def test_module_context_is_canonical_public_and_persists_every_projection(community):
    post = make_post(community, problem_id="architecture/attention", problem={"title": "Spoofed", "number": 999})
    assert post["problem_id"] == "architecture/attention"
    assert post["problem"] == {"id": "architecture/attention", "number": 22,
                               "title": "缩放点积注意力", "title_en": "Scaled Dot-Product Attention"}
    community.reply("b", post["id"], {"body": "What happens for a fully masked row?"})
    community.set_reaction("b", post["id"], "like", True)
    restarted = CommunityService(community.db_path, community.institutions)
    for projection in (restarted.get_post(post["id"]), restarted.list_posts()["items"][0],
                       restarted.activity("alice", "posts")["items"][0]["post"],
                       restarted.activity("bob", "replies")["items"][0]["post"],
                       restarted.activity("bob", "likes")["items"][0]["post"]):
        assert projection["problem"] == post["problem"]
        assert projection["problem_id"] == post["problem_id"]
        assert_private_fields_hidden(projection["author"])
    assert make_post(community)["problem"] is None


def test_module_scope_intersects_following_category_search_and_pagination(community):
    wanted = [make_post(community, problem_id="architecture/attention", category="questions", title=f"Mask gradient {i}") for i in range(3)]
    make_post(community, problem_id="finetuning/lora", category="questions", title="Mask gradient elsewhere")
    make_post(community, problem_id="architecture/attention", category="papers", title="Mask gradient paper")
    make_post(community, problem_id="architecture/attention", category="questions", title="Unrelated topic", body="No matching text")
    make_post(community, "c", problem_id="architecture/attention", category="questions", title="Mask gradient unfollowed")
    make_post(community, category="questions", title="Mask gradient without association")
    community.follow("b", "alice", True)
    scope = dict(viewer_id="b", problem_id="architecture/attention", following=True, category="questions", query="mask gradient", limit=2)
    first, second = community.list_posts(**scope), community.list_posts(**scope, offset=2)
    assert first["total"] == second["total"] == 3
    assert len(first["items"]) == 2 and len(second["items"]) == 1
    assert {p["id"] for p in first["items"] + second["items"]} == {p["id"] for p in wanted}
    assert community.list_posts(problem_id="architecture/attention", query="%_")["total"] == 0
    assert community.list_posts(problem_id="retired/module")["total"] == 0


@pytest.mark.parametrize("identifier", [22, True, {}, [], "22", "attention", "architecture/unknown", "../attention", "architecture/attention/extra", "", "a" * 201])
def test_module_posts_reject_unknown_or_noncanonical_associations(community, identifier):
    with pytest.raises(ValueError):
        make_post(community, problem_id=identifier)
    assert community.list_posts()["total"] == 0


@pytest.mark.parametrize("identifier", [22, True, {}, [], "22", "../attention", "architecture/attention' OR 1=1--", "", "a" * 201])
def test_module_filters_reject_malformed_identifiers(community, identifier):
    with pytest.raises(ValueError):
        community.list_posts(problem_id=identifier)


def test_module_schema_upgrade_preserves_old_posts_and_is_concurrency_safe(community):
    original = make_post(community)
    with community.connection() as db:
        db.execute("DROP INDEX community_posts_problem")
        db.execute("ALTER TABLE community_posts DROP COLUMN problem_id")
    with ThreadPoolExecutor(max_workers=4) as workers:
        services = list(workers.map(lambda _: CommunityService(community.db_path, community.institutions), range(8)))
    for service in services:
        migrated = service.get_post(original["id"])
        assert migrated["body"] == original["body"]
        assert migrated["problem_id"] is None and migrated["problem"] is None
    assert make_post(services[0], problem_id="finetuning/lora")["problem"]["id"] == "finetuning/lora"


def test_retired_module_threads_remain_readable_without_number_redirect(community, monkeypatch):
    post = make_post(community, problem_id="architecture/attention")
    monkeypatch.setattr("paperleet.community.problems", lambda: [])
    for projection in (community.get_post(post["id"]),
                       community.list_posts(problem_id="architecture/attention")["items"][0],
                       community.activity("alice", "posts")["items"][0]["post"]):
        assert projection["problem"] == {"id": "architecture/attention", "title": "architecture/attention", "retired": True}


def assert_private_fields_hidden(user):
    assert user["display_name"] and user["avatar_color"]
    assert "institution" not in user and "country" not in user
    for field in ("email", "ip", "institution_id", "role", "status", "show_country", "show_institution"):
        assert field not in user


def test_empty_community_is_not_seeded(community):
    assert community.list_posts()["items"] == []
    assert community.activity("alice", "posts")["total"] == 0
    assert community.profile("alice")["counts"] == {
        "posts": 0, "replies": 0, "likes": 0, "reposts": 0, "followers": 0, "following": 0,
        "likes_received": 0, "reposts_received": 0}


def test_profile_received_reactions_exclude_self_and_track_removal(community):
    first, second = make_post(community), make_post(community, title="Another post")
    for post in (first, second):
        for actor in ("a", "b", "c"):
            for kind in ("like", "repost"):
                community.set_reaction(actor, post["id"], kind, True)
                community.set_reaction(actor, post["id"], kind, True)
    counts = community.profile("alice")["counts"]
    assert counts["likes_received"] == counts["reposts_received"] == 4
    assert counts["likes"] == counts["reposts"] == 2
    assert community.profile("bob")["counts"]["likes_received"] == 0
    community.set_reaction("b", first["id"], "like", False)
    assert community.profile("alice")["counts"]["likes_received"] == 3
    community.delete_post("a", second["id"])
    counts = community.profile("alice")["counts"]
    assert counts["likes_received"] == 1 and counts["reposts_received"] == 2


def test_author_identity_is_taken_from_session_actor_and_persists(community):
    post = make_post(community, user_id="b", author={"id": "b", "display_name": "Spoofed"})
    assert post["author"]["id"] == "a"
    assert post["author"]["display_name"] == "Alice"
    assert post["viewer"]["can_delete"] is True
    assert_private_fields_hidden(post["author"])
    restarted = CommunityService(community.db_path, community.institutions)
    assert restarted.get_post(post["id"])["title"] == post["title"]
    assert restarted.get_post(post["id"])["viewer"]["can_delete"] is False


def test_reply_and_reactions_appear_in_profile_with_distinct_viewer_flags(community):
    post = make_post(community)
    reply = community.reply("b", post["id"], {"body": "The identity branch helps.", "user_id": "a"})
    assert reply["author"]["id"] == "b"
    for _ in range(2):
        community.set_reaction("b", post["id"], "like", True)
        community.set_reaction("b", post["id"], "repost", True)
    detail = community.get_post(post["id"], "b")
    assert detail["counts"] == {"likes": 1, "reposts": 1, "replies": 1}
    assert detail["viewer"] == {"liked": True, "reposted": True, "can_delete": False}
    assert detail["replies"][0]["can_delete"] is True
    assert not community.get_post(post["id"], "a")["viewer"]["liked"]
    profile = community.profile("bob", "b")
    assert profile["counts"]["replies"] == profile["counts"]["likes"] == profile["counts"]["reposts"] == 1
    assert profile["viewer"]["is_self"] is True
    assert community.activity("bob", "replies")["items"][0]["reply"]["id"] == reply["id"]
    for tab in ("likes", "reposts"):
        entry = community.activity("bob", tab)["items"][0]
        assert entry["kind"] == tab and entry["post"]["id"] == post["id"]
    for _ in range(2):
        community.set_reaction("b", post["id"], "like", False)
    assert community.profile("bob")["counts"]["likes"] == 0
    assert community.activity("bob", "likes")["items"] == []
    assert community.get_post(post["id"])["counts"]["reposts"] == 1


def test_follow_is_idempotent_and_filters_feed_by_authors(community):
    wanted = make_post(community, "a")
    make_post(community, "c", title="Unfollowed author")
    for _ in range(2):
        profile = community.follow("b", "alice", True)
    assert profile["counts"]["followers"] == 1
    assert profile["viewer"]["following"] is True
    assert community.profile("bob")["counts"]["following"] == 1
    assert [p["id"] for p in community.list_posts("b", following=True)["items"]] == [wanted["id"]]
    assert community.connections("alice", "followers")["items"][0]["username"] == "bob"
    assert community.connections("bob", "following", "b")["items"][0]["viewer"]["following"] is True
    with pytest.raises(ValueError, match="yourself"):
        community.follow("a", "alice", True)
    with pytest.raises(PermissionError):
        community.list_posts(following=True)
    for _ in range(2):
        community.follow("b", "alice", False)
    assert community.connections("bob", "following")["items"] == []
    assert community.profile("alice")["counts"]["followers"] == 0


def test_privacy_is_enforced_on_every_public_projection_including_self(community):
    post = make_post(community)
    community.reply("a", post["id"], {"body": "A follow-up"})
    community.set_reaction("b", post["id"], "like", True)
    community.follow("b", "alice", True)
    for user in (
        community.profile("alice", "a"),
        community.get_post(post["id"], "a")["author"],
        community.get_post(post["id"], "a")["replies"][0]["author"],
        community.activity("bob", "likes", "a")["items"][0]["post"]["author"],
        community.connections("bob", "following", "a")["items"][0],
    ):
        assert_private_fields_hidden(user)
    with community.connection() as db:
        db.execute("UPDATE users SET show_institution = 1, show_country = 1 WHERE id = 'a'")
    visible = community.profile("alice")
    assert visible["institution"]["id"] == "test-university"
    assert visible["country"] == "US"
    with community.connection() as db:
        db.execute("UPDATE users SET show_institution = 0, show_country = 0 WHERE id = 'a'")
    assert "Test University" not in json.dumps(community.get_post(post["id"]), ensure_ascii=False)
    assert_private_fields_hidden(community.profile("alice"))


@pytest.mark.parametrize("payload", [None, [], {}, {"title": " ", "body": "x"},
    {"title": "x", "body": " "}, {"title": "x" * 181, "body": "x"},
    {"title": "x", "body": "x" * 20001}, {"title": "x", "body": "x", "category": "admin"}])
def test_posts_validate_contract_before_persistence(community, payload):
    with pytest.raises(ValueError):
        community.create_post("a", payload)
    assert community.list_posts()["total"] == 0


def test_parent_reply_must_exist_on_same_post(community):
    first, second = make_post(community), make_post(community, title="Second post")
    parent = community.reply("b", first["id"], {"body": "Parent"})
    with pytest.raises(ValueError, match="belong"):
        community.reply("a", second["id"], {"body": "Wrong post", "parent_id": parent["id"]})
    with pytest.raises(ValueError, match="belong"):
        community.reply("a", first["id"], {"body": "Missing parent", "parent_id": "missing"})
    child = community.reply("a", first["id"], {"body": "Child", "parent_id": parent["id"]})
    assert child["parent_id"] == parent["id"]
    community.delete_reply("b", parent["id"])
    detail = community.get_post(first["id"])
    assert len(detail["replies"]) == 1 and detail["replies"][0]["parent_id"] is None


def test_inactive_actor_cannot_mutate_any_community_state(community):
    post = make_post(community)
    reply = community.reply("a", post["id"], {"body": "Existing reply"})
    with community.connection() as db:
        db.execute("UPDATE users SET status = 'suspended' WHERE id = 'a'")
    actions = [lambda: make_post(community), lambda: community.reply("a", post["id"], {"body": "x"}),
               lambda: community.set_reaction("a", post["id"], "like", True),
               lambda: community.follow("a", "bob", True),
               lambda: community.delete_post("a", post["id"]), lambda: community.delete_reply("a", reply["id"])]
    for action in actions:
        with pytest.raises(PermissionError, match="not active"):
            action()
    assert community.get_post(post["id"], "a")["viewer"]["can_delete"] is False
    with pytest.raises(PermissionError):
        make_post(community, None)


def test_delete_checks_ownership_and_cascades_to_activity(community):
    post = make_post(community)
    reply = community.reply("b", post["id"], {"body": "Reply"})
    community.set_reaction("b", post["id"], "like", True)
    community.set_reaction("b", post["id"], "repost", True)
    with pytest.raises(PermissionError):
        community.delete_post("b", post["id"])
    with pytest.raises(PermissionError):
        community.delete_reply("a", reply["id"])
    assert community.get_post(post["id"], "admin")["viewer"]["can_delete"] is True
    community.delete_post("admin", post["id"])
    with pytest.raises(KeyError):
        community.get_post(post["id"])
    for tab in ("posts", "replies", "likes", "reposts"):
        assert community.activity("bob", tab)["total"] == 0
    assert community.profile("alice")["counts"]["posts"] == 0
    with community.connection() as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert db.execute("SELECT COUNT(*) FROM community_replies").fetchone()[0] == 0


def test_concurrent_duplicate_actions_have_one_persisted_effect(community):
    post = make_post(community)

    def act(_):
        community.set_reaction("b", post["id"], "like", True)
        community.set_reaction("b", post["id"], "repost", True)
        community.follow("b", "alice", True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(act, range(24)))
    assert community.get_post(post["id"])["counts"] == {"likes": 1, "reposts": 1, "replies": 0}
    assert community.profile("alice")["counts"]["followers"] == 1
    assert community.profile("bob")["counts"]["following"] == 1


def test_search_categories_and_pagination_are_persisted_and_literal(community):
    make_post(community, title="论文讨论 100%", category="papers")
    make_post(community, title="PyTorch question", category="questions")
    assert community.list_posts(query="论文")["total"] == 1
    assert community.list_posts(query="%")["total"] == 1
    assert community.list_posts(query="PYTORCH", category="questions")["total"] == 1
    page = community.list_posts(limit=1, offset=1)
    assert page["total"] == 2 and len(page["items"]) == 1
    for kwargs in ({"limit": 0}, {"limit": 101}, {"offset": -1}, {"category": "' OR 1=1"}):
        with pytest.raises(ValueError):
            community.list_posts(**kwargs)
    with pytest.raises(ValueError):
        community.set_reaction("a", page["items"][0]["id"], "like", "false")
    with pytest.raises(ValueError):
        community.follow("a", "bob", "false")
