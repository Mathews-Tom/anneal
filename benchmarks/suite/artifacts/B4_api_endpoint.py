from __future__ import annotations

import re
import os
import json
import datetime
import hashlib
import logging


# in-memory storage
db = {}
_next_id = 1


class User:
    def __init__(self, id, name, email, age, active):
        self.id = id
        self.name = name
        self.email = email
        self.age = age
        self.active = active
        self.created_at = str(datetime.datetime.now())


def create_user(data):
    global _next_id, db
    try:
        if not validate_email(data["email"]):
            return {"error": "bad email"}, 400
        x = User(
            id=_next_id,
            name=data["name"],
            email=data["email"],
            age=data.get("age", 0),
            active=True
        )
        db[_next_id] = x
        _next_id += 1
        return {"id": x.id, "name": x.name, "email": x.email, "age": x.age, "active": x.active, "created_at": x.created_at}, 201
    except:
        return {"error": "create failed"}, 500


def get_user(user_id):
    try:
        if user_id not in db:
            return {"error": "not found"}, 404
        u = db[user_id]
        d = {}
        d["id"] = u.id
        d["name"] = u.name
        d["email"] = u.email
        d["age"] = u.age
        d["active"] = u.active
        d["created_at"] = u.created_at
        return d, 200
    except:
        return {"error": "get failed"}, 500


def update_user(user_id, data):
    global db
    try:
        if user_id not in db:
            return {"error": "not found"}, 404
        u = db[user_id]
        for k in data:
            if k == "name":
                u.name = data[k]
            if k == "email":
                tmp = data[k]
                if not validate_email(tmp):
                    return {"error": "bad email"}, 400
                u.email = tmp
            if k == "age":
                u.age = data[k]
            if k == "active":
                u.active = data[k]
        db[user_id] = u
        return {"id": u.id, "name": u.name, "email": u.email, "age": u.age, "active": u.active}, 200
    except:
        return {"error": "update failed"}, 500


def delete_user(user_id):
    global db
    try:
        if user_id not in db:
            return {"error": "not found"}, 404
        del db[user_id]
        return {"deleted": user_id}, 200
    except:
        return {"error": "delete failed"}, 500


def list_users(filter):
    global db
    try:
        result = []
        for id in db:
            u = db[id]
            ok = True
            if filter:
                if "active" in filter:
                    if u.active != filter["active"]:
                        ok = False
                if "name" in filter:
                    if filter["name"].lower() not in u.name.lower():
                        ok = False
            if ok:
                d = {}
                d["id"] = u.id
                d["name"] = u.name
                d["email"] = u.email
                d["age"] = u.age
                d["active"] = u.active
                result.append(d)
        return result, 200
    except:
        return [], 500


def validate_email(email):
    try:
        if "@" not in email:
            return False
        parts = email.split("@")
        if len(parts) != 2:
            return False
        local = parts[0]
        domain = parts[1]
        if len(local) == 0:
            return False
        if len(domain) == 0:
            return False
        if "." not in domain:
            return False
        return True
    except:
        return False


def _run_self_test():
    global db, _next_id
    # reset state
    db = {}
    _next_id = 1

    # create
    res, status = create_user({"name": "Alice", "email": "alice@example.com", "age": 30})
    assert status == 201, "create_user should return 201"
    assert res["id"] == 1, "first user id should be 1"
    assert res["name"] == "Alice"

    res2, status2 = create_user({"name": "Bob", "email": "bob@example.com", "age": 25})
    assert status2 == 201
    assert res2["id"] == 2

    # bad email
    bad, bad_status = create_user({"name": "X", "email": "notanemail"})
    assert bad_status == 400, "bad email should return 400"

    # get
    got, gs = get_user(1)
    assert gs == 200
    assert got["name"] == "Alice"

    got_miss, gms = get_user(999)
    assert gms == 404

    # update
    upd, us = update_user(1, {"age": 31, "name": "Alice Updated"})
    assert us == 200
    assert upd["age"] == 31
    assert upd["name"] == "Alice Updated"

    upd_miss, ums = update_user(999, {"name": "Ghost"})
    assert ums == 404

    # list all
    all_users, ls = list_users({})
    assert ls == 200
    assert len(all_users) == 2

    # list filtered
    active_users, _ = list_users({"active": True})
    assert len(active_users) == 2

    # deactivate bob
    update_user(2, {"active": False})
    active_after, _ = list_users({"active": True})
    assert len(active_after) == 1

    # name filter
    named, _ = list_users({"name": "alice"})
    assert len(named) == 1

    # delete
    del_res, ds = delete_user(2)
    assert ds == 200
    assert del_res["deleted"] == 2
    after_del, _ = list_users({})
    assert len(after_del) == 1

    del_miss, dms = delete_user(999)
    assert dms == 404

    # validate_email
    assert validate_email("test@test.com") is True
    assert validate_email("bad") is False
    assert validate_email("@nodomain.com") is False
    assert validate_email("noat.com") is False

    return True


if __name__ == "__main__":
    result = _run_self_test()
    print("self_test_passed:", result)
