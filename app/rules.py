# app/rules.py
def run_rules(changes: list[dict]) -> list[dict]:
    issues = []

    for c in changes:
        line = c["content"]

        if "TODO" in line:
            issues.append({
                "path": c["path"],
                "line": c["line"],
                "comment": "TODO left in code; please address before merge.",
            })

        if "print(" in line:
            issues.append({
                "path": c["path"],
                "line": c["line"],
                "comment": "print() found; consider proper logging.",
            })

        if "except:" in line:
            issues.append({
                "path": c["path"],
                "line": c["line"],
                "comment": "Bare except detected; catch specific exceptions.",
            })

    return issues
