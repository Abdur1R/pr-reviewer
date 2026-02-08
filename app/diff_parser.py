# app/diff_parser.py
def parse_diff(diff_text: str):
    results = []
    current_file = None
    current_line = 0

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            current_file = line.split(" b/")[-1]
        elif line.startswith("@@"):
            hunk = line.split(" ")[2]  # +start,count
            current_line = int(hunk.split(",")[0][1:])
        elif line.startswith("+") and not line.startswith("+++"):
            results.append({
                "path": current_file,
                "line": current_line,
                "content": line[1:],
            })
            current_line += 1
        elif not line.startswith("-"):
            current_line += 1

    return results
