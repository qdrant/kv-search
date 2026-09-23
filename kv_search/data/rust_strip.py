"""Deterministic Rust signature extractor.

strip_file(src) -> str : returns the source with function/method bodies (and
const/static initializer blocks and expression bodies) removed, but every item
signature, type definition, doc comment and attribute kept.

No dependencies, works offline. It is a lexer + recursive brace walker, not a
full Rust parser, so it favours robustness over perfection: anything it is unsure
about is kept verbatim rather than dropped.

Rules:
  fn / async fn / const fn / ... bodies      -> replaced with ` { ... }`
  struct / enum / union bodies (fields)      -> kept verbatim
  trait / impl / mod bodies                  -> recursed into (inner fn bodies stripped)
  const/static/let initializer `= {...}`     -> replaced with ` { ... }`
  macro invocations `name! { ... }`          -> kept verbatim
  macro_rules! name { ... }                  -> kept verbatim
"""

from __future__ import annotations

BODY_STUB = " { ... }"


def _skip_ws_and_trivia(src: str, i: int, n: int) -> int:
    """Skip whitespace only (comments handled inline so they're copied)."""
    while i < n and src[i] in " \t\r\n":
        i += 1
    return i


def _scan_trivia(src: str, i: int, n: int):
    """If src[i] begins a string/char/comment/raw-string, return the index just
    past it. Otherwise return None. Lifetimes ('a) are NOT treated as chars."""
    c = src[i]
    # line comment
    if c == "/" and i + 1 < n and src[i + 1] == "/":
        j = i + 2
        while j < n and src[j] != "\n":
            j += 1
        return j
    # block comment (nested, as in Rust)
    if c == "/" and i + 1 < n and src[i + 1] == "*":
        depth = 1
        j = i + 2
        while j < n and depth:
            if src[j] == "/" and j + 1 < n and src[j + 1] == "*":
                depth += 1
                j += 2
            elif src[j] == "*" and j + 1 < n and src[j + 1] == "/":
                depth -= 1
                j += 2
            else:
                j += 1
        return j
    # raw string:  r"..."  r#"..."#  br#"..."#  etc.
    if c in "rb":
        j = i
        if src[j] == "b":
            j += 1
        if j < n and src[j] == "r":
            j += 1
            hashes = 0
            while j < n and src[j] == "#":
                hashes += 1
                j += 1
            if j < n and src[j] == '"':
                j += 1
                closing = '"' + "#" * hashes
                end = src.find(closing, j)
                return (end + len(closing)) if end != -1 else n
    # normal / byte string
    if c == '"' or (c == "b" and i + 1 < n and src[i + 1] == '"'):
        j = i + (1 if c == '"' else 2)
        while j < n:
            if src[j] == "\\":
                j += 2
                continue
            if src[j] == '"':
                return j + 1
            j += 1
        return n
    # char literal vs lifetime:  'a'  '\n'  vs  'lifetime
    if c == "'":
        # char literal if it closes within a few chars
        if i + 1 < n and src[i + 1] == "\\":
            k = i + 2
            while k < n and k < i + 8 and src[k] != "'":
                k += 1
            if k < n and src[k] == "'":
                return k + 1
        elif i + 2 < n and src[i + 2] == "'":
            return i + 3
        # else: lifetime -> not trivia
        return None
    return None


# keywords that mark an item whose `{...}` block we want to KEEP verbatim
KEEP_KEYWORDS = ("struct", "enum", "union")
# keywords whose block we RECURSE into
RECURSE_KEYWORDS = ("trait", "impl", "mod")


def _classify_header(header: str, drop_tests: bool) -> str:
    """Classify the item whose body-opening `{` we just hit, based on the header
    text accumulated since the previous item boundary."""
    # attribute/doc lines matter for cfg(test) detection; keep them for that check
    attr_blob = "".join(header.split())  # whitespace-free, for cfg(test)/#[test]
    if drop_tests and ("cfg(test)" in attr_blob or "#[test]" in attr_blob):
        return "drop"

    # strip attributes / doc comments / line comments from keyword consideration
    toks = []
    for line in header.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith("//"):
            continue
        toks.append(s)
    flat = " ".join(toks)
    words = flat.replace("(", " ( ").replace("<", " < ").split()
    wordset = set(w.strip("!") for w in words)

    # macro invocation:  something!  {  }
    stripped = flat.rstrip()
    if stripped.endswith("!"):
        return "verbatim"
    if "macro_rules!" in flat:
        return "verbatim"
    # initializer block:  `= {` (const/static/let value)  -> strip
    if "=" in flat and "fn" not in wordset:
        return "strip"
    if "fn" in wordset:
        return "strip"
    for kw in KEEP_KEYWORDS:
        if kw in wordset:
            return "keep"
    for kw in RECURSE_KEYWORDS:
        if kw in wordset:
            return "recurse"
    # default: keep verbatim (safer than dropping content)
    return "verbatim"


def _find_matching_brace(src: str, i: int, n: int) -> int:
    """i points at '{'. Return index just past the matching '}'."""
    depth = 0
    while i < n:
        t = _scan_trivia(src, i, n)
        if t is not None:
            i = t
            continue
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _process(src: str, i: int, end: int, out: list, n: int, drop_tests: bool) -> int:
    """Process the item sequence in [i, end); append transformed text to `out`.
    Item headers (doc comments + attributes + declaration) are buffered in
    `pending` so a whole item can be dropped before any of it is emitted.
    Returns the index reached (== end for a block, or the close brace pos)."""
    pending: list = []

    def flush():
        out.extend(pending)
        pending.clear()

    while i < end:
        t = _scan_trivia(src, i, n)
        if t is not None:
            pending.append(src[i:t])
            i = t
            continue
        c = src[i]
        if c == "}":
            flush()
            return i
        if c == ";":
            pending.append(c)
            flush()
            i += 1
            continue
        if c == "{":
            header = "".join(pending)
            kind = _classify_header(header, drop_tests)
            block_end = _find_matching_brace(src, i, n)
            if kind == "drop":
                pending.clear()  # discard header (doc/attrs) and the block
            elif kind == "strip":
                flush()
                out.append(BODY_STUB)
            elif kind == "keep" or kind == "verbatim":
                flush()
                out.append(src[i:block_end])
            elif kind == "recurse":
                flush()
                out.append("{")
                inner_end = block_end - 1  # position of matching '}'
                reached = _process(src, i + 1, inner_end, out, n, drop_tests)
                out.append(src[reached:block_end])  # the closing '}'
            i = block_end
            continue
        pending.append(c)
        i += 1
    flush()
    return i


def strip_file(src: str, drop_tests: bool = True) -> str:
    out: list = []
    _process(src, 0, len(src), out, len(src), drop_tests)
    text = "".join(out)
    # collapse runs of >2 blank lines left behind
    lines = text.split("\n")
    cleaned = []
    blanks = 0
    for ln in lines:
        if ln.strip() == "":
            blanks += 1
            if blanks > 1:
                continue
        else:
            blanks = 0
        cleaned.append(ln.rstrip())
    return "\n".join(cleaned).strip() + "\n"


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        with open(path, encoding="utf-8", errors="ignore") as f:
            print(f"===== {path} =====")
            print(strip_file(f.read()))
