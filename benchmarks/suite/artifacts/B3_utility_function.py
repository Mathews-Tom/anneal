from __future__ import annotations

import time


# Embedded test data: two lists of ~260 strings each
# sized to keep runtime in the 500-2000ms window on modern hardware
_NAMES_A: list[str] = [
    "alice", "alexander", "alicia", "alison", "alicia",
    "benjamin", "bennett", "beatrice", "bernadette", "blake",
    "catherine", "carolina", "cassandra", "cecilia", "celeste",
    "daniel", "danielle", "darius", "darwin", "davina",
    "eleanor", "elizabeth", "elliot", "eloise", "elvira",
    "franklin", "francesca", "frederica", "felicity", "flora",
    "gabriel", "gabriella", "genevieve", "georgia", "geraldine",
    "harriet", "harrison", "hayden", "helena", "henrietta",
    "isabella", "isadora", "isidore", "isolde", "imelda",
    "jasper", "jacqueline", "janelle", "jarvis", "jessamine",
    "katherine", "katrina", "kendall", "kennedy", "kieran",
    "laurence", "lavinia", "lazarus", "leander", "leonora",
    "margaret", "marianne", "marlowe", "mathilda", "maximiliana",
    "nathaniel", "natasha", "neil", "nelson", "nerissa",
    "octavia", "oliver", "olympia", "ophelia", "orion",
    "penelope", "percival", "persephone", "petra", "philippa",
    "quinton", "quintessa", "quincey", "queenie", "quentin",
    "rosalind", "rosalinda", "rosamund", "rowena", "roxanne",
    "sebastian", "seraphina", "sigrid", "silvana", "solomon",
    "theodore", "theodosia", "thomas", "thomasina", "titania",
    "ulysses", "una", "uriah", "ursula", "umberto",
    "valentina", "valerian", "vera", "veronica", "victoria",
    "wilhelmina", "winifred", "winston", "wren", "wyatt",
    "xavier", "xena", "xerxes", "xiomara", "xochitl",
    "yolanda", "yolande", "yorick", "yvette", "yvonne",
    "zacharias", "zelda", "zenobia", "zephyr", "zinnia",
    "adriana", "adrienne", "agatha", "agnes", "aigerim",
    "barnabas", "bartholomew", "basil", "beckett", "belinda",
    "calista", "calliope", "camellia", "camille", "candida",
    "damaris", "damian", "daphne", "deidre", "delilah",
    "ebenezer", "edith", "edwina", "effie", "eglantine",
    "fabian", "faith", "fantine", "faye", "felicia",
    "galatea", "gareth", "gawain", "gemma", "gideon",
    "hector", "hedwig", "heloise", "hester", "hildegard",
    "ignatius", "ines", "ingrid", "irene", "iris",
    "jacob", "jocasta", "jocelyn", "joel", "johanna",
    "krista", "kristopher", "kyle", "kylie", "kyra",
    "lancelot", "lara", "lars", "latasha", "layla",
    "mackenzie", "madeline", "magnus", "malachi", "malone",
    "naomi", "narcissa", "nadia", "nigel", "niobe",
    "orlanda", "oswald", "ottilia", "owain", "ozymandias",
    "paloma", "pamela", "pandora", "pascal", "patience",
    "raffael", "ramona", "randolph", "raphaela", "reginald",
    "sabrina", "sacheverell", "salome", "samantha", "sapphire",
    "tabitha", "talitha", "tamsin", "tarquin", "tessa",
    "ulrika", "undine", "upton", "urbane", "urien",
    "violetta", "virginia", "vivian", "vivienne", "voss",
    "warwick", "wendell", "wendy", "wilbur", "wilfred",
]

_NAMES_B: list[str] = [
    "alyce", "alexender", "alecia", "allison", "alisha",
    "benjamine", "bennet", "beatrix", "bernadett", "blaise",
    "catharine", "karoline", "kassandra", "cecilya", "celest",
    "danyal", "daniele", "daryus", "darwinn", "davena",
    "elenor", "elizebeth", "eliott", "elois", "elveera",
    "franklyn", "fransesca", "frederika", "felicite", "flora",
    "gabriell", "gabriela", "geneviv", "georgea", "geraldyne",
    "harriett", "harryson", "haydn", "heleena", "henryetta",
    "izabella", "isadore", "isidor", "isold", "imelda",
    "gasper", "jackline", "janell", "jarvis", "jessamin",
    "katerine", "katreena", "kendal", "kennedee", "kierran",
    "laurens", "lavinia", "lazaros", "leandre", "leonore",
    "margeret", "mariann", "marlo", "matilda", "maximiliane",
    "nathanyal", "natacha", "neil", "nellson", "nerisa",
    "octavya", "olivar", "olimpia", "ofelia", "oreon",
    "penelopy", "percieval", "persephon", "peetra", "phillipa",
    "quinten", "quinteesa", "quincee", "queeny", "quenten",
    "rosalinde", "rosalynd", "rosamunde", "roweena", "roxann",
    "sebastien", "serafina", "sigred", "silvanna", "soloman",
    "theodor", "theodozia", "tomas", "tomasina", "titanya",
    "ulises", "oona", "uriyah", "ursulea", "umbertho",
    "valentyna", "valerien", "verra", "veronika", "victorya",
    "wilhemina", "winnifred", "winsten", "wren", "wyat",
    "xaviar", "zeena", "zerxes", "xiomarra", "xochitil",
    "yolander", "yolanda", "yorrik", "yvett", "ivonne",
    "zacharyus", "zeleda", "zenobya", "zefir", "zinyah",
    "adrienna", "adriann", "agather", "agness", "aygerim",
    "barnabis", "bartholemew", "basill", "becket", "belynda",
    "calister", "calliopee", "camelia", "camil", "candyda",
    "damares", "damyen", "dafne", "deidree", "delyla",
    "ebeneezer", "edyth", "edwinna", "effy", "eglantyne",
    "fabyian", "faithe", "fanteen", "fayye", "felisha",
    "galatea", "garreth", "gawaine", "jemma", "giddeon",
    "hecter", "hedwidge", "heloyse", "hesther", "hildigard",
    "ignatious", "inez", "ingred", "ireen", "irys",
    "jakob", "jocasta", "jocelin", "joell", "johana",
    "kresta", "kristofer", "kile", "kylee", "kira",
    "lancelott", "larra", "larrs", "latashia", "laila",
    "mackensie", "madelyne", "magnes", "malakhi", "malonne",
    "naomie", "narcissa", "nadja", "nigell", "niobee",
    "orlanda", "oswald", "ottilya", "owaine", "ozymandeas",
    "palomma", "pamella", "pandorra", "pascall", "pacience",
    "raffaell", "ramonna", "randolfe", "raphaella", "reginalde",
    "sabrina", "sacheverel", "salomee", "samanthia", "saphire",
    "tabithia", "talithia", "tamzin", "tarquine", "tess",
    "ulricka", "ondine", "uptown", "urbanee", "uryen",
    "violeta", "virginie", "vivyan", "vivienn", "vos",
    "warwicke", "wendall", "wendie", "wilbure", "wilfrid",
]


def _levenshtein(s: str, t: str) -> int:
    """Naive Levenshtein distance — no memoization, no early exit."""
    m = len(s)
    n = len(t)
    # Allocate a fresh matrix every call (no caching)
    prev: list[int] = list(range(n + 1))
    for i in range(1, m + 1):
        curr: list[int] = [i]
        for j in range(1, n + 1):
            # Redundant character comparison (computed twice inside condition)
            if s[i - 1] == t[j - 1]:
                cost = 0
            else:
                cost = 1
            # Three separate list accesses instead of local variables
            curr.append(min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[n]


def _find_close_pairs(list_a: list[str], list_b: list[str], threshold: int) -> list[tuple[str, str, int]]:
    """Return all (a, b, distance) pairs where distance <= threshold.

    Intentional inefficiencies:
    - O(n*m) nested loop with no pruning
    - Recomputes len(a) and len(b) on every iteration
    - Rebuilds result list by concatenation instead of append
    - Checks membership in list_a / list_b with linear scan on each iteration
    """
    results: list[tuple[str, str, int]] = []
    for a in list_a:
        for b in list_b:
            # Redundant length checks that do nothing useful
            if len(a) == 0 or len(b) == 0:
                continue
            # Redundant membership test (linear scan each time)
            if a not in list_a:
                continue
            dist = _levenshtein(a, b)
            if dist <= threshold:
                # String concatenation to build a label (thrown away immediately)
                _label = ""
                for ch in a:
                    _label = _label + ch
                _label = _label + ":" + b
                results.append((a, b, dist))
    return results


def _deduplicate_pairs(pairs: list[tuple[str, str, int]]) -> list[tuple[str, str, int]]:
    """Remove duplicate (a, b) pairs by linear scan instead of using a set."""
    seen: list[tuple[str, str]] = []
    deduped: list[tuple[str, str, int]] = []
    for pair in pairs:
        key = (pair[0], pair[1])
        # O(n) scan instead of O(1) set lookup
        if key not in seen:
            seen.append(key)
            deduped.append(pair)
    return deduped


def run_benchmark() -> tuple[int, float]:
    """Execute the benchmark and return (match_count, elapsed_ms)."""
    start = time.perf_counter()

    raw_pairs = _find_close_pairs(_NAMES_A, _NAMES_B, threshold=2)
    unique_pairs = _deduplicate_pairs(raw_pairs)
    match_count = len(unique_pairs)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return match_count, elapsed_ms


if __name__ == "__main__":
    count, elapsed = run_benchmark()
    print(f"match_count: {count}")
    print(f"execution_time_ms: {elapsed:.2f}")
