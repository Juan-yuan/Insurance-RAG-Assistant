import os

_LOADED = False


def _candidate_paths():
    root = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(root, '.env'),
        os.path.join(root, 'integrated_qa_system', '.env'),
    ]


def _unescape_quoted(inner):
    out = []
    index = 0
    while index < len(inner):
        char = inner[index]
        if char == '\\' and index + 1 < len(inner):
            nxt = inner[index + 1]
            if nxt == 'n':
                out.append('\n')
            elif nxt == 't':
                out.append('\t')
            elif nxt == '"':
                out.append('"')
            elif nxt == '\\':
                out.append('\\')
            else:
                out.append('\\')
                out.append(nxt)
            index += 2
            continue
        out.append(char)
        index += 1
    return ''.join(out)


def _parse_env_file(path):
    with open(path, encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, raw_value = line.partition('=')
            key = key.strip()
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                value = _unescape_quoted(value[1:-1])
            elif len(value) >= 2 and value[0] == "'" and value[-1] == "'":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


def load_env():
    global _LOADED
    if _LOADED:
        return
    for path in _candidate_paths():
        if os.path.isfile(path):
            _parse_env_file(path)
    lowercase_key = os.environ.get('openai_api_key', '').strip()
    if lowercase_key and not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = lowercase_key
    _LOADED = True


def require_env(name):
    load_env()
    value = os.environ.get(name)
    if value is None or value == '':
        raise RuntimeError(f'Missing required runtime configuration: {name}')
    return value


def require_int(name):
    return int(require_env(name))
