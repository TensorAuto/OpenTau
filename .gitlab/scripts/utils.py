import re


def grep_file(file: str, pattern: str, processor=None) -> list:
    processor = processor or (lambda x: x)
    values = []
    with open(file) as f:
        for line in f.readlines():
            match = re.search(pattern, line)
            if not match:
                continue
            values.append(processor(match.group(1)))
    return values
