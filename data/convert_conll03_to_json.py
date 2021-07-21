def read(file):
    with open(file) as f:
        yield from parse(f)


def parse(lines):
    tokens = []
    for line in lines:
        line = line.strip()
        if line.startswith("-DOCSTART-"):
            continue
        elif not line:
            if tokens:
                yield tokens
                tokens = []
        else:
            cols = line.split(" ")
            token = {
                "form": cols[0],
                "pos_tag": cols[1],
                "chunk_tag": cols[2],
                "ner_tag": cols[3],
            }
            tokens.append(token)
    if tokens:
        yield tokens


def bio2spans(tags):
    span = None
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            assert span["label"] == tag[2:]
            span["end"] = i
        else:
            if span:
                yield span
                span = None
            if tag.startswith("B-"):
                span = {"start": i, "end": i, "label": tag[2:]}
    if span:
        yield span


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--pretty", "-p", action="store_true")
    args = parser.parse_args()

    def _convert(tokens, doc_id):
        doc = {f"{attr}s": [t[attr] for t in tokens] for attr in tokens[0].keys()}
        doc["id"] = doc_id
        doc["entities"] = list(bio2spans(doc["ner_tags"]))
        return doc

    docs = list(map(lambda x: _convert(x[1], x[0]), enumerate(read(args.input_file))))
    indent = 2 if args.pretty else None
    with open(args.output_file, "w") as f:
        json.dump(docs, f, indent=indent)
