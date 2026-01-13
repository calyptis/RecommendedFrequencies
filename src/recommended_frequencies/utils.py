import json


def write_jsonl(data: list[dict], filepath: str) -> None:
    """
    Append data to a JSONL file.

    Parameters
    ----------
    data : list[dict]
        List of rows to write.
    filepath : str
        Path to the JSONL file.
    """
    with open(filepath, "a") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def read_jsonl(filepath: str) -> list[dict]:
    """
    Read data from a JSONL file.

    Parameters
    ----------
    filepath : str
        Path to the JSONL file.

    Returns
    -------
    data : list[dict]
        List of rows.
    """
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
