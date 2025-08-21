def pprint_messages(messages: list[dict[str, str]]):
    """Pretty print a pair of messages."""
    for i, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = message.get("content", "")

        print(f"==== {role.capitalize()} Message ====")
        print(content)
        print()


def sort_key_new(key):
    if key.startswith("lima_"):
        try:
            return int(key.split("_")[1])
        except (IndexError, ValueError):
            return key
    return key


# Sort the combined results numerically by the number after "lima_"
def sort_key(item):
    key = item[0]

    print(key)
    if key.startswith("lima_"):
        try:
            return int(key.split("_")[1])
        except (IndexError, ValueError):
            return key
    return key