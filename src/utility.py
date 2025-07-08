def pprint_messages(messages: list[dict[str, str]]):
    """Pretty print a pair of messages."""
    for i, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = message.get("content", "")

        print(f"==== {role.capitalize()} Message ====")
        print(content)
        print()
