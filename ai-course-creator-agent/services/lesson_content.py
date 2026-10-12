def display_text(value):
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return "" if text.casefold() in {"null", "none"} else text

def display_items(value):
    if not isinstance(value, list):
        return []

    items = []
    for item in value:
        if isinstance(item, list):
            items.extend(display_items(item))
        else:
            text = display_text(item)
            if text:
                items.append(text)
    return items
