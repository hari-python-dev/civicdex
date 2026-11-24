
import re


def improve_complaint(text):

    text = text.strip()

    if not text:
        return ""

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Capitalize first letter
    text = text[0].upper() + text[1:]

    # Add full stop if missing
    if text[-1] not in ".!?":
        text += "."

    return text
