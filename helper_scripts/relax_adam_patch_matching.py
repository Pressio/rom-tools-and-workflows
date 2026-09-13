from pathlib import Path

path = Path("helper_scripts/address_adam_review.py")
text = path.read_text()
old = '''def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)
'''
new = '''def replace_once(text, old, new, label):
    count = text.count(old)
    if count < 1:
        raise RuntimeError(f"{label}: expected at least one match, found {count}")
    return text.replace(old, new, 1)
'''
if old not in text:
    raise RuntimeError("replace_once definition not found")
path.write_text(text.replace(old, new, 1))
