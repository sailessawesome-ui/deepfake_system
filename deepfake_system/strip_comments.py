import os
import re

def remove_explanations_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    out_lines = []
    in_docstring = False
    docstring_quote = ""

    for line in lines:
        stripped = line.strip()

        # Handle docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote_type = stripped[:3]
                # Single-line docstring
                if len(stripped) >= 6 and stripped.endswith(quote_type):
                    continue
                # Multi-line docstring
                else:
                    in_docstring = True
                    docstring_quote = quote_type
                    continue
        else:
            if docstring_quote in stripped:
                # End of multi-line docstring
                in_docstring = False
                # If there's code after the docstring on the same line, this simple script might lose it,
                # but standard formatting puts the closing quotes on their own line or end of the docstring.
            continue
            
        # Handle full-line comments
        if stripped.startswith("#"):
            # Keep type hints or linter directives
            if "type: ignore" in stripped or "noqa" in stripped:
                out_lines.append(line)
            continue
            
        # Handle inline comments
        if "#" in line:
            # We must be careful not to remove '#' inside strings.
            # A very simple heuristic: if it's not a URL, and there is a '#' with a space before it.
            if " #" in line:
                idx = line.find(" #")
                comment_part = line[idx:]
                if "type: ignore" in comment_part or "noqa" in comment_part:
                    out_lines.append(line)
                else:
                    out_lines.append(line[:idx] + "\n")
                continue

        out_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        if ".venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                print(f"Cleaning {filepath}")
                remove_explanations_from_file(filepath)

if __name__ == "__main__":
    base_dir = r"c:\Users\Sailess Raj\Downloads\deepfake_system\deepfake_system"
    for d in ["app", "data", "models", "scripts"]:
        process_directory(os.path.join(base_dir, d))
