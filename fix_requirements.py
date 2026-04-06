# Save this as fix_requirements.py in the same folder as requirements.txt
lines = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        # Check if line looks like "package=version=build" or "package=version"
        if '=' in line and '==' not in line:
            parts = line.split('=')
            # Keep only the package name and the version
            if len(parts) >= 2:
                lines.append(f"{parts[0]}=={parts[1]}")
            else:
                lines.append(line)
        else:
            lines.append(line)

with open("requirements.txt", "w") as f:
    f.write('\n'.join(lines) + '\n')

print("Fixed requirements.txt syntax.")
