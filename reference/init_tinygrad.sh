#!/usr/bin/env bash
set -e

# Create reference folder
mkdir -p reference

# Loop through all items in the repo root
for item in * .*; do
    # Skip special dirs
    if [[ "$item" == "." || "$item" == ".." || "$item" == ".git" || "$item" == ".github" ]]; then
        continue
    fi
    
    # Skip the reference folder itself (avoid recursive move)
    if [[ "$item" == "reference" ]]; then
        continue
    fi

    # Move to reference/
    mv "$item" reference/ 2>/dev/null || true
    echo "Moved $item to reference/"
done

echo "Done. Everything except .git/ and .github/ is now in reference/."

