#!/bin/bash
# ============================================================
# SEGA code release: stage, anonymous-commit, and push to GitHub
# Run this script from inside the GASE/ folder on your Mac.
# ============================================================

set -e

cd "$(dirname "$0")"
echo "Working in: $(pwd)"
echo ""

# Step 1: clear stale git lock from prior incomplete operation
if [ -f .git/index.lock ]; then
    rm -f .git/index.lock
    echo "✓ Removed stale .git/index.lock"
fi

# Step 2: remove the leftover _test file
rm -f _test
echo "✓ Removed stray _test file"

# Step 3: configure anonymous git identity LOCAL to this repo only
git config user.name "Anonymous"
git config user.email "anonymous@example.com"
echo "✓ Local git identity set to Anonymous (will not affect your global identity)"

# Step 4: stage everything
git add -A
echo "✓ Staged $(git diff --cached --name-only | wc -l | tr -d ' ') files"

# Step 5: initial commit with the anonymous identity
git -c user.name="Anonymous" -c user.email="anonymous@example.com" \
    commit -m "Initial SEGA code release for double-blind review"
echo "✓ Created initial commit"

# Step 6: verify the commit is truly anonymous
echo ""
echo "── Commit author check (should say Anonymous): ──"
git log --pretty=format:'%h  author=%an <%ae>  %s' -1
echo ""
echo ""

# Step 7: push to GitHub
echo "── Pushing to GitHub (origin/main) ──"
echo "If this is the first push you may need to authenticate (gh auth login or password)."
git push -u origin main

echo ""
echo "════════════════════════════════════════════"
echo "  ✅ DONE"
echo "════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Visit https://github.com/pengtum/GASE — confirm files are there"
echo "     (your username 'pengtum' is visible on GitHub, but anonymous.4open will hide it)"
echo "  2. Go to https://anonymous.4open.science/anonymize"
echo "  3. Paste: https://github.com/pengtum/GASE"
echo "  4. It will give you an anonymous URL like:"
echo "       https://anonymous.4open.science/r/GASE-XXXX/"
echo "  5. Put that anonymous URL into the manuscript's Data Availability section."
