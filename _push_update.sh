#!/bin/bash
# Push the README fix (removes GeoEvolve author info that was de-anonymizing)
set -e
cd "$(dirname "$0")"

# Make sure git identity is still anonymous (local to this repo)
git config user.name "Anonymous"
git config user.email "anonymous@example.com"

# Stage and commit the README change
git add README.md
git -c user.name="Anonymous" -c user.email="anonymous@example.com" \
    commit -m "README: remove author info from GeoEvolve citation (double-blind fix)"

# Verify commit is anonymous
echo ""
echo "── Latest commit author check: ──"
git log -1 --pretty=format:'%h  author=%an <%ae>  %s'
echo ""

# Push
echo ""
echo "── Pushing to GitHub ──"
git push origin main

echo ""
echo "✅ Done — refresh https://github.com/pengtum/GASE/blob/main/README.md to verify"
