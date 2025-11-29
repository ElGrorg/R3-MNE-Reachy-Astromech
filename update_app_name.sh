#!/bin/bash

# Reachy Mini App Setup Script
# This script sets up a new Reachy Mini app based on the example template

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🤖 Reachy Mini App Setup Script${NC}"
echo "=================================="

# Get repository name from git
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Extract repository name from git remote or directory name
REPO_NAME=$(basename -s .git $(git config --get remote.origin.url) 2>/dev/null || basename "$(pwd)")

if [ -z "$REPO_NAME" ]; then
    echo -e "${RED}Error: Could not determine repository name${NC}"
    exit 1
fi

echo -e "${YELLOW}Repository name: $REPO_NAME${NC}"

# Convert repo name to different formats needed
# Python package name: lowercase, hyphens to underscores (my-app -> my_app)
PACKAGE_NAME=$(echo "$REPO_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/-/_/g')

# Python class name: PascalCase, remove hyphens/underscores (my-app -> MyApp)
CLASS_NAME=$(echo "$REPO_NAME" | sed 's/[-_]/ /g' | sed 's/\b\(.\)/\U\1/g' | sed 's/ //g')

# Entry point name: keep original repo format for CLI commands
ENTRY_POINT_NAME="$REPO_NAME"

echo -e "${YELLOW}Package name: $PACKAGE_NAME${NC}"
echo -e "${YELLOW}Class name: ${CLASS_NAME}App${NC}"
echo -e "${YELLOW}Entry point: $ENTRY_POINT_NAME${NC}"

echo -e "${YELLOW}Setting up app from repo: $REPO_NAME${NC}"

# Step 1: Update pyproject.toml
echo "📝 Updating pyproject.toml..."
if [ -f "pyproject.toml" ]; then
    # Update name (use package name format)
    sed -i.bak "s/name = \"reachy_mini_app_example\"/name = \"$PACKAGE_NAME\"/" pyproject.toml
    
    # Update entry point (entry point name can have hyphens, points to package.main:ClassApp)
    sed -i.bak "s/reachy_mini_app_example = \"reachy_mini_app_example.main:ExampleApp\"/$ENTRY_POINT_NAME = \"$PACKAGE_NAME.main:${CLASS_NAME}App\"/" pyproject.toml
    
    echo -e "${GREEN}✓ pyproject.toml updated${NC}"
else
    echo -e "${RED}Error: pyproject.toml not found${NC}"
    exit 1
fi

# Step 2: Update README.md
echo "📝 Updating README.md..."
if [ -f "README.md" ]; then
    # Replace example references with new names (use repo name for display)
    sed -i.bak "s/Reachy Mini App Example/${REPO_NAME^} App/g" README.md
    sed -i.bak "s/reachy_mini_app_example/$PACKAGE_NAME/g" README.md
    
    echo -e "${GREEN}✓ README.md updated${NC}"
else
    echo -e "${YELLOW}Warning: README.md not found, skipping...${NC}"
fi

# Step 3: Update index.html
echo "📝 Updating index.html..."
if [ -f "index.html" ]; then
    # Update title and content (use repo name for display)
    sed -i.bak "s/Reachy Mini New App Tutorial/${REPO_NAME^} App/g" index.html
    sed -i.bak "s/reachy_mini_new_app_tuto/$PACKAGE_NAME/g" index.html
    
    echo -e "${GREEN}✓ index.html updated${NC}"
else
    echo -e "${YELLOW}Warning: index.html not found, skipping...${NC}"
fi

# Step 4: Rename package directory
echo "📁 Renaming package directory..."
if [ -d "reachy_mini_app_example" ]; then
    mv "reachy_mini_app_example" "$PACKAGE_NAME"
    echo -e "${GREEN}✓ Package directory renamed to $PACKAGE_NAME${NC}"
elif [ -d "reachy_mini_new_app_tuto" ]; then
    mv "reachy_mini_new_app_tuto" "$PACKAGE_NAME"
    echo -e "${GREEN}✓ Package directory renamed to $PACKAGE_NAME${NC}"
else
    echo -e "${YELLOW}Warning: No package directory found to rename${NC}"
fi

# Step 5: Update main.py class name
echo "📝 Updating main.py class name..."
if [ -f "$PACKAGE_NAME/main.py" ]; then
    # Update class name (use PascalCase class name)
    sed -i.bak "s/class ExampleApp/class ${CLASS_NAME}App/g" "$PACKAGE_NAME/main.py"
    sed -i.bak "s/class ReachyMiniNewAppTuto/class ${CLASS_NAME}App/g" "$PACKAGE_NAME/main.py"
    
    echo -e "${GREEN}✓ main.py class name updated to ${CLASS_NAME}App${NC}"
else
    echo -e "${YELLOW}Warning: main.py not found in $PACKAGE_NAME directory${NC}"
fi

# Step 6: Clean up backup files
echo "🧹 Cleaning up backup files..."
find . -name "*.bak" -delete
echo -e "${GREEN}✓ Backup files cleaned up${NC}"

# Step 7: Final summary
echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo "=================================="
echo "Your new Reachy Mini app '$REPO_NAME' is ready!"
echo ""
echo "Generated names:"
echo "  📦 Package: $PACKAGE_NAME"
echo "  🏷️  Class: ${CLASS_NAME}App"
echo "  🔧 Entry point: $ENTRY_POINT_NAME"
echo ""
echo "Next steps:"
echo "1. Review the updated files"
echo "2. Install dependencies: pip install -e ."
echo "3. Test your app: $ENTRY_POINT_NAME"
echo "4. Commit and push to your repository"
echo ""
echo -e "${YELLOW}Happy coding with Reachy Mini! 🤖${NC}"