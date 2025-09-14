#!/bin/bash

# Crypto Price Forecast - Git Deployment Script

echo "🚀 Preparing to push crypto forecast system to main..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "💾 Creating commit..."
    git commit -m "feat: Complete crypto price forecast system

- XGBoost multi-horizon regression (15/30/60 min)
- FastAPI backend with comprehensive endpoints  
- React frontend with ticker-style display
- Real-time data fetching with sample data fallback
- Advanced feature engineering (54+ features)
- Automatic fallback to sklearn HistGradientBoosting
- Production-ready with proper error handling
- Complete documentation and setup scripts"
fi

# Check if remote exists
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "🔗 Please add your remote repository:"
    echo "git remote add origin https://github.com/yourusername/Cincinnati.git"
    echo ""
    echo "Then run: git push -u origin main"
    exit 1
fi

# Push to main
echo "🚀 Pushing to main branch..."
git branch -M main
git push -u origin main

echo "✅ Successfully pushed to main!"
echo "📊 Your crypto forecast system is now live on GitHub!"