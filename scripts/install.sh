#!/bin/bash

# AI Crypto Trader Installation Script

set -e

echo "🚀 Installing AI Crypto Trader..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8 or higher is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs models data

# Copy environment file
if [ ! -f .env ]; then
    echo "⚙️ Creating environment file..."
    cp .env.example .env
    echo "📝 Please edit .env file with your Kraken API credentials"
else
    echo "✅ Environment file already exists"
fi

# Set permissions
echo "🔐 Setting permissions..."
chmod +x run_trader.py
chmod +x scripts/*.sh

# Run tests
echo "🧪 Running tests..."
python test_trader.py

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your Kraken API credentials"
echo "2. Test in sandbox mode: python run_trader.py analyze BTC/USD"
echo "3. Start trading: python run_trader.py start"
echo ""
echo "📚 For more information, see README.md"