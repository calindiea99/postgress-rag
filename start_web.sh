#!/bin/bash
# Startup script for the Text Ingestion Web Interface

echo "🚀 Starting Text Ingestion Web Interface..."
echo "=========================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Flask not found. Installing web interface dependencies..."
    pip install -r web_requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

# Check if main ingestion dependencies are available
python -c "import langchain_community" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Main ingestion dependencies not found. Installing..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install main dependencies"
        exit 1
    fi
fi

echo "✅ Dependencies check complete"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p static/css
mkdir -p static/js
mkdir -p templates

echo "🌐 Starting web server..."
echo "Visit: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

# Start the web interface
python web_interface.py
