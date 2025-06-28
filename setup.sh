#!/bin/bash
echo "======================================================="
echo "🚀 Setting up Fenix RAG"
echo "======================================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file and add your GOOGLE_API_KEY"
    echo "   You can get it from: https://makersuite.google.com/app/apikey"
fi

# Install dependencies
echo "Installing dependencies with uv..."
uv sync

# Start database
echo "Starting PostgreSQL with pgvector..."
docker-compose up -d

# Wait for database
echo "⏳ Waiting for database to be ready..."
sleep 15

# Check if API key is set
if ! grep -q "your_google_api_key_here" .env; then
    echo "🔧 Running setup and demo..."
    uv run fenix --setup
else
    echo "  Please set your GOOGLE_API_KEY in .env file first!"
    echo "   Then run: uv run fenix --setup"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Available commands:"
echo "  uv run fenix --help        # Show all options"
echo "  uv run fenix --setup       # Initialize and load sample data"
echo "  uv run fenix --search      # Interactive search mode"
echo "  uv run fenix --rag         # RAG Q&A mode"
echo "  uv run python tests  # Run tests"
