# TRAVEL ORDER RESOLVER - Makefile

.PHONY: help setup docker-build docker-up docker-down install install-speech clean data train evaluate predict test notebooks format lint voice voice-text test-stt test-tts

# Default target
.DEFAULT_GOAL := help

# Help
help:
	@echo "=========================================="
	@echo "Travel Order Resolver - Make Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup         - Complete setup (Docker + Python env)"
	@echo "  make install       - Install Python dependencies"
	@echo "  make docker-build  - Build Docker containers"
	@echo "  make docker-up     - Start all services"
	@echo "  make docker-down   - Stop all services"
	@echo ""
	@echo "Data:"
	@echo "  make data          - Download SNCF data and generate dataset"
	@echo "  make generate      - Generate training sentences"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  make train         - Train all models (baseline + BERT)"
	@echo "  make train-baseline - Train baseline model only"
	@echo "  make train-bert    - Train BERT model only"
	@echo "  make evaluate      - Evaluate models on test set"
	@echo "  make predict       - Run prediction on sample data"
	@echo ""
	@echo "Voice Module (STT + TTS):"
	@echo "  make install-speech - Install system deps (ffmpeg, portaudio)"
	@echo "  make voice         - Run voice-enabled travel resolver"
	@echo "  make voice-text    - Run with text input + voice output"
	@echo "  make test-stt      - Test speech-to-text (Whisper)"
	@echo "  make test-tts      - Test text-to-speech (pyttsx3)"
	@echo ""
	@echo "Development:"
	@echo "  make test          - Run all tests"
	@echo "  make notebooks     - Start Jupyter Lab"
	@echo "  make format        - Format code with black & isort"
	@echo "  make lint          - Lint code with flake8"
	@echo "  make clean         - Clean generated files"
	@echo ""

# Setup
setup: docker-build install
	@echo "✅ Setup complete!"

install:
	@if [ -d "venv" ]; then \
		echo "Using existing venv..."; \
		venv/bin/pip install --upgrade pip; \
		venv/bin/pip install -r requirements.txt; \
		venv/bin/python -m spacy download fr_core_news_lg; \
	else \
		pip install --upgrade pip; \
		pip install -r requirements.txt; \
		python -m spacy download fr_core_news_lg; \
	fi
	@echo "✅ Python dependencies installed"

# Docker Commands
docker-build:
	cd docker && docker-compose build
	@echo "✅ Docker containers built"

docker-up:
	cd docker && docker-compose up -d
	@echo "✅ Services started:"
	@echo "   - Neo4j: http://localhost:7474"
	@echo "   - Jupyter: http://localhost:8888"
	@echo "   - MLflow: http://localhost:5000"

docker-down:
	cd docker && docker-compose down
	@echo "✅ Services stopped"

docker-logs:
	cd docker && docker-compose logs -f

docker-shell:
	cd docker && docker-compose exec nlp /bin/bash

# Data
data: download-sncf generate
	@echo "✅ Data preparation complete"

download-sncf:
	python scripts/download_sncf_data.py
	@echo "✅ SNCF data downloaded"

generate:
	python scripts/generate_dataset.py
	@echo "✅ Training dataset generated"

# Training
train: train-baseline train-bert
	@echo "✅ All models trained"

train-baseline:
	python scripts/train.py --model baseline
	@echo "✅ Baseline model trained"

train-bert:
	python scripts/train.py --model bert
	@echo "✅ BERT model trained"

# Evaluation
evaluate:
	python scripts/evaluate.py --all
	@echo "✅ Evaluation complete - see outputs/evaluation/"

evaluate-baseline:
	python scripts/evaluate.py --model baseline

evaluate-bert:
	python scripts/evaluate.py --model bert

# Prediction
predict:
	python scripts/predict.py --input data/test_samples.csv --output outputs/predictions.csv
	@echo "✅ Predictions saved to outputs/predictions.csv"

# Development
test:
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ Tests complete - see htmlcov/index.html"

test-fast:
	pytest tests/ -v -x

notebooks:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Code Quality
format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/
	@echo "✅ Code formatted"

lint:
	flake8 src/ scripts/ tests/ --max-line-length=120
	mypy src/ --ignore-missing-imports
	@echo "✅ Linting complete"

# Cleanup

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "✅ Cleaned up"

clean-all: clean
	rm -rf models/
	rm -rf outputs/
	rm -rf logs/
	rm -rf data/processed/
	rm -rf data/training/
	rm -rf data/datasets/
	@echo "✅ Deep clean complete"

# Database
neo4j-reset:
	cd docker && docker-compose stop neo4j
	cd docker && docker-compose rm -f neo4j
	cd docker && docker volume rm docker_neo4j-data
	cd docker && docker-compose up -d neo4j
	@echo "✅ Neo4j reset complete"

neo4j-import:
	python scripts/import_to_neo4j.py
	@echo "✅ SNCF data imported to Neo4j"

# Monitoring
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000

monitor:
	watch -n 2 'docker stats --no-stream'

# Python executable (use venv if available)
PYTHON := $(shell if [ -d "venv" ]; then echo "venv/bin/python"; else echo "python"; fi)

# Voice Module
install-speech:
	@echo "Installing speech dependencies..."
	@if command -v apt-get >/dev/null 2>&1; then \
		echo "Detected Debian/Ubuntu - installing with apt-get"; \
		sudo apt-get update && sudo apt-get install -y ffmpeg portaudio19-dev espeak-ng; \
	elif command -v pacman >/dev/null 2>&1; then \
		echo "Detected Arch Linux - installing with pacman"; \
		sudo pacman -S --needed --noconfirm ffmpeg portaudio espeak-ng; \
	elif command -v brew >/dev/null 2>&1; then \
		echo "Detected macOS - installing with homebrew"; \
		brew install ffmpeg portaudio; \
	else \
		echo "❌ Unsupported OS. Please install ffmpeg and portaudio manually."; \
		exit 1; \
	fi
	@echo "✅ Speech system dependencies installed"

voice:
	PYGAME_HIDE_SUPPORT_PROMPT=1 SDL_AUDIODRIVER=dsp $(PYTHON) scripts/travel_resolver_voice.py 2>/dev/null
	@echo "✅ Voice session ended"

voice-text:
	$(PYTHON) scripts/travel_resolver_voice.py --text
	@echo "✅ Voice-text session ended"

voice-small:
	$(PYTHON) scripts/travel_resolver_voice.py --model-size small
	@echo "✅ Voice session (small model) ended"

test-stt:
	@echo "Testing Speech-to-Text (Whisper)..."
	@echo "Dites quelque chose dans 3 secondes..."
	@$(PYTHON) -c "from src.speech.transcriber import SpeechTranscriber; t = SpeechTranscriber('base'); t.calibrate(1); print('Parlez maintenant...'); result = t.listen(timeout=10); print(f'\nTranscription: {result}')"

test-tts:
	@echo "Testing Text-to-Speech (pyttsx3)..."
	@$(PYTHON) -c "from src.speech.synthesizer import SpeechSynthesizer; s = SpeechSynthesizer(); s.speak('Bonjour, ceci est un test du module de synthese vocale en francais.'); print('✅ TTS test complete')"

demo-voice:
	@echo "Demo: Complete voice pipeline"
	@echo "1. Whisper model will load..."
	@echo "2. Microphone calibration..."
	@echo "3. Speak a travel request in French"
	@echo ""
	$(PYTHON) scripts/travel_resolver_voice.py --model-size tiny
