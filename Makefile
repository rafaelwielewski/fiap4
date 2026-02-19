# Install pyenv if not installed
install-pyenv:
	@command -v pyenv >/dev/null 2>&1 || curl https://pyenv.run | bash

# First project setup command - install pyenv, python 3.12.0, create venv, install dev dependencies
setup-project:install-pyenv
	pyenv versions | grep 3.12.0 || pyenv install 3.12.0
	pyenv local 3.12.0
	python3 --version
	pip install --upgrade pip
	pip install poetry
	poetry install

# Install all dependencies including training extras
setup-training:install
	poetry install --extras training

# Install dependencies locally
install:
	poetry install

pre-commit:
	@poetry run pre-commit run --all-files

test:
	@echo "Running tests"
	@poetry run pytest --disable-warnings -vv --cov-report=html --cov .

# Start dev server locally on port 8081
dev:
	poetry run uvicorn api.main:app --host 0.0.0.0 --port 8081 --reload

# Train the LSTM model
train:
	poetry run python scripts/train_model.py

deploy-prod:
	vercel --prod

deploy-dev:
	vercel

# Start Streamlit monitoring dashboard
dashboard:
	poetry run streamlit run api/dashboard.py --server.port 8501 --server.address localhost

# Start both API and dashboard
start-all:
	@echo "🚀 Starting API and Dashboard..."
	@echo "📊 API: http://localhost:8081"
	@echo "📈 Dashboard: http://localhost:8501"
	@echo "📚 Documentation: http://localhost:8081/docs"
	@echo ""
	@echo "Press Ctrl+C to stop both services"
	@trap 'kill 0' SIGINT; ENV=LOCAL poetry run uvicorn api.main:app --host 0.0.0.0 --port 8081 --reload & poetry run streamlit run api/dashboard.py --server.port 8501 --server.address localhost & wait

# Setup project with logs directory
setup: install
	@mkdir -p logs
	@mkdir -p data
	@echo "✅ Setup complete! Run 'make dev' to start the API"
