.PHONY: install train api streamlit docker test clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python -m src.training.train_flight_price
	python -m src.training.train_flight_classifier
	python -m src.training.train_recommendation

api:
	python -m api.app

streamlit:
	streamlit run streamlit_app/app.py --server.port 8501

mlflow:
	mlflow ui --port 5001

docker:
	docker-compose up --build -d

docker-down:
	docker-compose down

k8s-deploy:
	kubectl apply -f kubernetes/namespace.yaml
	kubectl apply -f kubernetes/configmap.yaml
	kubectl apply -f kubernetes/mlflow/deployment.yaml
	kubectl apply -f kubernetes/api/deployment.yaml
	kubectl apply -f kubernetes/streamlit/deployment.yaml

test:
	pytest tests/ -v --tb=short

clean:
	rm -rf outputs/models/*.pkl
	rm -rf outputs/models/*.json
	rm -rf mlruns/
	rm -rf __pycache__ **/__pycache__
