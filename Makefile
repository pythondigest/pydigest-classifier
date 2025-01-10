RESOURCES_PATH := ./resources
RESOURCES_FOLDER =$$(realpath $(RESOURCES_PATH))
ZIPS_PATH := ./resources/zips
ZIPS_FOLDER =$$(realpath $(ZIPS_PATH))

pip-tools:
	pip install -U pip
	# pip install -U poetry
	poetry add poetry-plugin-up --group dev

install-requirements: pip-tools
	poetry install

install-check:
	poetry run pre-commit install
	poetry run pre-commit install-hooks

install-nltk:
	poetry run python -c "import nltk; nltk.download('punkt_tab')"

install: install-requirements
	echo "Done"

update: pip-tools
	poetry update
	poetry run poetry up
	poetry lock
	# poetry run pre-commit autoupdate

check:
	poetry run pre-commit run --show-diff-on-failure --color=always --all-files

run:
	poetry run python src/api/wsgi.py
	
train:
	poetry run python src/model/train.py "./resources/dataset/" "./resources/models/classifier.pkl"

report:
	poetry run python src/model/report.py "./resources/dataset/report.csv"

test:
	poetry run pytest 

download:
	rm -rf ${RESOURCES_FOLDER}/dataset
	echo ${ZIPS_FOLDER}; rclone copy yandex-pydigest:backups/pythondigest/zips/ ${ZIPS_FOLDER} --progress
	unzip ${ZIPS_FOLDER}/dataset.zip -d ${RESOURCES_FOLDER}