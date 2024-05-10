format:
	black .

dataset:
	cd Datasets && kaggle competitions download -c feedback-prize-english-language-learning && unzip -q *.zip

create-env:
	conda env create -f environment.yaml

export-env:
	conda env export --from-history > environment.yaml
