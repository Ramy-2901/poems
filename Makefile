theme_classifier = classifier.bin
clean_corpus = clean_corpus.csv
language_model = model.keras
word_limit = 50
history_path = history.json
training_plot = training_plot.jpeg

all: train_classifier theme_classification build_model

train_classifier: functions.py datasets
	python3 train_classifier.py $(theme_classifier)

theme_classification: functions.py $(theme_classifier) datasets
	python3 theme_classification.py $(theme_classifier) $(clean_corpus)

build_model: $(clean_corpus)
	python3 poem_generation.py $(clean_corpus) $(word_limit) $(language_model) $(history_path) $(training_plot)