install:
	@echo "Installing..."
	pip install -U .

uninstall:
	@echo "Uninstalling..."
	pip uninstall anime_recommender

clean:
	@echo "Cleaning..."
	rm -rf build dist *.egg-info