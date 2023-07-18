build:
	docker build --tag ocr:0.1 .

run:
	docker run --rm -d -p 8010:8000 --name ocr ocr:0.1