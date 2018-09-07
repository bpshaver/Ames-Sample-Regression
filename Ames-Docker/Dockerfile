FROM python:3

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY ./src/ames-regression .

 # Run interactively
ENTRYPOINT ["python", "./main.py"]