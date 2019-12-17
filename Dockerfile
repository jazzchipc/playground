FROM python:3.6.8

COPY ./requirements.txt /
COPY ./requirements_extra.txt /

RUN pip install -r requirements.txt
RUN pip install -r requirements_extra.txt

COPY . /

CMD [ "python", "./my_training.py" ]