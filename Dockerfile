FROM continuumio/anaconda3

WORKDIR /app

RUN apt update \
  && apt install -y gnupg \
  && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
  && wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
  && apt install -y ./google-chrome-stable_current_amd64.deb \
  && apt clean \
  && rm -rf /var/lib/apt/lists/ \
  && rm google-chrome-stable_current_amd64.deb

ADD requirements.txt /app/
RUN pip install -U pip && pip install -r requirements.txt
