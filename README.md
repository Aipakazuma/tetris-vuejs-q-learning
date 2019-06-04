## 強化学習で遊ぶ

```sh
$ pip install selenium
```


```
selenium.common.exceptions.WebDriverException: Message: unknown error: call function result missing 'value'
```

エラー

https://qiita.com/orangeboy/items/6fdddebc1dc919f6d9e1


動いた


あとでDockerfileにする.

```sh
$ wget -q -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip && \
unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/ && \
apt-get install -y libappindicator1 fonts-liberation libasound2 libnspr4 libnss3 libxss1 lsb-release xdg-utils && \
touch /etc/default/google-chrome && \
apt-get install -f -y \
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
dpkg -i google-chrome-stable_current_amd64.deb && \
apt-get install -y fonts-migmix
```
