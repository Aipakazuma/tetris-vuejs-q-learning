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
$ docker build . -t reinforce_learning_platform 
$ docker run --rm -v $(pwd):/app -it reinforce_learning_platform bash
```
