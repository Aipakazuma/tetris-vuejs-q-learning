# 強化学習で遊ぶ

## Usage

```sh
$ docker build . -t reinforce_learning_platform 
$ docker run --rm -v $(pwd):/app -it reinforce_learning_platform python game.py http://host.docker.internal:9000/
```


## Learning

### DQN

```sh
$ docker run --rm -v $(pwd):/app -it reinforce_learning_platform python keras_rl.py http://host.docker.internal:9000/
```

### Rainbow

### Ape-x
