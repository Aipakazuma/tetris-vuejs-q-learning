from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from chromedriver_binary.utils import get_chromedriver_path
import numpy as np
from time import time
from copy import copy
from settings import GAME_URL
import sys


class Game():
    def __init__(self, game_url=None):
        self.name = 'tetris'
        options = Options()
        # ヘッドレスモードを有効にする（次の行をコメントアウトすると画面が表示される）。
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        # ChromeのWebDriverオブジェクトを作成する。
        self.driver = Chrome(chrome_options=options)
        
        # Googleのトップ画面を開く。
        self.driver.get(GAME_URL if game_url is None else game_url)
        self.enable_actions = [Keys.LEFT, Keys.RIGHT, Keys.UP, Keys.DOWN]
        self.time = time()
        self.before_state = None

    def reset(self):
        self.game_start()
        return None, None

    def step(self, action):
        self.driver.find_element_by_tag_name('body').send_keys(action)

        block_text = self.driver.find_element_by_class_name('block-text')
        states = [int(i) for i in block_text.get_attribute('data-value').split(',')]

        reward = int(self.driver.find_element_by_id('point').text)
        s = np.array(states).reshape(20, 10)
        under_state = s[-1]
        if self.before_state is None:
            self.before_state = copy(under_state)
            a = self.before_state
        else:
            a = np.clip(under_state - self.before_state, 0, 1)
            self.before_state = copy(under_state)

        reward += np.sum(a) * 0.5
        game_over = self.game_over()
        if game_over:
            reward -= 3
            self.before_state = None

        return states, reward, game_over, {}

    def game_start(self):
        start_button = self.driver.find_element_by_id('button')
        start_button.click()

    def game_over(self):
        ids = self.driver.find_elements(By.ID, 'game-over')
        if len(ids) > 0:
            return True

        return False


def main(episode=10, game_url=None):
    game = Game(game_url=game_url)

    # gameの進行
    try:
        for e in range(episode):
            _, reward = game.reset()
            print('start episode: {}'.format(e))
            rewards = 0
            # action
            while True:
                action = np.random.choice(game.enable_actions)
                states, reward, done, _ = game.step(action)
                rewards += reward
                print(np.array(states).reshape(20, 10), reward)
                
                if done:
                    print('end episode: {}, reward: {}'.format(e, rewards, reward, done))
                    break

    except KeyboardInterrupt:
        driver.quit()  # ブラウザーを終了する。

if __name__ == '__main__':
    game_url = sys.argv[1]
    main(game_url=game_url)
