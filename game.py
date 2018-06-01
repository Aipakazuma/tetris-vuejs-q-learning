from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import numpy as np


class Game():
    def __init__(self):
        self.name = 'tetris'
        options = ChromeOptions()
        # ヘッドレスモードを有効にする（次の行をコメントアウトすると画面が表示される）。
        # options.add_argument('--headless')
        # ChromeのWebDriverオブジェクトを作成する。
        self.driver = Chrome(options=options)
        
        # Googleのトップ画面を開く。
        self.driver.get('http://localhost:1234/')
        self.enable_actions = [Keys.LEFT, Keys.RIGHT, Keys.UP, Keys.DOWN]

    def reset(self):
        self.game_start()

    def step(self, action):
        self.driver.find_element_by_tag_name('body').send_keys(action)

        block_texts = self.driver.find_elements_by_class_name('block-text')
        states = []
        for block_text in block_texts:
            states.append(block_text.get_attribute('data-value'))

        reward = int(self.driver.find_element_by_id('point').text)
        game_over = self.game_over()
        if game_over:
            reward -= 1

        return states, reward, game_over

    def game_start(self):
        start_button = self.driver.find_element_by_id('button')
        start_button.click()

    def game_over(self):
        ids = self.driver.find_elements(By.ID, 'game-over')
        if len(ids) > 0:
            return True

        return False


def main(episode=10):
    game = Game()
    _, reward = game.reset()

    # gameの進行
    try:
        for e in range(episode):
            print('start episode: {}'.format(e))
            rewards = 0
            # action
            while True:
                action = np.random.choice(Game.enable_actions)
                _, reward = game.step(action)
                rewards += reward
                
                if game.game_over():
                    print('end episode: {}, reward: {}'.format(e, rewards))
                    break

    except KeyboardInterrupt:
        driver.quit()  # ブラウザーを終了する。

if __name__ == '__main__':
    main()
