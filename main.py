from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import numpy as np


class QLearning():
    def __init__(self, shape, gamma=0.99):
        self.q_table = np.zeros(shape)
        self.gamma = 

    def update(self):
        pass


class Agent():
    def __init__(self, eps=1e-2):
        # [state, action]
        self.model = QLearning(shape=[20, 10, 4])
        self.eps = eps

    def action(self, next_state):
        if self.eps < np.random.uniform(0, 1):
            next_action = np.argmax(self.model.q_table[next_state])
        else:
            next_action = np.random.randint(0, 4)

        return next_action

    def update(self):
        self.model.update()


class Game():
    def __init__(self):
        options = ChromeOptions()
        # ヘッドレスモードを有効にする（次の行をコメントアウトすると画面が表示される）。
        # options.add_argument('--headless')
        # ChromeのWebDriverオブジェクトを作成する。
        self.driver = Chrome(options=options)
        
        # Googleのトップ画面を開く。
        self.driver.get('http://localhost:1234/')
        self.actions = [Keys.LEFT, Keys.RIGHT, Keys.UP, Keys.DOWN]

    def reset(self):
        self.game_start()
        return np.zeros([20, 10]), 0

    def step(self, action):
        a = self.actions[action]
        driver.find_element_by_tag_name('body').send_keys(a)
        
        states = np.zeros([20, 10])
        reward = 0

        return states, reward

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
    agent = Agents()

    state, reward = game.reset()

    # gameの進行
    try:
        for e in range(episode):
            print('start episode: {}'.format(e))
            rewards = 0
            # action
            while True:
                action = agent.action(state)
                next_state, reward = game.step(action)
                rewards += reward
                
                if game.game_over():
                    print('end episode: {}, reward: {}'.format(e, rewards))
                    break

                # update
                agents.update()
                state = next_state


    except KeyboardInterrupt:
        driver.quit()  # ブラウザーを終了する。

if __name__ == '__main__':
    main()
