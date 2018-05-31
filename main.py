from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import numpy as np


def game_start(driver):
    start_button = driver.find_element_by_id('button')
    start_button.click()


def game_over(driver):
    ids = driver.find_elements(By.ID, 'game-over')
    if len(ids) > 0:
        return True

    return False


def main():
    options = ChromeOptions()
    # ヘッドレスモードを有効にする（次の行をコメントアウトすると画面が表示される）。
    # options.add_argument('--headless')
    # ChromeのWebDriverオブジェクトを作成する。
    driver = Chrome(options=options)
    
    # Googleのトップ画面を開く。
    driver.get('http://localhost:1234/')

    # game start
    game_start(driver)

    try:
        # action
        while True:
            action = np.random.choice([Keys.LEFT, Keys.RIGHT, Keys.UP, Keys.DOWN])
            driver.find_element_by_tag_name('body').send_keys(action)

            if game_over(driver):
                game_start(driver)

    except KeyboardInterrupt:
        driver.quit()  # ブラウザーを終了する。

if __name__ == '__main__':
    main()
