import time

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from secret import EMAIL, PASSWORD

DEBUG = True

SCROLL_SCRIPT = "window.scrollTo(0, document.body.scrollHeight);"
HEIGHT_SCRIPT = "return document.body.scrollHeight"

FRIEND_CLASSES = "x193iq5w xeuugli x13faqbe x1vvkbs x10flsy6 x1lliihq x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x1tu3fi x3x7a5m x1lkfr7t x1lbecb7 x1s688f xzsf02u"
POST_CLASSES =  "xdj266r x11i5rnm xat24cr x1mh8g0r x1vvkbs x126k92a"


def create_driver() -> Chrome:
    options = Options()
    if not DEBUG:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option(
        "prefs", {"profile.default_content_setting_values.notifications": 2}
    )
    driver = Chrome(service=Service(ChromeDriverManager().install()), options=options)

    return driver


def sign_into_facebook(driver: Chrome) -> None:
    driver.get("https://facebook.com/")

    current_url = driver.current_url

    email_box = driver.find_element(By.ID, "email")
    password_box = driver.find_element(By.ID, "pass")
    email_box.send_keys(EMAIL)
    password_box.send_keys(PASSWORD + Keys.ENTER)

    WebDriverWait(driver, 10).until(EC.url_changes(current_url))


def classes_to_xpath(classes: str) -> str:
    return (
        "//*["
        + " and ".join(
            f"contains(concat(' ', @class, ' '), ' {c} ')" for c in classes.split(" ")
        )
        + "]"
    )


def webscrape_facebook_friends(url: str) -> list[str]:
    friends_url = url + "&sk=friends"

    driver = create_driver()
    sign_into_facebook(driver)

    driver.get(friends_url)

    for _ in range(10):
        driver.execute_script(SCROLL_SCRIPT)
        time.sleep(0.5)

    friend_xpath = classes_to_xpath(FRIEND_CLASSES)
    friend_elements = driver.find_elements(By.XPATH, friend_xpath)
    friend_list = [e.text for e in friend_elements if e.text]

    driver.close()

    return friend_list


def webscrape_friend_posts(url: str, friend_name: str) -> list[str]:
    friends_url = url + "&sk=friends"

    driver = create_driver()
    sign_into_facebook(driver)

    driver.get(friends_url)

    friend_xpath = f"//*[contains(text(), '{friend_name}')]"

    while True:
        if driver.find_elements(By.XPATH, friend_xpath):
            break

        driver.execute_script(SCROLL_SCRIPT)
        time.sleep(0.5)

    friend_element = driver.find_element(By.XPATH, friend_xpath)
    current_url = driver.current_url
    friend_element.click()
    WebDriverWait(driver, 10).until(EC.url_changes(current_url))

    last_height = driver.execute_script(HEIGHT_SCRIPT)
    while True:
        driver.execute_script(SCROLL_SCRIPT)
        time.sleep(0.5)

        new_height = driver.execute_script(HEIGHT_SCRIPT)
        if new_height == last_height:
            break
        last_height = new_height

    post_xpath = classes_to_xpath(POST_CLASSES)
    post_elements = driver.find_elements(By.XPATH, post_xpath)
    post_contents = [e.text for e in post_elements if not e.text.isspace()]

    driver.close()

    return post_contents
