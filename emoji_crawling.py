import requests
from bs4 import BeautifulSoup
# import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
import re


url = "https://www.emojiall.com/ko/copy"
driver = webdriver.Chrome('./chromedriver')
# driver.get(url)

# lis = driver.find_elements(By.CSS_SELECTOR, "ul.copy_list.emoji_list.row row-cols-lg-5/row-cols-4 > li")


soup = BeautifulSoup(driver.page_source, 'lxml')

d = soup.find('li', class_="col emoji_item emoji_")