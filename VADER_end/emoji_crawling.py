#!/usr/bin/env python
# coding: utf-8

# ## Emoji Recognition Chart - Korean Ver. Crawling
# 
# * url : https://www.emojiall.com/ko/copy

# In[4]:


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
import json
import re


# In[26]:


url = "https://www.emojiall.com/ko/copy"
driver = webdriver.Chrome('chromedriver')
driver.get(url)

lis = driver.find_elements(By.CSS_SELECTOR, "ul.copy_list.emoji_list.row.row-cols-lg-5.row-cols-4 > li")
driver.close()

emoji_list = []
for li in lis:
    data = li.get_attribute('data-keyword')
    data_filter = [i for i in data.split("^") if i]
    emoji_list.append(data_filter)

emoji_df = pd.DataFrame(emoji_list)
emoji_df.head()


# In[27]:


emoji_df.tail()


# In[28]:


emoji_df.dropna(axis=1, inplace=True)


# In[30]:


emoji_df.columns = ['emoji', 'korean', 'expressions']


# In[31]:


emoji_df.info()


# In[32]:


emoji_df.to_csv('emoji_korean.csv', index=False, encoding="utf-8")


# In[ ]:




