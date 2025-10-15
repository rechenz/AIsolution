from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
import time
import csv
import json
service = EdgeService()
options = Options()  # 先声明一个options变量
# 隐藏自动化控制标头
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option(
    'excludeSwitches', ['enable-automation'])  # 隐藏自动化标头
options.add_argument('--ignore-ssl-errosr')  # 忽略ssl错误
options.add_argument('--ignore-certificate-errors')  # 忽略证书错误
# prefs = {
#     'download.default_directory': '文件夹路径',  # 设置文件默认下载路径
#     "profile.default_content_setting_values.automatic_downloads": True  # 允许多文件下载
# }
# options.add_experimental_option("prefs", prefs)  # 将prefs字典传入options
browser = webdriver.ChromiumEdge(options)

browser.get('https://www.zhihu.com/topic/19554298/top-answers')
# input()
# cookies = browser.get_cookies()
# with open("AIsolution/work2/zhihuwork/cookies.json", "w") as file:
#     json.dump(cookies, file)
with open("AIsolution/work2/zhihuwork/cookies.json", "r") as file:
    cookies = json.load(file)
    for cookie in cookies:
        browser.add_cookie(cookie)
browser.refresh()
with open("AIsolution/work2/zhihuwork/zhihu.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["问题名", "问题具体内容", "回答信息"])
    for num in range(2, 22):
        curelement = browser.find_element(
            By.XPATH, f'//*[@id="TopicMain"]/div[4]/div/div/div/div[{num}]/div/div/h2/div/a')
        #              //*[@id="TopicMain"]/div[4]/div/div/div/div[3]/div/div/h2/div/a
        curelement.click()
