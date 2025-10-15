from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
import time
import csv
import json
import bs4
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
# exit()
with open("AIsolution/work2/zhihuwork/zhihu.csv", "w", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["问题名", "问题具体内容", "回答信息"])
    for num in range(2, 22):  # 从问题开始
        curelement = browser.find_element(
            By.XPATH, f'//*[@id="TopicMain"]/div[4]/div/div/div/div[{num}]/div/div/h2/div/a')
        browser.execute_script("arguments[0].scrollIntoView();", curelement)
        surl = curelement.get_attribute("href")
        if surl == None:
            print("error")
            break
        browser.get(surl)
        problemname = ''
        problemcontent = ''
        answercontent = ''
        li = []
        scurelement = browser.find_element(
            By.XPATH, '//*[@id="root"]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/h1')
        problemname = scurelement.text
        scurelement = browser.find_element(
            By.XPATH, '//*[@id="root"]/div/main/div/div/div[1]/div[2]/div/div[1]/div[1]/div[6]/div/div/div/button')
        scurelement.click()
        # print("pass")
        scurelement = browser.find_element(
            By.CSS_SELECTOR, '#content > span.RichText.ztext.css-oqi8p3')
        problemcontent = scurelement.text
        li.append(problemname)
        li.append(problemcontent)
        for i in range(2, 12):  # 10个回答
            scurelement = browser.find_element(
                By.XPATH, f'/html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div/div[{i}]/div/div/div[2]/span[1]/div/div/span/span[1]')
            #               /html/body/div[1]/div/main/div/div/div[3]/div[1]/div/div[1]/div/div/div/div[2]/div/div[3]/div/div/div[2]/span[1]/div/div/span/span[1]
            answercontent = scurelement.text
            li.append(answercontent)
            browser.execute_script(
                "arguments[0].scrollIntoView();", scurelement)
        writer.writerow(li)
        browser.back()
