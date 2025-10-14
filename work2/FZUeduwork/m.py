import requests
from bs4 import BeautifulSoup
import csv
import re
import lxml
url = "https://jwch.fzu.edu.cn/jxtz.htm"
response = requests.get(url)
response.encoding = 'utf-8'
content = response.text
soup = BeautifulSoup(content, 'lxml')
li_find = soup.find_all('li')
with open('fzu.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['通知人', '标题', '日期', '链接'])
    for li in li_find:
        annoucer = re.search("【(.*?)】", li.text)
        if annoucer:
            annoucer = annoucer.group(1)
        else:
            annoucer = ""
        temp = li.find('a')
        title = None
        date = None
        href = None
        if temp:
            title = temp.string
            href = temp.get('href')
        temp = li.find('span')
        if temp:
            date = temp.string
        writer.writerow([annoucer, title, date, href])
