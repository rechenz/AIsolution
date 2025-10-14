import requests
from bs4 import BeautifulSoup
import csv
import re
url = "https://jwch.fzu.edu.cn/jxtz.htm"
response = requests.get(url)
response.encoding = 'utf-8'
content = response.text
soup = BeautifulSoup(content, 'lxml')
check = soup.find('ul', attrs={"class": "list-gl"})
li_find = None
if check != None:
    li_find = check.find_all_next('li')
else:
    print("error")
with open('AIsolution/work2/FZUeduwork/fzu.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['通知人', '标题', '日期', '链接'])
    if li_find == None:
        exit()
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
            if date == None:
                tt = temp.find('font')
                if tt:
                    date = tt.string
        href = str(href)
        href = "https://jwch.fzu.edu.cn/"+href
        writer.writerow([annoucer, title, date, href])
    # exit()
    for page in range(1, 206, 1):
        url = f"https://jwch.fzu.edu.cn/jxtz/{206-page}.htm"
        print(page)
        # writer.writerow([1, 1, 1, 1, 1])
        response = requests.get(url)
        response.encoding = 'utf-8'
        content = response.text
        soup = BeautifulSoup(content, 'lxml')
        check = soup.find('ul', attrs={"class": "list-gl"})
        li_find = None
        if check != None:
            li_find = check.find_all_next('li')
        else:
            print("error in for")
        if li_find == None:
            exit()
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
                if date == None:
                    tt = temp.find('font')
                    if tt:
                        date = tt.string
            href = str(href)
            href = str.replace(href, "../", "https://jwch.fzu.edu.cn/")
            writer.writerow([annoucer, title, date, href])
