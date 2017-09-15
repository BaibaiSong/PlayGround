# -*- coding:utf-8 -*-
__author__ = 'Song'
import urllib.request
import re

def craw(url):
    # headers = ("User-Agent", "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    #                          "Chrome/60.0.3112.113 Safari/537.36")
    # opener = urllib.request.build_opener()
    # opener.addheaders = [headers]
    #
    # urllib.request.install_opener(opener)
    #
    # data = urllib.request.urlopen(url).read()
    #
    # # print(data)
    # with open("Data/daxue/2.html", "wb") as file:
    #     file.write(data)

    with open("Data/daxue/2.html", "rb") as file:
        data = str(file.read())
    # print(data)
    part1 = "/uploads/allimg/.*?\.jpg"
    part2 = "\'>\d+?</a>"

    imagelist = re.compile(part1).findall(data)
    print(imagelist)
    print(len(imagelist))

    good1 = re.compile(part2).findall(data)
    print(good1)
    print(len(good1))
    return data


if __name__ == '__main__':
    # url = "http://www.xiaohua100.cn/daxue/"
    url = "http://xiaohua100.cn/plus/waterfall.php?tid=1&sort=lastpost&totalresult=856&pageno=4"
    craw(url)
