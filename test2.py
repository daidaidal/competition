# coding=utf-8
from selenium import webdriver
import time


# 正确答案
ans = "ABDCDBACBA CAACBBDBBB BACBBACAAC CCCCCBDDCC BABADCBDCD \
       DBACBCBDAC ABBDCCDDAC BCDDCBDACA CBBDDCAACB BAACACCDAB"


def Run(chrome, name, num, college=u"计算机科学与技术学院"):
    chrome.get("http://wmb.hljgov.com/e/tool/college/college.php ")
    chrome.find_element_by_xpath('//*[@id="optionS"]/div[1]/label[65]').click()
    chrome.find_element_by_xpath('//*[@id="uname"]').send_keys(name)
    chrome.find_element_by_xpath('//*[@id="idcard"]').send_keys(num)
    chrome.find_element_by_xpath('//*[@id="faculty"]').send_keys(college)
    chrome.find_element_by_xpath('//*[@id="information"]/div/a').click()
    num = 1  # 每一题的xpath地址里li的地址会加一 用num更改相应按钮地址

    for i in ans:
        if i == " ":  # 这是上面ans字符串里的空格分隔符 无意义
            continue
        # 生成相应题数的li地址
        A = '/html/body/div[1]/ul/li[' + str(num) + ']/div[1]\
            /div/div[2]/label[1]/span'
        B = '/html/body/div[1]/ul/li[' + str(num) + ']/div[1]\
            /div/div[2]/label[2]/span'
        C = '/html/body/div[1]/ul/li[' + str(num) + ']/div[1]\
            /div/div[2]/label[3]/span'
        D = '/html/body/div[1]/ul/li[' + str(num) + ']/div[1]\
            /div/div[2]/label[4]/span'
        # 点击答案选项
        if i == "A":
            chrome.find_element_by_xpath(A).click()
        elif i == "B":
            chrome.find_element_by_xpath(B).click()
        elif i == "C":
            chrome.find_element_by_xpath(C).click()
        else:
            chrome.find_element_by_xpath(D).click()
        # 点击确定按钮
        chrome.find_element_by_xpath('/html/body/div[1]/ul/li[' + str(num) + ']\
                                     /div[1]/button').click()
        # 点击下一题按钮
        chrome.find_element_by_xpath('/html/body/div[1]/ul/li[' + str(num) + ']\
                                     /div[2]/button').click()
        num += 1


if __name__ == "__main__":
    name1 = u'戴尚峰'
    num1 = '230604199505300212'

    chrome1 = webdriver.Chrome()
    # for i in range(20):
    #     print(20-i)
    #     time.sleep(1)
    try:
        Run(chrome1, name1, num1)
        print("搞定")
        time.sleep(60)
        chrome1.close()
    except BaseException:
        print("error")
