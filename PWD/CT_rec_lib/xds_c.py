#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/13 15:19
# @Author  : wanglinghang
# @File    : xds_c.py
import json
import os

import requests

ip = '192.168.11.102'
host = "http://{}:2001/".format(ip)


def select(has_tag=None, tag_eq=None, no_tag=None):
    if has_tag is None:
        has_tag = []
    if tag_eq is None:
        tag_eq = {}
    if no_tag is None:
        no_tag = []
    para = {
        "has_tag": has_tag,
        "tag_eq": tag_eq,
        "no_tag": no_tag
    }
    res = requests.post(url="{}select".format(host), json=para)
    infos = json.loads(res.text)
    return infos


def get_tag(uid, tag):
    p = {'uid': uid, 'tag': tag}
    res = requests.get(url="{}get_tag".format(host), params=p)
    return res.text


def update_help(tag, info):
    p = {'tag': tag, "info": info}
    res = requests.get(url="{}update_help".format(host), params=p)
    return res.text


def get_help(tag=None):
    if tag is None:
        p = {'tag': 'None'}
        helps = requests.get(url="{}get_help".format(host), params=p)
        helps = json.loads(helps.text)
        res = ''
        for h in helps:
            res += "{}: {}\n".format(h, helps[h])
    else:
        p = {'tag': tag}
        res = requests.get(url="{}get_help".format(host), params=p)
        res = res.text
    return res


def get_path(uid, tag):
    p = {'uid': uid, 'tag': tag}
    res = requests.get(url="{}path".format(host), params=p)
    data_path = res.text
    data_path = data_path.replace('/data/', '//{}/share/'.format(ip))
    data_path = data_path.replace('/', os.sep)
    return data_path


def set_tag(uid, tag, value):
    p = {'uid': uid, 'tag': tag, 'value': value}
    res = requests.get(url="{}set_tag".format(host), params=p)
    if res.text != 'success':
        print(res.text)


def save_file(uid, tag, src_file, file_name):
    if file_name is None:
        file_name = src_file.split(os.sep)[-1]
    p = {'uid': uid, 'tag': tag, 'file_name': file_name}
    with open(src_file, 'rb') as f:
        res = requests.post(url="{}save_file".format(host), data=f, params=p)
    if res.text != 'success':
        print("上传失败！！！")


def save_file_to_tag(uid, tag, src_file, file_name):
    if file_name is None:
        file_name = src_file.split(os.sep)[-1]
    p = {'uid': uid, 'tag': tag, 'file_name': file_name}
    with open(src_file, 'rb') as f:
        res = requests.post(url="{}save_file_to_tag".format(host), data=f, params=p)
    if res.text != 'success':
        print("上传失败！！！")


def delete_tag(uid, tag):
    p = {'uid': uid, 'tag': tag}
    res = requests.get(url="{}delete_tag".format(host), params=p)
    print(res.text)


def delete(uid):
    p = {'uid': uid}
    res = requests.get(url="{}delete".format(host), params=p)
    print(res.text)


if __name__ == '__main__':
    infos = select(tag_eq={"device": "Jirox"})
    print(len(infos))
    # info = xds_c.get_help()
    # print()
    # print(1)
    # xds_c.update_help('Intelligent_reading_img', 'deep care 扒出来的数据')

    # for key in info:
    #     print("{}:{}".format(key, info[key]))

    # h = get_help()
    # for key in h:
    #     print("{}:{}".format(key, h[key]))
    # infos = select(['panor_to_disease'])
    # db = XDataset('/data/disk6/ai_data')
    # file_num = []
    # for i in infos:
    #     l = len(os.listdir(db.path(i['uid'], 'panor_to_disease')))
    #     if l == 8:
    #         print(i['uid'])
    #         break
    #     if l not in file_num:
    #         file_num.append(l)
    # print(file_num)

    # print(get_help('test'))
    # save_file('1.2.156.89797079.1516317.1178856356.2532189334', 'test',
    #                  r'D:\2022data\manual_label\1\old\check\0000___0.png', '0.png')
    # delete_tag('123', '456')
