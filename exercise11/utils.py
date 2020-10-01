#!/usr/bin/python
# -*- coding: UTF-8 -*-
def print_line(text):
    total=90;
    lenTxt = len(text)
    lenTxt_utf8 = len(text.encode('utf-8'))
    size = int((lenTxt_utf8 - lenTxt) / 2 + lenTxt)

    remainder=(total - size)%2
    left= (total - size)//2
    right=left if remainder==0 else left+1
    print("*" * left+text+"*" * right)