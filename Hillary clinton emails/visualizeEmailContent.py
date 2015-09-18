# -*- coding: utf-8 -*-
# visualize using wordcloud
from os import path
from PIL import Image
import pandas as pd
import sqlite3

from wordcloud import WordCloud
from wordcloud import STOPWORDS
import numpy as np
import matplotlib.pyplot as plt


d = path.dirname(__file__)

# email contents
con = sqlite3.connect(path.join(d, "database.sqlite"))
data = pd.read_sql_query(
    "SELECT ExtractedBodyText FROM Emails WHERE ExtractedBodyText like '%President%' limit 20", con)

# content
content = []
for i in range(len(data.ExtractedBodyText)):
    content.append(data.ExtractedBodyText[i].encode('utf-8').strip())

# write content to txt file
with open("Email_Output.txt", "w") as text_file:
    for i in range(len(content)):
        text_file.write(content[i])

text = open(path.join(d, "Email_Output.txt")).read()

img = Image.open("hc.png")
img = img.resize((980, 1080), Image.ANTIALIAS)
hcmask = np.array(img)
wc = WordCloud(
    background_color="white", max_words=2000, mask=hcmask, stopwords=STOPWORDS)
wc.generate(text)

wc.to_file("wc.png")
plt.imshow(wc)
plt.axis("off")
plt.figure()
plt.imshow(hcmask, cmap=plt.cm.gray)
plt.axis("off")
plt.show()
