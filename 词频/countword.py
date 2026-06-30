import string
def word_from_txt(filename):
    with open(filename,'r',encoding="utf-8") as f:
        for line in f:
            line=line.lower()  
            for ch in string.punctuation:
                line=line.replace(ch," ")
            for w in line.split():
                yield w


def count_word(filename):
    freq={}
    for w in word_from_txt(filename):
        freq[w]=freq.get(w,0)+1

    result=sorted(freq.items(),key=lambda x:x[1],reverse=True)
    return result

if __name__=='__main__':
    filename=input("请输入文档名字")
    result=count_word(filename)
    for word,count in result:
        print(f"{word}:{count}")






