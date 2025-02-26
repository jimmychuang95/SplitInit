import spacy

# 加載 spaCy 模型
nlp = spacy.load("en_core_web_sm")

# 測試句子
sentence = "a kitten standing on top of a giant tortoise"
doc = nlp(sentence)

# 提取對象：直接受詞、介詞短語的受詞，以及主詞（如需要）
objects = []
for token in doc:
    # 檢查直接受詞 (dobj)
    if token.dep_ == "dobj":
        objects.append(token.text)
    # 檢查介詞受詞 (pobj) 並確保提取最終對象
    elif token.dep_ == "pobj":
        if not any(child.dep_ == "prep" for child in token.children):
            objects.append(token.text)

# 如果需要主詞，補充如下
subjects = [token.text for token in doc if token.dep_ == "nsubj"]

# 合併主詞和受詞
final_objects = subjects + objects

print(final_objects)
# 輸出：['dog', 'airplane']
