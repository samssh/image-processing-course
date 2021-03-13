s = input()
words = s.split()
result = []
c = 0
for t in words:
    a = True
    for j in result:
        if j.lower() == t.lower():
            a = False
    if a:
        result.append(t)
    else:
        c += 1
print(*result, sep=" ")
print(format(c * 100 / (len(words)), ".0f"))
