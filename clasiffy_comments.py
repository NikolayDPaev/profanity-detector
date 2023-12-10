import json
import codecs
import msvcrt

def load_comments():
    comments = []
    with open('data/blitz_comments_classified.json', 'r', encoding="utf-8") as f:
        comments = json.load(f)
    return comments

def save_comments(comments):
    json_object = json.dumps(list(comments), indent=4, ensure_ascii=False)
    with codecs.open("data/blitz_comments_classified.json", "w", "utf-8") as outfile:
        outfile.write(json_object)

def main():
    comments = load_comments()
    for comment in comments:
        if len(comment) == 3:
            continue
        print(comment[0])
        ch: str = msvcrt.getch().decode("utf-8")
        if ch == 'p':
            comment.append('p')
        if ch == 'n':
            comment.append('n')
        if ch == 'c':
            comment.clear()
        if ch == 'e':
            break
    save_comments([c for c in comments if len(c) > 0])

if __name__ == "__main__":
    main()