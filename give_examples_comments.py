import json
import codecs
import msvcrt

def load_comments():
    comments = []
    with open('data/blitz_comments.json', 'r', encoding="utf-8") as f:
        comments = json.load(f)
    return comments

def save_comments(comments):
    json_object = json.dumps(list(comments), indent=4, ensure_ascii=False)
    with codecs.open("data/blitz_comments.json", "w", "utf-8") as outfile:
        outfile.write(json_object)

def main():
    comments = load_comments()
    for comment in comments:
        if 'label' not in comment or comment['label'] == 'n' or 'examples' in comment:
            continue
        print(comment['comment'])
        string: str = input()
        if string == 'n':
            comment['label'] = 'n'
            continue
        if string == 'c':
            comment['clear'] = True
            continue
        if string == 'e':
            break
        comment['examples'] = []
        while len(string) > 0:
            comment['examples'].append(string)
            string: str = input()
    save_comments([c for c in comments if 'clear' not in c])

if __name__ == "__main__":
    main()