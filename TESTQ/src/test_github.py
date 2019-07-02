from github import Github
import datetime
import eml_parser
from os import listdir
from os.path import isfile, join


def json_serial(obj):
    if isinstance(obj, datetime.datetime):
        serial = obj.isoformat()
        return serial


def link_breaker(text):
    """deletes any link in http*"""
    res = ''
    state = 0
    for char in text:
        if state == 0:
            if char == 'h':
                state = 1
            else:
                res += char
        elif state == 1:
            if char == 't':
                state = 2
            else:
                res += 'h' + char
                state = 0
        elif state == 2:
            if char == 't':
                state = 3
            else:
                res += 'ht' + char
                state = 0
        elif state == 3:
            if char == 'p':
                state = 4
            else:
                res += 'htt' + char
                state = 0
        elif state == 4:
            if char == ' ':
                state = 0
                res += 'ext_link '
    return res


def main():
    """Will send any unchecked message to github"""
    verbose = False
    online = True

    if online:
        TOKEN = ""
        g = Github(base_url="https://github.ibm.com/api/v3", login_or_token=TOKEN)
        repo = g.get_repo("Raphael-Lambert/test_note")

    path = "C:/Users/RaphaelLambert/Documents/git_issues"
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    if verbose:
        print(onlyfiles)
    treated = []
    issues = []

    with open(join(path, 'log.txt'), 'r') as doc:
        for line in doc:
            treated.append(line.rstrip('\n'))

    with open(join(path, 'issues.txt'), 'r') as doc:
        for line in doc:
            issues.append(int(line.rstrip('\n')))

    for title in onlyfiles:
        if title != 'log.txt' and title != 'issues.txt' and title not in treated:
            with open(join(path, title), 'rb') as fhdl:
                raw_email = fhdl.read()

            parsed_eml = eml_parser.eml_parser.decode_email_b(raw_email, include_raw_body=True)
            if verbose:
                print('-----------------')
                print(title)
                print('-----------------')
                print(parsed_eml)
                print('-----------------')
            body = parsed_eml['body']
            if len(body) > 0:
                raw_text = body[0]['content']
            else:
                raw_text = "message chiffré"
            raw_text = link_breaker(raw_text)
            num_get = 0
            if online and title[:4] == 'Re  ' and title[4:] in treated:
                cont_issue = repo.get_issue(issues[treated.index(title[4:])])
                num_get = cont_issue.number
                cont_issue.create_comment(body=raw_text)
            elif online:
                new_issue = repo.create_issue(title="Conversation number {}: {}".format(len(treated), title[:10]+"..."),
                                              body=raw_text)
                if verbose:
                    print(new_issue)
                num_get = new_issue.number
            treated.append(title)
            issues.append(num_get)

    if verbose:
        print(treated)

    with open(join(path, 'log.txt'), 'w') as doc:
        for title in treated:
            doc.write(title+'\n')
    with open(join(path, 'issues.txt'), 'w') as doc:
        for title in issues:
            doc.write(str(title)+'\n')
    # doc = open(join(path, 'log.txt'), 'w')
    # doc.close()
    # doc = open(join(path, 'issues.txt'), 'w')
    # doc.close()


if __name__ == "__main__":
    main()
