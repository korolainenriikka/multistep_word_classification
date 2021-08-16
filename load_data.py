from lxml import etree
import urllib.request
import mlflow

# Returns a list of Finnish words
def load_finnish():
    finnish_url="https://www.cs.helsinki.fi/u/jttoivon/dap/data/kotus-sanalista_v1/kotus-sanalista_v1.xml"
    filename="data/kotus-sanalista_v1.xml"
    load_from_net=False
    if load_from_net:
        with urllib.request.urlopen(finnish_url) as data:
            lines=[]
            for line in data:
                lines.append(line.decode('utf-8'))
        doc="".join(lines)
    else:
        with open(filename, "rb") as data:
            doc=data.read()
    tree = etree.XML(doc)
    s_elements = tree.xpath('/kotus-sanalista/st/s')
    return list(map(lambda s: s.text, s_elements))


def load_english():
    with open("data/words", encoding="utf-8") as data:
        lines=map(lambda s: s.rstrip(), data.readlines())
    return list(lines)


def main():
    with mlflow.start_run():
        mlflow.set_tag('step-name', 'load data')

        finnish = load_finnish()
        english = load_english()

        with open('finnish_raw.txt', 'w') as file:
            for line in finnish:
                file.write(line + '\n')

        with open('english_raw.txt', 'w') as file:
            for line in english:
                file.write(line + '\n')

        print('Uploading word lists')
        mlflow.log_artifact('finnish_raw.txt', "finnish-list-raw")
        mlflow.log_artifact('english_raw.txt', "english-list-raw")


if __name__ == "__main__":
    main()
