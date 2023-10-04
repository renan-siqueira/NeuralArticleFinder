from sklearn.datasets import fetch_20newsgroups


def create_newsgroupt_dataset():

    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    with open("data/raw/articles.txt", 'w', encoding='utf-8') as file:
        for item in newsgroups.data:
            file.write(f"{item}\n")


if __name__ == '__main__':
    create_newsgroupt_dataset()
