from bs4 import BeautifulSoup
from glob import glob

STYLESHEET_HREF = "https://ghcdn.rawgit.org/henriquebferreira/faceit-predictor-ML/master/src/notebooks/style.css"


def style_reports():
    for report in glob(f"*/reports/*.html"):
        with open(report, encoding='utf8', errors='ignore') as html_file:
            soup = BeautifulSoup(html_file.read(), features='html.parser')

            # Remove all style tags
            for s in soup.select('style'):
                s.extract()

            # Remove existent stylesheets
            for s in soup.findAll("link", {"href": STYLESHEET_HREF}):
                s.extract()

            stylesheet_tag = soup.new_tag(
                "link", rel="stylesheet", href=STYLESHEET_HREF)
            soup.title.insert_after(stylesheet_tag)

            new_text = soup.prettify()

        # Write new contents to test.html
        with open(report, mode='w') as new_html_file:
            new_html_file.write(new_text)


if __name__ == '__main__':
    style_reports()
