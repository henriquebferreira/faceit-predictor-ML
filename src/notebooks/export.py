import os
from glob import glob
import nbformat
from nbconvert import HTMLExporter


def export_notebooks():
    for nb_file_path in glob(f"*/notebooks/*.ipynb"):
        nb_file = open(nb_file_path).read()

        nb = nbformat.reads(nb_file, as_version=4)
        html_exporter = HTMLExporter()

        (body, _) = html_exporter.from_notebook_node(nb)

        report_file_path = nb_file_path.replace("notebooks", "reports")
        report_file_path = report_file_path.replace(".ipynb", ".html")
        with open(report_file_path, mode='w', encoding='utf8', errors='ignore') as new_html_file:
            new_html_file.write(body)


if __name__ == '__main__':
    export_notebooks()
