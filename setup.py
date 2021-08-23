import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("setup.cfg", "r", encoding="utf-8") as scfg:
    for line in scfg:
        if line.startswith("version = "):
            version = line.split()[-1]
            break

setuptools.setup(
    name="waltlabtools",
    version=version,
    author="Tyler Dougan",
    author_email="author@example.com",
    description="A collection of tools for biomedical research assay analysis in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tylerdougan/waltlabtools",
    project_urls={
        "Bug Tracker": "https://github.com/tylerdougan/waltlabtools/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
