from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="generalist",
        # py_modules=["submodules.aokvqa"]
        # packages=['torchtask', 'submodules.aokvqa'],
        packages=find_packages(),
    )
