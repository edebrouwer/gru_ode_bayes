import setuptools

setuptools.setup(
    name="gru_ode_bayes",
    version="0.0.1",
    author="Anonymous",
    author_email="author@example.com",
    description="GRU ODE package",
    long_description="Some long description",
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    install_requires = ["numpy","pandas","sklearn","tensorflow","torch","argparse","tqdm","matplotlib"])
