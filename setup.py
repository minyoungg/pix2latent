import setuptools


setuptools.setup(
    name="pix2latent",
    version="0.0.1",
    author="Minyoung Huh",
    author_email="minhuh@mit.edu",
    description="Framework for inverting generative models.",
    url="https://github.com/minyoungg/pix2latent",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
