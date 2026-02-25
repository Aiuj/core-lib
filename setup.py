from setuptools import setup, find_packages

setup(
    name='core-lib',
    version='0.3.5',
    description='Shared library for MCP agent tools (internal use only)',
    author='Julien Nadaud',
    author_email='nad.julien@gmail.com',
    url='https://github.com/Aiuj/core-lib',
    packages=find_packages(),
    install_requires=[
        'fast-langdetect>=1.0.0',
        'fastmcp>=2.10.6',
        'fastapi>=0.120.2',
        'google-genai>=0.6.0',
        'ollama>=0.3.1',
        'langfuse>=3.2.1',
        'openinference-instrumentation-google-genai>=0.1.5',
        'redis>=6.4.0',
        'valkey>=6.0.0',
        'openai>=1.41.0',
        'openpyxl>=3.1.5',
        'markdown>=3.8.2',
        'tabulate>=0.9.0',
        'numpy>=2.3.3',
        'pyjwt>=2.10.1',

    ],
    python_requires='>=3.12',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    license='MIT',
)
