from setuptools import setup, find_packages



setup(
    name='bsq',  # 模块名称
    version='1.0.0',
    description='tsinghua gaojingjian yinshouyi',
    author='wangbin@gaojingjian',
    package_dir={'':'src'},
    packages=find_packages('src'),
    package_data = {'':['*.py']},
)
