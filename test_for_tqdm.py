from time import sleep

from tqdm import tqdm
from multiprocessing import Pool


def test(str1):
    # for i in tqdm(range(100), desc='Current Json', colour='green'):
    #     # print(filename)
    #     sleep(1)
    return '=='


if __name__ == '__main__':

    p = Pool(processes=8)
    # p.map(test)
    for char in tqdm(range(5), desc='Parsing Json', colour='red'):
        sleep(0.25)
        str1 = p.map(test, 'owo===')
        print(str1)
