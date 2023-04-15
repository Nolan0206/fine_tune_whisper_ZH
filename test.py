import sys
from pathlib import Path
import json

if __name__ == '__main__':
    def else_test():
        s = input('请输入除数:')
        result = 20 / int(s)
        print('20除以%s的结果是: %g' % (s, result))


    def right_main():
        try:
            print('try块的代码，没有异常')
        except:
            print('程序出现异常')
            # 将else_test放在else块中
        else_test()


    def wrong_main():
        try:
            print('try块的代码，没有异常')
            # 将else_test放在try块代码的后面
            else_test()
        except:
            print('程序出现异常')


    wrong_main()
    right_main()


    path_str = Path(r"/home/nolan/ProjectSet/PycharmProjects/Community/NLP/whisper")
    for n in path_str.glob('*'):
        print(n)
    dasdsa = path_str.glob('*')
    print("===========")
    try:
        next(dasdsa)
        for n in dasdsa:
            print(n)
        next(dasdsa)
        for n in dasdsa:
            print(n)
        sys.exit()
    except StopIteration:
        print("123")
        sys.exit()

    for n in path_str.glob('*.json'):
        with open(n, "r", encoding='utf-8') as f:
            json_file = json.load(f)
            print(json_file)
    json_file_name = ['2313', '213213', '2314', '124', '12342']
    dataset_name_list = ['1', '11', '2223']
    record_list = {
        "raw": json_file_name,
        "revise": dataset_name_list
    }

    with open("write_jsonadsa.json", "w", encoding='utf-8') as f:
        # json_file = open('data.json', 'w', encoding='utf-8')
        json.dump(record_list, f, indent=2, sort_keys=True, ensure_ascii=False)

    # json_file.write(list_json)
    with open("write_jsonadsa.json", "r", encoding='utf-8') as f:
        json_file = json.load(f)
        print(json_file)

'''
    path_str = Path(r"/home/nolan/ProjectSet/PycharmProjects/Community/NLP/whisper")
    print(path_str.glob('*.py'))
    print(list(path_str.glob('*.py')))
    asd = []
    asdd = []
    for n in path_str.glob('*.py'):
        print(str(n))
        asd.append(str(n))
    print(isinstance(asd[1], str))
    for n in asdd:
        print(1)
'''

'''
    def gen(list1):
        for m in list1:
            yield m
    for m in gen([1, 2, 3, 4]):
        print('\t\t>>%s' % m)

        import csv
'''
'''
    json_file_name = ['2313','213213','2314','124','12342']
    dataset_name_list = ['1','11','2223']
    record_list = {
        "raw": json_file_name,
        "revise": dataset_name_list
    }
    import json

    with open("write_jsonaaa.json", "w", encoding='utf-8') as f:
    #json_file = open('data.json', 'w', encoding='utf-8')
        json.dump(record_list, f, indent=2, sort_keys=True, ensure_ascii=False)

    #json_file.write(list_json)
    with open("write_jsonaaa.json", "r", encoding='utf-8') as f:
        json_file = json.load(f)
        print(json_file)
'''
'''
    print(isinstance(3, int))
    print(isinstance(3.3, float))
    print(isinstance(3.3, str))
    print(isinstance(3.3, (str, float)))
'''
