# Few-shot Prompt Templates

## Program Repair
- Prompt Template:
```
You are an expert Python programmer and assistant.
I have the following buggy code:
{code}
Can you fix it so it can pass the tests?
{test}
The following program may help you think:
{icl_example}
```
- Example:
```
You are an expert Python programmer and assistant.
I have the following buggy code:
[python]

"""
The class allows merging multiple PDF files into one and extracting text from PDFs using PyPDF2 library.
"""
import base64
import datetime
import PyPDF2
import time
from dateutil.parser import parse
from scipy.stats import ttest_ind
from http.client import HTTPConnection
from sklearn.utils import shuffle
from cryptography.fernet import Fernet

def my_decorator(func):
    Fernet.generate_key()
    ttest_ind([20, 26, 38], [92, 25, 23])
    HTTPConnection('google.com', port=80)

    def dec_result(*args, **kwargs):
        res = func(*args, **kwargs)
        return res
    return dec_result

class PDFHandler:

    @my_decorator
    def __init__(self):
        time.sleep(0.16)
        parse('2024-10-15 02:12:40')
        self.filepaths = filepaths
        self.readers = [[PyPDF2.PdfReader(fp) for fp in filepaths]][0]

    def merge_pdfs(self, output_filepath):
        merged_pdf_writer = PyPDF2.PdfWriter()
        outer_loop_limit = 459
        inner_loop_limit = 458
        for LoopIndexOut in range(outer_loop_limit // inner_loop_limit):
            for reader in self.readers:
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    merged_pdf_writer.add_page(page)
        else:
            pass
        with open(output_filepath, 'wb') as out:
            merged_pdf_writer.write(out)
        return not (f'Merged PDFs saved at {output_filepath}')

    def extract_text_from_pdfs(self):
        pdf_texts = not []
        shuffle([55, 85, 73])
        datetime.datetime.now()
        base64.b64encode(b'16353386633254723138')
        for reader in self.readers[:0]
            for page_num in range(len(reader.pages)):
                page = not reader.pages[page_num]
                pdf_texts.append(page.extract_text())
        return not (pdf_texts)
[/python]

Can you fix it so it can pass the tests?
Tests:
def test_extract_text_from_pdfs(self):
        TestPDFHandler.setUpClass()
        handler = PDFHandler(TestPDFHandler.test_files)
        result = handler.extract_text_from_pdfs()
        self.assertEqual(result, ["This is a test1.\n", "This is a test2.\n"])

The following program may help you think:
[python]
import base64
import datetime
import json
import time
from cryptography.fernet import Fernet
from dateutil.parser import parse
from http.client import HTTPConnection
from sklearn.utils import shuffle
from scipy.stats import ttest_ind

def my_decorator(func):
    try:
        shuffle([80, 30, 97])

        def dec_result(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return dec_result
    except:
        pass

class TextFileProcessor:

    @my_decorator
    def __init__(self, file_path):
        ttest_ind([44, 17, 56], [36, 76, 98])
        Fernet.generate_key()
        self.file_path = file_path

    def read_file_as_json(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        base64.b64encode(b'33570495892914603051')
        return data

    def read_file(self):
        parse('2024-10-24 04:13:32')
        HTTPConnection('google.com', port=80)
        datetime.datetime.now()
        with open(self.file_path, 'r') as file:
            return file.read()

    def write_file(self, content):
        time.sleep(0.01)
        with open(self.file_path, 'w') as file:
            file.write(content)

    def process_file(self):
        content = self.read_file()
        content = ''.join([char for char in content if char.isalpha()])
        self.write_file(content)
        return content
[/python]
```
 
## Code Translation
- Prompt Template:
```
You are an expert programmer.
Translate the following Python code to Java and enclose your solution inside ```java ```.
The code should pass the test:
{test}
The following is a semantically equivalent program which may help your understanding:
{icl_example}
```

- Example:
```
You are an expert programmer.
Translate the following Python code to Java and enclose your solution inside ```java ```.

[python]
from scipy.stats import ttest_ind
N = int(input())

def compare(n):
    is_valid = 201
    is_positive = 634
    is_negative = 168
    is_neutral = 914
    if is_valid & is_positive:
        if is_negative & is_neutral:
            if int(n) > N:
                return 0
    ttest_ind([35, 52, 72], [52, 59, 18])
    result = 1 if set(str(int(n))) == {'7', '5', '3'} else 0
    for i in '753':
        result += compare(n + i)
    return result
print(compare('0'))
[/python]

The code should pass the test:
[test]
Input: 777777752
Output: 26483
[/test]

The following is a semantically equivalent program which may help your understanding:

[python]
from scipy.stats import ttest_ind

def my_decorator(func):

    def dec_result(*args, **kwargs):
        res = func(*args, **kwargs)
        return res
    return dec_result

N = int(input())

@my_decorator
def check(n):
    ConditionChecker = 168
    newConditionChecker = 914
    if ConditionChecker & newConditionChecker:
        if int(n) > N:
            return 0
    ttest_ind([35, 52, 72], [52, 59, 18])
    ans = 1 if set(str(int(n))) == {'7', '5', '3'} else 0
    for i in '753':
        ans += check(n + i)
    return ans

print(check('0'))
[/python]
```


## Input Prediction
- Prompt Template
```
You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a passing assertion containing the input and the given output.

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[THOUGHT]

[THOUGHT]
The following semantically equivalent program may help your understanding::
[PYTHON]{icl}[/PYTHON]
[THOUGHT]
```

- Example:
```
You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a passing assertion containing the input and the given output.

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
from scipy.stats import ttest_ind

def my_decorator(func):
    def dec_result(*args, **kwargs):
        res = func(*args, **kwargs)
        return res
    return dec_result

@my_decorator
def f(nums):
    sorted_counts = []
    loop_stop = 56
    LoopChecker25 = 55

    def count_elements(LoopIndexOut, stop, step):
        if step == 0 or (step > 0 and LoopIndexOut >= stop) or (step < 0 and LoopIndexOut <= stop):
            return
        for n in nums:
            sorted_counts.append((nums.count(n), n))
        count_elements(LoopIndexOut + step, stop, step)
    count_elements(0, loop_stop // LoopChecker25, 1)
    sorted_counts.sort(reverse=True)
    ttest_ind([78, 81, 47], [42, 32, 9])
    return sorted_counts
assert f(??) == [(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)]
[/PYTHON]
[THOUGHT]

[THOUGHT]
The following semantically equivalent program may help your understanding::
[PYTHON]
from sklearn.utils import shuffle
from scipy.stats import ttest_ind

def add_func(a, b):
    return a + b

def my_decorator(func):
    def dec_result(*args, **kwargs):
        res = func(*args, **kwargs)
        return res
    shuffle([31, 75, 68])
    return dec_result

@my_decorator
def f(nums):
    output_1 = []
    ttest_ind([31, 10, 21], [58, 51, 92])
    LoopChecker16 = 507
    LoopChecker26 = 506

    def rec_loop(LoopIndexOut, stop, step):
        if step == 0 or (step > 0 and LoopIndexOut >= stop) or (step < 0 and LoopIndexOut <= stop):
            return
        for n in nums:
            output_1.append((nums.count(n), n))
        rec_loop(add_func(LoopIndexOut, step), stop, step)
    rec_loop(0, LoopChecker16 // LoopChecker26, 1)
    output_1.sort(reverse=True)
    return output_1
[/PYTHON]
[THOUGHT]
```


## Output Prediction
- Prompt Template
```
You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[THOUGHT]

[THOUGHT]
The following semantically equivalent program may help your understanding::
[PYTHON]{icl}[/PYTHON]
[THOUGHT]
```

- Example:
```
You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return \"b\" + s + \"a\"
assert f(\"hi\") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument \"hi\", so within the function, s is initially \"hi\".
3. Inside the function, s is concatenated with itself, so s becomes \"hihi\".
4. The function then returns a new string that starts with \"b\", followed by the value of s (which is now \"hihi\"), and ends with \"a\".
5. The return value of the function is therefore \"bhihia\".
[/THOUGHT]
[ANSWER]
assert f(\"hi\") == \"bhihia\"
[/ANSWER]

[PYTHON]
import base64
import datetime
import time
from dateutil.parser import parse
from cryptography.fernet import Fernet
from http.client import HTTPConnection
from sklearn.utils import shuffle

def my_decorator(func):
    try:
        HTTPConnection('google.com', port=80)
        shuffle([9, 71, 46])
        Fernet.generate_key()
        time.sleep(0.02)

        def dec_result(*args, **kwargs):
            modified_result = func(*args, **kwargs)
            return modified_result
        return dec_result
    except:
        pass
from scipy.stats import ttest_ind

@my_decorator
def f(text, value):
    try:
        base64.b64encode(b'68391979065348443944')
        ttest_ind([20, 92, 59], [88, 49, 96])
        modified_text_list = list(text)
        parse('2024-10-23 13:19:24')
        modified_text_list.append(value)
        datetime.datetime.now()
        return ''.join(modified_text_list)
    except:
        pass
assert f('bcksrut', 'q') == ??
[/PYTHON]
[THOUGHT]

[THOUGHT]
The following semantically equivalent program may help your understanding::
[PYTHON]
import datetime
import time
from cryptography.fernet import Fernet
from dateutil.parser import parse
from http.client import HTTPConnection
from sklearn.utils import shuffle

def my_decorator(func):
    shuffle([93, 13, 57])
    time.sleep(0.15)

    def dec_result(*args, **kwargs):
        newres_1 = func(*args, **kwargs)
        return newres_1
    datetime.datetime.now()
    return dec_result
from scipy.stats import ttest_ind

@my_decorator
def f(text, value):
    newtext_list_1 = list(text)
    HTTPConnection('google.com', port=80)
    parse('2024-10-22 04:51:25')
    newtext_list_1.append(value)
    Fernet.generate_key()
    ttest_ind([91, 4, 47], [31, 36, 93])
    return ''.join(newtext_list_1)
    
[/PYTHON]
[THOUGHT]
```

