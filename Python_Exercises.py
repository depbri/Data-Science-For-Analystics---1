# Getting help
help(5)
dir(5)
abs.__doc__

# Syntax
myvar = 3
myvar += 2
myvar
myvar -= 1
myvar
mystring = "Hello"
mystring += " world."
print mystring
myvar, mystring = mystring, myvar

# Data types
sample = [1, ["another", "list"], ("a", "tuple")]
mylist = ["List item 1", 2, 3.14]
mylist[0] = "List item 1 again" # We're changing the item.
mylist[-1] = 3.21 # Here, we refer to the last item.
mydict = {"Key 1": "Value 1", 2: 3, "pi": 3.14}
mydict["pi"] = 3.15 # This is how you change dictionary values.
mytuple = (1, 2, 3)
myfunction = len
print myfunction(mylist)
mylist = ["List item 1", 2, 3.14]
print mylist[:]
print mylist[0:2]
print mylist[-3:-1]
print mylist[1:]
print mylist[::2]

# String
strString = """This is
a multiline
string."""
print "This %(verb)s a %(noun)s." % {"noun": "test", "verb": "is"}

# Flow control statements
rangelist = range(10)
print rangelist
for number in rangelist:
    if number in (3, 4, 7, 9):
        break
    else:
        continue
if rangelist[1] == 1:
    print "The second item (lists are 0-based) is 1"
elif rangelist[1] == 3:
    print "The second item (lists are 0-based) is 3"
else:
    print "Dunno"
while rangelist[1] == 1:
    pass

# Functions
funcvar = lambda x: x + 1
print funcvar(1)
def funcvar(x):
    return(x + 1)
def passing_example(a_list, an_int=2, a_string="A default string"):
    a_list.append("A new item")
    an_int = 4
    return a_list, an_int, a_string

my_list = [1, 2, 3]
my_int = 10
print passing_example(my_list, my_int)
my_list
my_int

# Classes
class MyClass(object):
    common = 10
    def __init__(self):
        self.myvariable = 3
    def myfunction(self, arg1, arg2):
        return self.myvariable
classinstance = MyClass()
classinstance.myfunction(1, 2)
classinstance2 = MyClass()
classinstance.common
classinstance2.common
MyClass.common = 30
classinstance.common
classinstance2.common
classinstance.common = 10
classinstance.common
classinstance2.common
MyClass.common = 50
classinstance.common
classinstance2.common
class OtherClass(MyClass):
    def __init__(self, arg1):
        self.myvariable = 3
        print arg1
classinstance = OtherClass("hello")
classinstance.myfunction(1, 2)
classinstance.test = 10
classinstance.test

# Exceptions
def some_function():
    try:
        10 / 0
    except ZeroDivisionError:
        print "Oops, invalid."
    else: # Exception didn't occur, we're good.
        pass
    finally: # This is executed after the code block is run and all exceptions have been handled, even if a new exception is raised while handling.
        print "We're done with that."
some_function()

# Importing
import random
from time import clock
randomint = random.randint(1, 100)
print randomint

# File I/O
import pickle
mylist = ["This", "is", 4, 13327]
myfile = open(r"C:\\binary.dat", "w") # Open the file C:\\binary.dat for writing. The letter r before the filename string is used to prevent backslash escaping.
pickle.dump(mylist, myfile)
myfile.close()
myfile = open(r"C:\\text.txt", "w")
myfile.write("This is a sample string")
myfile.close()
myfile = open(r"C:\\text.txt")
print myfile.read()
myfile.close()
myfile = open(r"C:\\binary.dat")
loadedlist = pickle.load(myfile)
myfile.close()
print loadedlist

# Miscellaneous
lst1 = [1, 2, 3]
lst2 = [3, 4, 5]
print [x * y for x in lst1 for y in lst2]
print [x for x in lst1 if 4 > x > 1]
any([i % 3 for i in [3, 3, 4, 4, 3]])
sum(1 for i in [3, 3, 4, 4, 3] if i == 4)
del lst1[0]
print lst1
del lst1
number = 5
def myfunc():
    print number
def anotherfunc():
    print number
def yetanotherfunc():
    global number
    number = 3
