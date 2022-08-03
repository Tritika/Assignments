# Summary of different python programs
"""
### 1. Function to generate factors of a number
def factors(a):
    
    for x in range(1,a+1):
        if a%x== 0:
            print(x)
            
n = int(input('Give me factors of : '))
F = factors(n)

print(F)


### 2. Digital root of a number
def sum_of_digits(a):
    sum = 0
    while a > 0:
        d = a % 10
        sum = sum + d
        a = a // 10
   
    
    if(sum>=10):
       return sum_of_digits(sum)
    else:
        return sum
    
    
a = int(input(" Enter a number with more than 1 digit: "))

print("The digital root of ", a, " is ", sum_of_digits(a))

### 3. Recursive function for factorial and binomial
def fact(n):
    
   if n == 1:
       return n
   else:
       return n*fact(n-1)
  
def binom(n,k):
    if n == k:
        return n
    elif n < k:
        print("n must be greater than k.")
    else:
        return (fact(n)/(fact(k) * fact(n-k)))
    
n = int(input("Enter an integer: "))
k = int(input("Enter an integer: "))

print("\n")
print("The factorial of", n, "is", fact(n))
print()
print("Binom function returned for ", n, " and ", k, "is ", binom(n, k))


### 4. Find a character in given string input
s = input("Enter a string: ")
c = input("Enter a character you wish to find: ")


def findall(s,c):
        b=[]
        l = len(s)
        for i in range(0,l):
            if s[i] == c:
                b.append(i)
        return b
    

a = findall(s,c)

print("Given character is appeared in the following indices: ")
print(a)

### 5. Sum of cubes of n numbers
num = int(input("Enter a positive integer: "))

def sumall(n):
    if n < 0:
        print("You have entered a negative integer.")
    elif n == 1:
        return n
    else:
        return n**3 + sumall((n-1))

sum = sumall(num)  
print( "Sum of the cubes of integers is ", sum)

### 6. Printing numbers divisible by both 4 and 6 within a given range
L = range(2000, 3001)
f = list(filter((lambda x: x%4 == 0 and x%6 != 0), L))
print(f)

### 7. Generating prime numbers using List Comprehension
n = int(input("Give me prime numbers between 0 and : "))
print(list(set(range(2,n)) - {x for x in range(n) for y in range(2,x) if x%y == 0}))

### 8. Returning sequence of numbers using generating function 
def gen_fun(i, last, step):
    sum = i
    while sum <= last:
        yield sum
        sum = sum + step

print("Using Generating Function :")
        
seq_1 = gen_fun(5, 122, 3)
for x in seq_1:
    print(x, end = " ")


seq_2 = gen_fun(12, 215, 7)
print("\n")
for x in seq_2:
    print(x, end = " ")
    
string = "Ritika"
rev_s = ''

### 9. Reversing a string and checking for palindrome
def reverseiter(s):
    L = [x for x in s]
    i = len(L) - 1
    
    while i >= 0:
        yield L[i]
        i = i-1
    
items = reverseiter(string)

for x in items:
    print(x)
    rev_s = rev_s + x
    
print("\n")
print("Given string : ", string)
print("Reversed string : ", rev_s)

if string == rev_s:
    print("Palindrome it is!")
else:
    print("Try something else!")
    

### 10. Binet's formula
import numpy as np

a = (1 + np.sqrt(5))/2
b = (1 - np.sqrt(5))/2

i = int(input("Enter an integer value: "))

n = np.arange(i)

# Generates floating point values
Fn = ((a**n) - (b**n))/ np.sqrt(5)

# Converting floating values to integer for clarification
print(Fn.astype(int))
 
### 11. Masking of array 
import numpy as np

x = np.random.randint(0, 201, 50)
print(" Array generated: ", x)

print("\n")

arr = np.ma.array(x, mask = (x % 4 != 0))
print("Masked Array: ", arr)

### 12. Visualizing distribution of marks of a class using matplotlib
import numpy as np
import matplotlib.pyplot as plt

x = np.array(np.random.randint(0, 101, size = 45))

a = [i for i in x if (i >= 0 and i <= 20)]
b = [i for i in x if (i >= 21 and i <= 40)]
c = [i for i in x if (i >= 41 and i <= 60)]
d = [i for i in x if (i >= 61 and i <= 80)]
e = [i for i in x if (i >=  81 and i <= 101)]

y = [len(a), len(b), len(c), len(d), len(e)]
x = ["0-20", "21-40", "41-60", "61-80", "81-100"]

plt.bar(x,y)
plt.xlabel('Marks scored')
plt.ylabel('No. of students')
plt.title('Marks Distribution of a class of students')


### 13. Gram-Schmidt Process

import numpy as np

# (i) u1 = (3, 0, −6), u2 = (−4, 1, 7), u3 = (−2, 1, 5)

u1 = np.array([3, 0, -6])
u2 = np.array([-4, 1, 7])
u3 = np.array([-2, 1, 5])

A = np.array([u1, u2, u3])

# Checking for linear dependency of given matrix before applying Gram-Schmidt Orthogonalization 

if np.linalg.det(A)==0:
    print("Since the vectors are Linearly Dependent, we cannot use Gram-Schmidt process on A.")

else:
    v1 = u1
    v2 = u2-(np.dot(v1,u2)/np.dot(v1,v1))*v1
    v3 = u3-(np.dot(v1,u3)/np.dot(v1,v1))*v1-(np.dot(v2,u3)/np.dot(v2,v2))*v2
    
    V = np.array([v1,v2,v3])
    
    # Normalizing the orthogonalized vectors 
    w1 = v1/np.linalg.norm(v1)
    w2 = v2/np.linalg.norm(v2)
    w3 = v3/np.linalg.norm(v3)
    
    W = np.array([w1,w2,w3])
    
    print("(i)")
    print("Given matrix: \n", A)
    print("\n")
    print("Orthogonalized matrix: \n", V)
    print("\n")
    print("Ortho-normal matrix: \n", W)
    print("\n")


### 14. Integrals - scipy
import numpy as np
import scipy.integrate as si


fa = lambda x:np.exp(-x**2)
f1 = si.quad(fa, -(np.inf), np.inf)
print(f1)

f2 = si.quad(fa, 0, np.inf)
print(f2)

fc = lambda x: (x**4)*((1 - x)**4)/(1 + x**4)
f3 = si.quad(fc, 0, 1)
print(f3)

fd = lambda y, x: 1
l4 = lambda x: x**2
u4 = lambda x: 2*x
f4 = si.dblquad(fd, 0, 2, l4, u4)
print(f4)

fe = lambda y, x: (x**2)*np.exp(x*y)
l5 = lambda x: x/2
f5 = si.dblquad(fe, 0, 2, l5, 1)
print(f5)

ff = lambda y, x: x*np.exp(x**2)
f6 = si.dblquad(ff, 0, np.sqrt(2), 0, lambda x: 2 - x**2)
print(f6)

fg = lambda y, x: 6 - 2*x - 3*y
f7 = si.dblquad(fg, 0, 3, 0, lambda x: 2 - ((2*x)/3))
print(f7)

fh = lambda y, x: 2*x + 5*y
f8 = si.dblquad(fh, -2, 0, 0, lambda x: (x+2)**2)
print(f8)

fi = lambda x, y: x*y 
f9 = si.dblquad(fi, 0, 0.5, 0, lambda y: 1 - 2*y)
print(f9)

fj = lambda x, y: y* np.sin(x) + x*np.cos(y)
f10 = si.dblquad(fj, -np.pi, 2*np.pi, 0, np.pi)
print(f10)

fk = lambda z, y, x: x*y*z
f11 = si.tplquad(fk, 1, 2, 2, 3, 0, 1)
print(f11)

fl = lambda c, b, a: a**b - c
f12 = si.tplquad(fl, 0,1,0,1,0,1)
print(f12)

### 15. Catlan Numbers
import numpy as np
import scipy.integrate as si

for n in range (0,10):
    func = lambda x: (x**n)*(np.sqrt((4-x)/x))
    
    
    In, err = si.quad(func, 0, 4)
    
    Cat_num = In/(2*np.pi)
          
    print(round(Cat_num))
