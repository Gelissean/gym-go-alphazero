class A():
    def funkcia(self):
        print("som v A")

class B():
    def funkcia(self):
        print("som v B")

class C(A):
    def funkcia(self):
        super(C, self).funkcia()
        print("som v C")

class D(A,B):
    def daco(self):
        print("haha daco")

class F(B, A):
    def daco(self):
        print("haha daco")

class E(A,B):
    def funkcia(self):
        print("som v E")

a = A()
b = B()
c = C()
d = D()
e = E()
f = F()

print("A")
a.funkcia()
print("B")
b.funkcia()
print("C")
c.funkcia()
print("D")
d.funkcia()
print("E")
e.funkcia()
print("F")
f.funkcia()