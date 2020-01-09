class Int_validation:
    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        if isinstance(value, int) and 0 < value < 100:
            self.value = value  # 这个要注意 要用value，不能用instance 否则会陷入死循环
        else:
            print("请输入合法的数字")

    def __delete__(self, instance):
        pass


class A:
    def __init__(self):
        self.a = Int_validation()
        self.b = Int_validation()


A = A()
A.a = 1
print(A.a)
print(A.b)