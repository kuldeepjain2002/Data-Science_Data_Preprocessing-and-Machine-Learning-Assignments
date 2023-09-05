def fac(n):
    if n==0:
        return 1
    else:
        a =1
        for i in range(1,n+1):
            a = a*i
        return a

n = int(input("No. of founders"))
m = int(input("No. of board members"))

ncm= fac(n)/(fac(n-m))
print(1/ncm)


