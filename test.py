def mutable_parameter(lst=list()):
    lst.append(1)
    lst.append(2)
    return lst

print(mutable_parameter())
print(mutable_parameter())
print(mutable_parameter())

