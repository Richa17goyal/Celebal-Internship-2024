import matplotlib.pyplot as plt
x = ([4,7,3,9,6])
y = ([9,1,5,8,4])
plt.subplot(2,1,1)
plt.bar(x,y, color = 'Navy', width= 0.4)
plt.xlabel("CLASS")
plt.ylabel("OBJECT")
plt.legend(["women"])
plt.grid(True, color = "yellow")

x = [1,7,9,4,6]
y = [7,1,3,8,4]
plt.subplot(2,1,2)
plt.bar(x,y, color = 'r', width= 0.7)
plt.xlabel("CLASS")
plt.ylabel("OBJECT")
plt.tight_layout()
plt.show()