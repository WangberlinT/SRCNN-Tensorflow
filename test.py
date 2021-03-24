import matplotlib.pyplot as plt

loss = range(10)
plt.title("SRCNN Train")
plt.xlabel("epoch")
plt.ylabel("Loss - MSE")
plt.plot(range(10), loss)
plt.savefig("./train_loss.png")