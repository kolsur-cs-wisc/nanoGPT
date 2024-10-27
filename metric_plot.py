import matplotlib.pyplot as plt

tokens = [5643, 28218, 56437, 282189, 564378, 1410947, 2821894, 5079409]
kl = [0.431358, 0.417665, 0.390727, 0.384812, 0.360561, 0.341253, 0.339637, 0.33464]
p = [21.1367, 21.0673, 20.3980, 20.2999, 19.9957, 19.9869, 19.9768, 19.25215]

plt.plot(tokens, kl, label='Training Loss', color='blue')
plt.xlabel('Number of Training Characters/Tokens [log Scale]')
plt.ylabel('KL Divergence')
plt.xscale('log')
plt.grid()
plt.title('Specific Evaluation Metric')
plt.savefig('kl.png')
plt.clf()


plt.plot(tokens, p, label='Validation Loss', color='red')
plt.xlabel('Number of Training Characters/Tokens [log Scale]')
plt.ylabel('Perplexity')
plt.xscale('log')
plt.grid()
plt.title('General Evaluation Metric')
plt.savefig('perplex.png')